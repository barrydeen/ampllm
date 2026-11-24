pub mod metrics;
pub mod parser;
pub mod routes;
pub mod scheduler;
pub mod vision;

pub use scheduler::{ServerConfig, SlotPublicStatus};

use crate::metrics::ServerMetrics;
use crate::parser::{ChatOutputMode, ParsedEvent};
use crate::routes::{
    chat_completions, completions, health, list_models, list_slots, metrics, AppState, EngineJob,
    ServerModelKind, StreamEvent,
};
use crate::scheduler::{RealBackend, SlotScheduler};
use ampllm_engine::AnyEngine;
use ampllm_tokenizer::BpeTokenizer;
use anyhow::Result;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

pub(crate) fn collect_stop_ids(tokenizer: &BpeTokenizer, specials: &[&str]) -> Vec<u32> {
    let mut ids = vec![tokenizer.eos_token_id];
    for s in specials {
        if let Some(id) = tokenizer.token_id(s) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

pub(crate) fn forward_parsed_event(ev: ParsedEvent, tx: &mpsc::UnboundedSender<StreamEvent>) {
    match ev {
        ParsedEvent::Token(t) => {
            let _ = tx.send(StreamEvent::Token(t));
        }
        ParsedEvent::Reasoning(r) => {
            let _ = tx.send(StreamEvent::Reasoning(r));
        }
        ParsedEvent::ToolCall {
            index,
            id,
            name,
            arguments,
        } => {
            let _ = tx.send(StreamEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            });
        }
        ParsedEvent::ToolCallStart { index, id, name } => {
            let _ = tx.send(StreamEvent::ToolCallStart { index, id, name });
        }
        ParsedEvent::ToolCallArgsDelta { index, delta } => {
            let _ = tx.send(StreamEvent::ToolCallArgsDelta { index, delta });
        }
    }
}

/// Collect stop token IDs for GLM-5.3-Flash turn boundaries and end-of-sequence tokens.
pub fn glm5_stop_ids(tokenizer: &BpeTokenizer) -> Vec<u32> {
    collect_stop_ids(
        tokenizer,
        &["<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"],
    )
}

/// Collect stop token IDs for Hy4-Flash turn boundaries and end-of-sequence tokens.
pub fn hy4_stop_ids(tokenizer: &BpeTokenizer, hytk: &str) -> Vec<u32> {
    let hy_end = format!("<｜hy_end{hytk}｜>");
    let hy_start = format!("<｜hy_start{hytk}｜>");
    collect_stop_ids(
        tokenizer,
        &[
            hy_end.as_str(),
            hy_start.as_str(),
            "<｜end▁of▁sentence｜>",
            "<|endoftext|>",
        ],
    )
}

/// Collect stop token IDs for Qwen3.8-Flash-Next turn boundaries and end-of-sequence tokens.
pub fn qwen4_stop_ids(tokenizer: &BpeTokenizer) -> Vec<u32> {
    collect_stop_ids(tokenizer, &["<|im_end|>", "<|endoftext|>"])
}

pub fn mm3_stop_ids(tokenizer: &BpeTokenizer) -> Vec<u32> {
    collect_stop_ids(tokenizer, &["[e~[", "]~b]", "]~!b["])
}

pub(crate) fn decode_for_mode(
    tokenizer: &BpeTokenizer,
    mode: &ChatOutputMode,
    ids: &[u32],
) -> String {
    match mode {
        ChatOutputMode::Glm5 { .. } | ChatOutputMode::Qwen4 { .. } => {
            tokenizer.decode_skip_control(ids)
        }
        ChatOutputMode::Mm3 { .. } => tokenizer.decode(ids, false),
        ChatOutputMode::Hy4 { .. } => tokenizer.decode_hy4_aliased(ids),
        ChatOutputMode::Inkling2 { .. } => tokenizer.decode(ids, false),
        ChatOutputMode::Plain => tokenizer.decode(ids, true),
    }
}

pub async fn run_server(
    engine: AnyEngine,
    tokenizer: BpeTokenizer,
    model_name: String,
    addr: SocketAddr,
) -> Result<()> {
    run_server_with_config(engine, tokenizer, model_name, addr, ServerConfig::default()).await
}

pub async fn run_server_with_config(
    engine: AnyEngine,
    tokenizer: BpeTokenizer,
    model_name: String,
    addr: SocketAddr,
    config: ServerConfig,
) -> Result<()> {
    let model_kind = if engine.is_mm3() {
        ServerModelKind::Mm3
    } else if engine.is_qwen4() {
        ServerModelKind::Qwen4
    } else if engine.is_hy4() {
        ServerModelKind::Hy4
    } else if engine.is_glm5() {
        ServerModelKind::Glm5
    } else {
        ServerModelKind::Inkling2
    };
    let hy4_tag_suffix = if model_kind == ServerModelKind::Hy4 {
        tokenizer
            .hy4_tag_suffix()
            .map(|s| format!(":{s}"))
            .unwrap_or_else(|| ":opensource".to_owned())
    } else {
        String::new()
    };
    let has_vision = engine.has_vision();
    let max_ctx = engine.max_ctx();
    let server_metrics = Arc::new(ServerMetrics::default());
    let metrics_worker = Arc::clone(&server_metrics);
    let pending_jobs = Arc::new(AtomicUsize::new(0));
    let pending_worker = Arc::clone(&pending_jobs);
    let slot_statuses = Arc::new(RwLock::new(Vec::<SlotPublicStatus>::new()));
    let statuses_worker = Arc::clone(&slot_statuses);
    let (tx, rx) = mpsc::unbounded_channel::<EngineJob>();
    let hy4_tag_worker = hy4_tag_suffix.clone();

    // Dedicated inference thread owns engine + SlotScheduler (single-writer CUDA context safety)
    std::thread::Builder::new()
        .name("ampllm-infer-worker".to_owned())
        .spawn(move || {
            let backend = RealBackend::new(engine, tokenizer, model_kind, &hy4_tag_worker);
            let scheduler = SlotScheduler::new(
                backend,
                config,
                metrics_worker,
                pending_worker,
                statuses_worker,
            );
            scheduler.run(rx);
        })?;

    let state = AppState {
        model_name,
        model_kind,
        hy4_tag_suffix,
        has_vision,
        max_ctx,
        tx,
        metrics: server_metrics,
        pending_jobs,
        slot_statuses,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/slots", get(list_slots))
        .route("/models", get(list_models))
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions))
        .route("/v1/slots", get(list_slots))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("ampllm HTTP server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::parser::{ChatOutputMode, OutputParser, ParsedEvent, ToolParamSchema};
    use crate::routes::{
        extract_tool_schemas, format_glm5_chat, format_hy4_chat, format_inkling2_chat,
        format_mm3_chat, format_qwen4_chat, resolve_glm5_thinking, resolve_hy4_thinking,
        resolve_mm3_thinking, resolve_qwen4_thinking, ChatMessage,
    };
    use crate::vision::{
        decode_and_prepare_image, decode_and_prepare_mm3_image, decode_and_prepare_qwen4_image,
        expand_hy4_image_tokens, expand_mm3_image_tokens, expand_qwen4_image_tokens,
        load_image_source,
    };
    use ampllm_gguf::{Hy4VisionConfig, Mm3VisionConfig, Qwen4VisionConfig};
    use base64::Engine as _;
    use serde_json::json;

    fn user_msg(s: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_owned(),
            content: Some(json!(s)),
            ..Default::default()
        }
    }

    #[test]
    fn test_multi_turn_chat_prefix_extension() {
        let t1 = vec![user_msg("Hello")];
        let t2 = vec![
            user_msg("Hello"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: Some(json!("Hi there!")),
                ..Default::default()
            },
            user_msg("How are you?"),
        ];
        let (p1, st1) = format_inkling2_chat(&t1, None, None, None);
        let (p2, st2) = format_inkling2_chat(&t2, None, None, None);
        assert!(st1 && st2);
        assert!(
            p2.starts_with(&p1),
            "multi-turn chat prompt must strictly extend Turn 1 prefix"
        );
    }

    #[test]
    fn test_glm5_chat_template_and_streaming_parser() {
        let kw = json!({"enable_thinking": false});
        let (en, eff) = resolve_glm5_thinking(Some(&kw), None);
        assert!(!en);
        assert_eq!(eff, "Low");

        let msgs = vec![user_msg("Hi")];
        let (p_off, exp_off, pf_off, imgs_off) =
            format_glm5_chat(&msgs, None, None, false, "Low", false).unwrap();
        assert!(!exp_off);
        assert!(pf_off.is_empty());
        assert!(imgs_off.is_empty());
        assert_eq!(
            p_off,
            "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think></think>"
        );

        let (p_on, exp_on, pf_on, _) =
            format_glm5_chat(&msgs, None, None, true, "Max", false).unwrap();
        assert!(exp_on);
        assert!(pf_on.is_empty());
        assert_eq!(
            p_on,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>"
        );

        let msgs2 = vec![
            user_msg("Hi"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: Some(json!("Hello!")),
                ..Default::default()
            },
            user_msg("Next?"),
        ];
        let (p_t2, _, _, _) = format_glm5_chat(&msgs2, None, None, false, "Low", false).unwrap();
        assert!(
            p_t2.starts_with(&p_off),
            "GLM-5 multi-turn should preserve exact prefix"
        );

        let mut parser = OutputParser::new(
            ChatOutputMode::Glm5 {
                expect_think_close: true,
                parse_tools: false,
                prefill: String::new(),
                schemas: ToolParamSchema::new(),
            },
            "call_1".to_owned(),
        );
        let mut r = String::new();
        let mut c = String::new();
        for ch in ["th", "ink", "ing</", "thi", "nk>\nans", "wer"] {
            parser.push(ch, &mut |ev| match ev {
                ParsedEvent::Reasoning(s) => r.push_str(&s),
                ParsedEvent::Token(s) => c.push_str(&s),
                ParsedEvent::ToolCall { .. }
                | ParsedEvent::ToolCallStart { .. }
                | ParsedEvent::ToolCallArgsDelta { .. } => panic!("unexpected tool call"),
            });
        }
        parser.finish(&mut |ev| match ev {
            ParsedEvent::Reasoning(s) => r.push_str(&s),
            ParsedEvent::Token(s) => c.push_str(&s),
            ParsedEvent::ToolCall { .. }
            | ParsedEvent::ToolCallStart { .. }
            | ParsedEvent::ToolCallArgsDelta { .. } => panic!("unexpected tool call"),
        });
        assert_eq!(r, "thinking");
        assert_eq!(c, "\nanswer");
    }

    #[test]
    fn test_glm5_tool_prompt_and_output_parser() {
        let tools = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Multiply integers",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": { "type": "integer" },
                            "b": { "type": "integer" },
                            "tag": { "type": "string" }
                        }
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "hidden_tool",
                    "defer_loading": true,
                    "parameters": {}
                }
            }),
        ];
        let msgs = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: Some(json!([{"type": "text", "text": "Compute 7 * 6"}])),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_owned(),
                reasoning_content: Some("Need calc.".to_owned()),
                content: None,
                tool_calls: Some(vec![
                    json!({"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{\"a\":7,\"b\":6}"}}),
                    json!({"id": "c2", "type": "function", "function": {"name": "calc", "arguments": "{\"a\":2,\"b\":3}"}}),
                ]),
                ..Default::default()
            },
            // Out-of-order tool responses to verify GLM-5.3 ID reordering
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c2".to_owned()),
                content: Some(json!("6")),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c1".to_owned()),
                content: Some(json!("42")),
                ..Default::default()
            },
        ];

        let (prompt, expect_think, prefill, _) =
            format_glm5_chat(&msgs, Some(&tools), None, true, "High", false).unwrap();
        assert!(expect_think);
        assert!(prefill.is_empty());
        assert!(prompt.contains("<tools>\n{\"description\":\"Multiply integers\",\"name\":\"calc\""));
        assert!(!prompt.contains("hidden_tool"));
        assert!(!prompt.contains("\"strict\""));
        assert!(prompt.contains("<|observation|><tool_response>42</tool_response><tool_response>6</tool_response><|assistant|><think>"));

        let forced_tc = json!({"type": "function", "function": {"name": "calc"}});
        let (fp, f_think, f_prefill, _) =
            format_glm5_chat(&[user_msg("go")], Some(&tools), Some(&forced_tc), true, "Max", false)
                .unwrap();
        assert!(!f_think);
        assert_eq!(f_prefill, "<tool_call>calc");
        assert!(fp.ends_with("<|assistant|><think></think><tool_call>calc"));

        // Verify chunked streaming parser with <think>, XML tool call, code fence literal, and parallel call
        let schemas = extract_tool_schemas(Some(&tools));
        let mut p = OutputParser::new(
            ChatOutputMode::Glm5 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_9".to_owned(),
        );
        let raw = "checking</think>Here is an example:\n```xml\n<tool_call>ignored</tool_call>\n```\n<tool_call>calc\n<arg_key>a</arg_key><arg_value>7</arg_value><arg_key>b</arg_key><arg_value>6</arg_value><arg_key>tag</arg_key><arg_value>007</arg_value></tool_call><tool_call>calc<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>";
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut calls: Vec<(usize, String, String, String)> = Vec::new();
        let mut push_ev = |ev: ParsedEvent| match ev {
            ParsedEvent::Reasoning(s) => reasoning.push_str(&s),
            ParsedEvent::Token(s) => text.push_str(&s),
            ParsedEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => calls.push((index, id, name, arguments)),
            ParsedEvent::ToolCallStart { index, id, name } => {
                calls.push((index, id, name, String::new()));
            }
            ParsedEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(c) = calls.iter_mut().find(|c| c.0 == index) {
                    c.3.push_str(&delta);
                }
            }
        };
        for ch in raw.as_bytes().chunks(5) {
            let piece = std::str::from_utf8(ch).unwrap();
            p.push(piece, &mut push_ev);
        }
        p.finish(&mut push_ev);
        assert_eq!(reasoning, "checking");
        assert!(text.contains("<tool_call>ignored</tool_call>"));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1, "call_9_0");
        assert_eq!(calls[0].2, "calc");
        let a0: serde_json::Value = serde_json::from_str(&calls[0].3).unwrap();
        assert_eq!(a0, json!({"a": 7, "b": 6, "tag": "007"}));
        let a1: serde_json::Value = serde_json::from_str(&calls[1].3).unwrap();
        assert_eq!(a1, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_glm5_incremental_tool_call_streaming() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": { "type": "string" },
                        "content": { "type": "string" },
                        "line": { "type": "integer" }
                    }
                }
            }
        })];
        let schemas = extract_tool_schemas(Some(&tools));
        let mut p = OutputParser::new(
            ChatOutputMode::Glm5 {
                expect_think_close: false,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_inc".to_owned(),
        );

        let mut events = Vec::new();
        p.push("<tool_call>write_file\n<arg_key>filePath</arg_key><arg_value>", &mut |ev| {
            events.push(ev);
        });
        assert!(
            matches!(
                events.first(),
                Some(ParsedEvent::ToolCallStart { index: 0, name, .. }) if name == "write_file"
            ),
            "expected ToolCallStart before closing </tool_call>, got: {events:?}"
        );

        let deltas_before_close = |evs: &[ParsedEvent]| -> usize {
            evs.iter()
                .filter(|e| matches!(e, ParsedEvent::ToolCallArgsDelta { .. }))
                .count()
        };
        let count_0 = deltas_before_close(&events);
        p.push("src/ma", &mut |ev| events.push(ev));
        p.push("in.rs</arg_value><arg_key>content</arg_key><arg_value>", &mut |ev| {
            events.push(ev);
        });
        assert!(
            deltas_before_close(&events) > count_0,
            "expected streamed argument deltas mid-tool-call"
        );

        let count_1 = deltas_before_close(&events);
        p.push("fn main() {\n    println!(\"hi</arg", &mut |ev| events.push(ev));
        assert!(
            deltas_before_close(&events) > count_1,
            "expected long string body to stream before </arg_value>"
        );
        // Partial closing tag `</arg` must be held back, not leaked into JSON args
        let partial_args: String = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::ToolCallArgsDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !partial_args.contains("</arg"),
            "held-back tag prefix leaked: {partial_args:?}"
        );

        p.push(
            "_not_tag\\n\");\n}</arg_value><arg_key>line</arg_key><arg_value>42</arg_value></tool_call>",
            &mut |ev| events.push(ev),
        );
        p.finish(&mut |ev| events.push(ev));

        let final_args: String = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::ToolCallArgsDelta { index: 0, delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        let parsed: serde_json::Value = serde_json::from_str(&final_args).unwrap();
        assert_eq!(
            parsed,
            json!({
                "filePath": "src/main.rs",
                "content": "fn main() {\n    println!(\"hi</arg_not_tag\\n\");\n}",
                "line": 42
            })
        );
    }

    #[test]
    fn test_glm5_think_missing_close_before_tool_call() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                }
            }
        })];
        let schemas = extract_tool_schemas(Some(&tools));
        let mut p = OutputParser::new(
            ChatOutputMode::Glm5 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas: schemas.clone(),
            },
            "call_miss".to_owned(),
        );
        let raw = "Checking chromium logs,Desktop reviewed:<tool_call>read<arg_key>path</arg_key><arg_value>/tmp/a.png</arg_value></tool_call><tool_call>read<arg_key>path</arg_key><arg_value>/tmp/b.png</arg_value></tool_call>";
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut calls: Vec<(usize, String, String, String)> = Vec::new();
        let mut push_ev = |ev: ParsedEvent| match ev {
            ParsedEvent::Reasoning(s) => reasoning.push_str(&s),
            ParsedEvent::Token(s) => text.push_str(&s),
            ParsedEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => calls.push((index, id, name, arguments)),
            ParsedEvent::ToolCallStart { index, id, name } => {
                calls.push((index, id, name, String::new()));
            }
            ParsedEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(c) = calls.iter_mut().find(|c| c.0 == index) {
                    c.3.push_str(&delta);
                }
            }
        };
        for ch in raw.as_bytes().chunks(7) {
            p.push(std::str::from_utf8(ch).unwrap(), &mut push_ev);
        }
        p.finish(&mut push_ev);
        assert!(p.has_tool_calls());
        assert_eq!(reasoning, "Checking chromium logs,Desktop reviewed:");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].2, "read");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].3).unwrap(),
            json!({"path": "/tmp/a.png"})
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[1].3).unwrap(),
            json!({"path": "/tmp/b.png"})
        );

        // Prose <tool_call> mention + fenced <tool_call> inside <think> must remain reasoning
        let mut p2 = OutputParser::new(
            ChatOutputMode::Glm5 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_prose".to_owned(),
        );
        let raw2 = "I should emit a <tool_call> block, or like:\n```xml\n<tool_call>read<arg_key>path</arg_key><arg_value>/tmp/fake.png</arg_value></tool_call>\n```\nNow call it.</think><tool_call>read<arg_key>path</arg_key><arg_value>/tmp/real.png</arg_value></tool_call>";
        let mut r2 = String::new();
        let mut calls2: Vec<(usize, String, String, String)> = Vec::new();
        let mut push2 = |ev: ParsedEvent| match ev {
            ParsedEvent::Reasoning(s) => r2.push_str(&s),
            ParsedEvent::Token(_) => {}
            ParsedEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => calls2.push((index, id, name, arguments)),
            ParsedEvent::ToolCallStart { index, id, name } => {
                calls2.push((index, id, name, String::new()));
            }
            ParsedEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(c) = calls2.iter_mut().find(|c| c.0 == index) {
                    c.3.push_str(&delta);
                }
            }
        };
        for ch in raw2.as_bytes().chunks(6) {
            p2.push(std::str::from_utf8(ch).unwrap(), &mut push2);
        }
        p2.finish(&mut push2);
        assert!(r2.contains("<tool_call>read"));
        assert!(r2.ends_with("Now call it."));
        assert_eq!(calls2.len(), 1);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls2[0].3).unwrap(),
            json!({"path": "/tmp/real.png"})
        );
    }

    #[test]
    fn test_inkling2_tool_prompt_and_output_parser() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Compute",
                "parameters": {"type": "object"}
            }
        })];
        let msgs = vec![
            user_msg("6*7?"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: None,
                tool_calls: Some(vec![
                    json!({"id": "c1", "function": {"name": "calc", "arguments": "{\"b\":6,\"a\":7}"}}),
                ]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c1".to_owned()),
                content: Some(json!("42")),
                ..Default::default()
            },
        ];
        let (p, start_in_text) = format_inkling2_chat(&msgs, Some(&tools), None, Some(&json!("low")));
        assert!(!start_in_text);
        assert!(p.starts_with("<|message_system|>tool_declare<|content_xml|>"));
        assert!(p.contains("Thinking effort level: 0.2"));
        assert!(p.contains("<|message_model|>calc<|content_invoke_tool_json|>{\"name\":\"calc\",\"args\":{\"a\":7,\"b\":6}}<|end_message|><|content_model_end_sampling|>"));
        assert!(p.contains("<|message_tool|>calc<|content_text|>42<|end_message|>"));
        assert!(p.ends_with("<|message_model|>"));

        let mut parser = OutputParser::new(
            ChatOutputMode::Inkling2 {
                start_in_text: false,
                parse_tools: true,
            },
            "call_3".to_owned(),
        );
        let sample = "<|content_thinking|>pondering<|end_message|><|message_model|>calc<|content_invoke_tool_json|>{\"name\":\"calc\",\"args\":{\"a\":7,\"b\":6}}<|end_message|><|content_model_end_sampling|>";
        let mut r = String::new();
        let mut c = String::new();
        let mut calls = Vec::new();
        for ch in sample.as_bytes().chunks(7) {
            parser.push(std::str::from_utf8(ch).unwrap(), &mut |ev| match ev {
                ParsedEvent::Reasoning(s) => r.push_str(&s),
                ParsedEvent::Token(s) => c.push_str(&s),
                ParsedEvent::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => calls.push((index, id, name, arguments)),
                ParsedEvent::ToolCallStart { index, id, name } => {
                    calls.push((index, id, name, String::new()));
                }
                ParsedEvent::ToolCallArgsDelta { index, delta } => {
                    if let Some(call) = calls.iter_mut().find(|x| x.0 == index) {
                        call.3.push_str(&delta);
                    }
                }
            });
        }
        parser.finish(&mut |_| {});
        assert_eq!(r, "pondering");
        assert!(c.is_empty(), "header function name must not leak into content: {c:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "call_3_0");
        assert_eq!(calls[0].2, "calc");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].3).unwrap(),
            json!({"a": 7, "b": 6})
        );
    }

    #[test]
    fn test_hy4_chat_template_tools_and_streaming_parser() {
        assert_eq!(resolve_hy4_thinking(None, None), (true, "high"));
        assert_eq!(
            resolve_hy4_thinking(Some(&json!({"enable_thinking": false})), None),
            (false, "no_think")
        );
        assert_eq!(
            resolve_hy4_thinking(None, Some(&json!("no_think"))),
            (false, "no_think")
        );
        assert_eq!(
            resolve_hy4_thinking(None, Some(&json!("max"))),
            (true, "max")
        );

        let msgs = vec![user_msg("Hi")];
        let (p_off, exp_off, pf_off, imgs_off) =
            format_hy4_chat(&msgs, None, None, false, "no_think", ":opensource", false).unwrap();
        assert!(!exp_off);
        assert!(pf_off.is_empty());
        assert!(imgs_off.is_empty());
        assert_eq!(
            p_off,
            "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_end:opensource｜><｜hy_start:opensource｜>user<｜hy_middle:opensource｜>Hi<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>"
        );

        let (p_hi, exp_hi, pf_hi, _) =
            format_hy4_chat(&msgs, None, None, true, "high", ":opensource", false).unwrap();
        assert!(exp_hi);
        assert!(pf_hi.is_empty());
        assert!(p_hi.ends_with("<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>"));

        let msgs2 = vec![
            user_msg("Hi"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: Some(json!("Hello!")),
                ..Default::default()
            },
            user_msg("Next?"),
        ];
        let (p_t2, _, _, _) =
            format_hy4_chat(&msgs2, None, None, false, "no_think", ":opensource", false).unwrap();
        assert!(p_t2.starts_with(&p_off));

        let tools = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Multiply integers",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": { "a": {"type": "integer"}, "b": {"type": "integer"} }
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "skip_me",
                    "defer_loading": true,
                    "parameters": {}
                }
            }),
        ];
        let tool_msgs = vec![
            user_msg("Compute 7*6 and 2*3"),
            ChatMessage {
                role: "assistant".to_owned(),
                reasoning_content: Some("Need calc.".to_owned()),
                content: None,
                tool_calls: Some(vec![
                    json!({"id": "c1", "function": {"name": "calc", "arguments": "{\"a\":7,\"b\":6}"}}),
                    json!({"id": "c2", "function": {"name": "calc", "arguments": "{\"a\":2,\"b\":3}"}}),
                ]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c2".to_owned()),
                content: Some(json!("6")),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c1".to_owned()),
                content: Some(json!("42")),
                ..Default::default()
            },
        ];
        let (pt, _, _, _) = format_hy4_chat(
            &tool_msgs,
            Some(&tools),
            None,
            true,
            "high",
            ":opensource",
            false,
        )
        .unwrap();
        assert!(!pt.contains("skip_me"));
        assert!(!pt.contains("\"strict\""));
        assert!(pt.contains("<｜hy_start:opensource｜>tool<｜hy_middle:opensource｜><tool_response:opensource>42</tool_response:opensource><tool_response:opensource>6</tool_response:opensource><｜hy_end:opensource｜>"));

        let forced = json!({"type": "function", "function": {"name": "calc"}});
        let (pf_prompt, pf_think, pf_str, _) = format_hy4_chat(
            &[user_msg("go")],
            Some(&tools),
            Some(&forced),
            true,
            "high",
            ":opensource",
            false,
        )
        .unwrap();
        assert!(!pf_think);
        assert_eq!(pf_str, "<tool_call>calc");
        assert!(pf_prompt.ends_with("<tool_calls:opensource><tool_call:opensource>calc"));

        // Reject image in system message and reject image when vision disabled
        let img_msg = vec![ChatMessage {
            role: "user".to_owned(),
            content: Some(json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "describe"}
            ])),
            ..Default::default()
        }];
        assert!(format_hy4_chat(&img_msg, None, None, true, "high", ":opensource", false).is_err());
        let (img_prompt, _, _, img_srcs) =
            format_hy4_chat(&img_msg, None, None, true, "high", ":opensource", true).unwrap();
        assert_eq!(img_srcs, vec!["data:image/png;base64,AA=="]);
        assert!(img_prompt.contains(
            "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>describe"
        ));

        let schemas = extract_tool_schemas(Some(&tools));
        let mut parser = OutputParser::new(
            ChatOutputMode::Hy4 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_7".to_owned(),
        );
        let decoded_aliased = "thinking</think><tool_call>calc<arg_key>a</arg_key><arg_value>7</arg_value><arg_key>b</arg_key><arg_value>6</arg_value></tool_call>";
        let mut r = String::new();
        let mut calls: Vec<(usize, String, String, String)> = Vec::new();
        let mut push_ev = |ev: ParsedEvent| match ev {
            ParsedEvent::Reasoning(s) => r.push_str(&s),
            ParsedEvent::Token(_) => {}
            ParsedEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => calls.push((index, id, name, arguments)),
            ParsedEvent::ToolCallStart { index, id, name } => {
                calls.push((index, id, name, String::new()));
            }
            ParsedEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(call) = calls.iter_mut().find(|c| c.0 == index) {
                    call.3.push_str(&delta);
                }
            }
        };
        for ch in decoded_aliased.as_bytes().chunks(6) {
            parser.push(std::str::from_utf8(ch).unwrap(), &mut push_ev);
        }
        parser.finish(&mut push_ev);
        assert_eq!(r, "thinking");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, "calc");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].3).unwrap(),
            json!({"a": 7, "b": 6})
        );
    }

    #[test]
    fn test_hy4_vision_resize_rgba_and_expand() {
        let cfg = Hy4VisionConfig {
            dim: 1152,
            heads: 16,
            head_dim: 72,
            ff: 4304,
            layers: 27,
            patch_size: 16,
            spatial_merge: 2,
            grid: 128,
            out_dim: 3072,
            eps: 1e-5,
            min_pixels: 4096,
            max_pixels: 4_194_304,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_token_id: 120120,
            image_start_id: 120118,
            image_end_id: 120119,
        };
        let rgba = image::RgbaImage::from_pixel(300, 200, image::Rgba([255, 0, 0, 0]));
        let mut png_bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut png_bytes, image::ImageFormat::Png)
            .unwrap();
        let data_url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png_bytes.get_ref())
        );

        let raw = load_image_source(&data_url).unwrap();
        let pic = decode_and_prepare_image(&raw, &cfg).unwrap();
        assert_eq!((pic.width, pic.height), (288, 192));
        assert_eq!(pic.n_tokens, (192 / 32) * (288 / 32 + 1));
        assert_eq!(pic.rgb.len(), 288 * 192 * 3);
        assert_eq!(&pic.rgb[..3], &[255, 255, 255]);

        let (expanded, starts) = expand_hy4_image_tokens(
            &[1, cfg.image_start_id, cfg.image_token_id, cfg.image_end_id, 2],
            std::slice::from_ref(&pic),
            &cfg,
        )
        .unwrap();
        assert_eq!(starts, vec![2]);
        assert_eq!(expanded.len(), 4 + pic.n_tokens);
        assert_eq!(&expanded[2..2 + pic.n_tokens], vec![cfg.image_token_id; pic.n_tokens].as_slice());

        // Missing image_start/image_end delimiters must fail cleanly
        assert!(expand_hy4_image_tokens(&[1, cfg.image_token_id, cfg.image_end_id], std::slice::from_ref(&pic), &cfg).is_err());
        assert!(expand_hy4_image_tokens(&[1, 2], std::slice::from_ref(&pic), &cfg).is_err());
    }

    #[test]
    fn test_qwen4_chat_parser_and_vision_expand() {
        let (th_on, eff_on) = resolve_qwen4_thinking(None, None).unwrap();
        assert_eq!((th_on, eff_on), (true, "xhigh"));
        let (th_high, eff_high) = resolve_qwen4_thinking(None, Some(&json!("high"))).unwrap();
        assert_eq!((th_high, eff_high), (true, "xhigh"));
        let (th_low, eff_low) = resolve_qwen4_thinking(None, Some(&json!("low"))).unwrap();
        assert_eq!((th_low, eff_low), (true, "low"));
        let (th_off, eff_off) =
            resolve_qwen4_thinking(Some(&json!({"enable_thinking": false})), None).unwrap();
        assert!(!th_off && eff_off.is_empty());

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "expr": {"type": "string"}
                    }
                }
            }
        }]);
        let msgs = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: Some(json!("Be terse.")),
                ..Default::default()
            },
            user_msg("compute"),
        ];
        let (prompt, expect_think, prefill, imgs) = format_qwen4_chat(
            &msgs,
            tools.as_array().map(Vec::as_slice),
            None,
            true,
            "low",
            false,
        )
        .unwrap();
        assert!(expect_think);
        assert!(prefill.is_empty());
        assert!(imgs.is_empty());
        assert!(prompt.starts_with("<|im_start|>system\nReasoning effort is set to low."));
        assert!(prompt.contains("<function=example_function_name>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));

        let schemas = extract_tool_schemas(tools.as_array().map(Vec::as_slice));
        let mut parser = OutputParser::new(
            ChatOutputMode::Qwen4 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_q4".to_owned(),
        );
        let stream_txt = "reason</think>\n\n<tool_call>\n<function=calc>\n<parameter=a>\n42\n</parameter>\n<parameter=expr>\nline1\nline2\n</parameter>\n</function>\n</tool_call>";
        let mut r = String::new();
        let mut calls = Vec::new();
        for chunk in stream_txt.as_bytes().chunks(5) {
            parser.push(std::str::from_utf8(chunk).unwrap(), &mut |ev| match ev {
                ParsedEvent::Reasoning(s) => r.push_str(&s),
                ParsedEvent::Token(_) => {}
                ParsedEvent::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => calls.push((index, id, name, arguments)),
                ParsedEvent::ToolCallStart { index, id, name } => {
                    calls.push((index, id, name, String::new()));
                }
                ParsedEvent::ToolCallArgsDelta { index, delta } => {
                    if let Some(c) = calls.iter_mut().find(|x| x.0 == index) {
                        c.3.push_str(&delta);
                    }
                }
            });
        }
        parser.finish(&mut |_| {});
        assert_eq!(r, "reason");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, "calc");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].3).unwrap(),
            json!({"a": 42, "expr": "line1\nline2"})
        );

        let vcfg = Qwen4VisionConfig {
            dim: 1152,
            heads: 16,
            head_dim: 72,
            ff: 4304,
            layers: 27,
            patch_size: 16,
            spatial_merge: 2,
            image_size: 768,
            grid: 48,
            out_dim: 2560,
            eps: 1e-6,
            min_pixels: 4096,
            max_pixels: 4_194_304,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_token_id: 248056,
            vision_start_id: 248053,
            vision_end_id: 248054,
        };
        let rgba = image::RgbaImage::from_pixel(256, 256, image::Rgba([10, 20, 30, 255]));
        let mut png_bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut png_bytes, image::ImageFormat::Png)
            .unwrap();
        let pic = decode_and_prepare_qwen4_image(png_bytes.get_ref(), &vcfg).unwrap();
        assert_eq!((pic.width, pic.height, pic.n_tokens), (256, 256, 64));
        let (expanded, starts) = expand_qwen4_image_tokens(
            &[248045, vcfg.vision_start_id, vcfg.image_token_id, vcfg.vision_end_id, 248046],
            std::slice::from_ref(&pic),
            &vcfg,
        )
        .unwrap();
        assert_eq!(starts, vec![2]);
        assert_eq!(expanded.len(), 4 + 64);
    }

    #[test]
    fn test_mm3_chat_parser_and_vision_expand() {
        let kw = json!({"thinking_mode": "disabled", "reasoning_effort": "low"});
        let (tm, eff) = resolve_mm3_thinking(Some(&kw), None);
        assert_eq!(tm, "disabled");
        assert_eq!(eff.as_deref(), Some("low"));

        let tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Evaluate expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "items": {"type": "array"}
                        }
                    }
                }
            }
        ]);
        let msgs = vec![
            ChatMessage {
                role: "developer".to_owned(),
                content: Some(json!("Be concise.")),
                ..Default::default()
            },
            user_msg("compute 6 * 7"),
        ];
        let (prompt, expect_think, prefill, imgs) = format_mm3_chat(
            &msgs,
            tools.as_array().map(Vec::as_slice),
            None,
            "enabled",
            None,
            false,
        )
        .unwrap();
        assert!(expect_think);
        assert!(prefill.is_empty());
        assert!(imgs.is_empty());
        assert!(prompt.starts_with("]~!b[]~b]system\n"));
        assert!(prompt.contains("]<]minimax[>[<invoke name=\"tool-name-1\">"));
        assert!(prompt.ends_with("]~b]ai\n<mm:think>"));

        let schemas = extract_tool_schemas(tools.as_array().map(Vec::as_slice));
        let mut parser = OutputParser::new(
            ChatOutputMode::Mm3 {
                expect_think_close: true,
                parse_tools: true,
                prefill: String::new(),
                schemas,
            },
            "call_m3".to_owned(),
        );
        let stream_txt = "thinking...</mm:think>Hello ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"calc\">]<]minimax[>[<a>42]<]minimax[>[</a>]<]minimax[>[<items>]<]minimax[>[<item>x]<]minimax[>[</item>]<]minimax[>[<item>y]<]minimax[>[</item>]<]minimax[>[</items>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
        let mut r = String::new();
        let mut txt = String::new();
        let mut calls = Vec::new();
        for chunk in stream_txt.as_bytes().chunks(7) {
            parser.push(std::str::from_utf8(chunk).unwrap(), &mut |ev| match ev {
                ParsedEvent::Reasoning(s) => r.push_str(&s),
                ParsedEvent::Token(s) => txt.push_str(&s),
                ParsedEvent::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => calls.push((index, id, name, arguments)),
                ParsedEvent::ToolCallStart { index, id, name } => {
                    calls.push((index, id, name, String::new()));
                }
                ParsedEvent::ToolCallArgsDelta { index, delta } => {
                    if let Some(c) = calls.iter_mut().find(|x| x.0 == index) {
                        c.3.push_str(&delta);
                    }
                }
            });
        }
        parser.finish(&mut |_| {});
        assert_eq!(r, "thinking...");
        assert_eq!(txt, "Hello ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, "calc");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].3).unwrap(),
            json!({"a": 42, "items": ["x", "y"]})
        );

        let mut p_adapt = OutputParser::new(
            ChatOutputMode::Mm3 {
                expect_think_close: false,
                parse_tools: true,
                prefill: String::new(),
                schemas: std::collections::HashMap::new(),
            },
            "call_m3_adapt".to_owned(),
        );
        let mut adapt_txt = String::new();
        for chunk in "</mm:think>Hello".as_bytes().chunks(4) {
            p_adapt.push(std::str::from_utf8(chunk).unwrap(), &mut |ev| {
                if let ParsedEvent::Token(s) = ev {
                    adapt_txt.push_str(&s);
                }
            });
        }
        p_adapt.finish(&mut |ev| {
            if let ParsedEvent::Token(s) = ev {
                adapt_txt.push_str(&s);
            }
        });
        assert_eq!(adapt_txt, "Hello");

        let vcfg = Mm3VisionConfig {
            hidden_size: 1280,
            num_heads: 16,
            head_dim: 80,
            num_layers: 32,
            intermediate_size: 5120,
            projection_dim: 6144,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            image_size: 672,
            max_image_resolution: 672,
            min_pixels: 3136,
            max_pixels: 451584,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            image_token_id: 200025,
            image_start_token_id: 200029,
            image_end_token_id: 200030,
            video_token_id: 200026,
        };
        let rgba = image::RgbaImage::from_pixel(280, 280, image::Rgba([10, 20, 30, 255]));
        let mut png_bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut png_bytes, image::ImageFormat::Png)
            .unwrap();
        let pic = decode_and_prepare_mm3_image(png_bytes.get_ref(), &vcfg).unwrap();
        assert_eq!((pic.width, pic.height, pic.n_tokens), (280, 280, 100));
        let (expanded, starts) = expand_mm3_image_tokens(
            &[10, vcfg.image_token_id, 11],
            std::slice::from_ref(&pic),
            &vcfg,
        )
        .unwrap();
        assert_eq!(starts, vec![2]);
        assert_eq!(expanded.len(), 2 + 1 + 100 + 1);
        assert_eq!(expanded[1], vcfg.image_start_token_id);
        assert_eq!(expanded[102], vcfg.image_end_token_id);
    }

    #[test]
    fn test_glm5_vision_chat_format_and_token_expand() {
        let msgs = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: Some(json!([
                    {"type": "image_url", "image_url": {"url": "/tmp/user.png"}},
                    {"type": "text", "text": "Inspect this"}
                ])),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: None,
                tool_calls: Some(vec![
                    json!({"id": "c1", "function": {"name": "read", "arguments": "{\"path\":\"/tmp/a.png\"}"}}),
                    json!({"id": "c2", "function": {"name": "read", "arguments": "{\"path\":\"/tmp/b.png\"}"}}),
                ]),
                ..Default::default()
            },
            // Out-of-order tool results with images: both prompt order and image_sources must reorder cleanly
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c2".to_owned()),
                content: Some(json!([
                    {"type": "image_url", "image_url": {"url": "/tmp/b.png"}},
                    {"type": "text", "text": "imgB"}
                ])),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_owned(),
                tool_call_id: Some("c1".to_owned()),
                content: Some(json!([
                    {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
                    {"type": "text", "text": "imgA"}
                ])),
                ..Default::default()
            },
        ];

        let (prompt, expect_think, _, imgs) =
            format_glm5_chat(&msgs, None, None, true, "Max", true).unwrap();
        assert!(expect_think);
        assert_eq!(imgs, vec!["/tmp/user.png", "/tmp/a.png", "/tmp/b.png"]);
        assert!(prompt.contains("<|user|><|begin_of_image|><|image|><|end_of_image|>Inspect this"));
        assert!(prompt.contains(
            "<|observation|><tool_response><|begin_of_image|><|image|><|end_of_image|>imgA</tool_response><tool_response><|begin_of_image|><|image|><|end_of_image|>imgB</tool_response>"
        ));

        let vcfg = ampllm_gguf::Glm5VisionConfig {
            dim: 1024,
            heads: 16,
            head_dim: 64,
            ff: 4096,
            layers: 24,
            patch_size: 14,
            spatial_merge: 2,
            image_size: 448,
            out_dim: 4096,
            merger_ff: 10240,
            eps: 1e-5,
            ffn_clamp: 10.0,
            min_tokens: 16,
            max_tokens: 2560,
            min_pixels: 16 * 28 * 28,
            max_pixels: 2560 * 28 * 28,
            image_mean: [0.48145467, 0.4578275, 0.40821072],
            image_std: [0.26862955, 0.2613026, 0.2757771],
            image_token_id: 154854,
            vision_start_id: 154830,
            vision_end_id: 154831,
        };
        let rgba = image::RgbaImage::from_pixel(448, 448, image::Rgba([40, 80, 120, 255]));
        let mut png_bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut png_bytes, image::ImageFormat::Png)
            .unwrap();
        let pic = crate::vision::decode_and_prepare_glm5_image(png_bytes.get_ref(), &vcfg).unwrap();
        assert_eq!((pic.width, pic.height, pic.n_tokens), (448, 448, 256));
        let (expanded, starts) = crate::vision::expand_glm5_image_tokens(
            &[10, vcfg.vision_start_id, vcfg.image_token_id, vcfg.vision_end_id, 11],
            std::slice::from_ref(&pic),
            &vcfg,
        )
        .unwrap();
        assert_eq!(starts, vec![2]);
        assert_eq!(expanded.len(), 2 + 256 + 2);
        assert_eq!(expanded[1], vcfg.vision_start_id);
        assert_eq!(expanded[2], vcfg.image_token_id);
        assert_eq!(expanded[257], vcfg.image_token_id);
        assert_eq!(expanded[258], vcfg.vision_end_id);
    }
}
