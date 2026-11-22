pub mod metrics;
pub mod parser;
pub mod routes;

use crate::metrics::ServerMetrics;
use crate::parser::{ChatOutputMode, OutputParser, ParsedEvent};
use crate::routes::{
    chat_completions, completions, health, list_models, metrics, AppState, EngineJob, StreamEvent,
};
use ampllm_engine::AnyEngine;
use ampllm_tokenizer::BpeTokenizer;
use anyhow::Result;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::mpsc;

fn collect_stop_ids(tokenizer: &BpeTokenizer, specials: &[&str]) -> Vec<u32> {
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

fn forward_parsed_event(ev: ParsedEvent, tx: &mpsc::UnboundedSender<StreamEvent>) {
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
    }
}

fn decode_for_mode(tokenizer: &BpeTokenizer, mode: &ChatOutputMode, ids: &[u32]) -> String {
    match mode {
        ChatOutputMode::Glm5 { .. } => tokenizer.decode_skip_control(ids),
        ChatOutputMode::Inkling2 { .. } => tokenizer.decode(ids, false),
        ChatOutputMode::Plain => tokenizer.decode(ids, true),
    }
}

pub async fn run_server(
    mut engine: AnyEngine,
    tokenizer: BpeTokenizer,
    model_name: String,
    addr: SocketAddr,
) -> Result<()> {
    let is_glm5 = engine.is_glm5();
    let max_ctx = engine.max_ctx();
    let server_metrics = Arc::new(ServerMetrics::default());
    let metrics_worker = Arc::clone(&server_metrics);
    let (tx, mut rx) = mpsc::unbounded_channel::<EngineJob>();

    // Dedicated inference thread owns engine (single-writer CUDA context safety)
    std::thread::Builder::new()
        .name("ampllm-infer-worker".to_owned())
        .spawn(move || {
            let glm5_stops = collect_stop_ids(
                &tokenizer,
                &["<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"],
            );
            let inkling_text_stops = collect_stop_ids(
                &tokenizer,
                &[
                    "<|endoftext|>",
                    "<|end_message|>",
                    "<|content_model_end_sampling|>",
                    "<|message_user|>",
                    "<|message_system|>",
                    "<|message_tool|>",
                ],
            );
            let inkling_structured_stops = collect_stop_ids(
                &tokenizer,
                &[
                    "<|endoftext|>",
                    "<|content_model_end_sampling|>",
                    "<|message_user|>",
                    "<|message_system|>",
                    "<|message_tool|>",
                ],
            );
            let plain_stops = collect_stop_ids(&tokenizer, &["<|endoftext|>"]);

            while let Some(mut job) = rx.blocking_recv() {
                let mode_stops: &[u32] = match &job.output_mode {
                    ChatOutputMode::Glm5 { .. } => &glm5_stops,
                    ChatOutputMode::Inkling2 {
                        start_in_text,
                        parse_tools,
                    } => {
                        if *start_in_text && !*parse_tools {
                            &inkling_text_stops
                        } else {
                            &inkling_structured_stops
                        }
                    }
                    ChatOutputMode::Plain => &plain_stops,
                };
                for &s in mode_stops {
                    if !job.params.stop_token_ids.contains(&s) {
                        job.params.stop_token_ids.push(s);
                    }
                }

                let prompt_ids = match tokenizer.encode(&job.prompt, true) {
                    Ok(ids) => ids,
                    Err(e) => {
                        let _ = job.resp_tx.send(StreamEvent::Error(e.to_string()));
                        continue;
                    }
                };
                let avail_tokens = max_ctx.saturating_sub(prompt_ids.len()).max(1);
                job.params.max_tokens = job.params.max_tokens.min(avail_tokens);

                let mut gen_ids = Vec::new();
                let mut emitted_bytes = 0usize;
                let mut hit_stop_str = false;
                let mut parser =
                    OutputParser::new(job.output_mode.clone(), job.call_id_prefix.clone());
                let max_stop_len = job.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

                let resp_tx_cb = &job.resp_tx;
                let mut emit_piece = |s: &str| {
                    parser.push(s, &mut |ev| forward_parsed_event(ev, resp_tx_cb));
                };

                let res = engine.generate_stream(&prompt_ids, &job.params, |tok_id| {
                    gen_ids.push(tok_id);
                    let decoded = decode_for_mode(&tokenizer, &job.output_mode, &gen_ids);
                    if !job.stop_strings.is_empty() {
                        for stop_s in &job.stop_strings {
                            if let Some(pos) = decoded.find(stop_s.as_str()) {
                                if pos > emitted_bytes && decoded.is_char_boundary(pos) {
                                    emit_piece(&decoded[emitted_bytes..pos]);
                                }
                                hit_stop_str = true;
                                return false;
                            }
                        }
                    }
                    // Hold back trailing potential partial stop match or replacement char U+FFFD
                    let mut send_until = decoded.len();
                    if decoded.ends_with('\u{FFFD}') {
                        send_until = decoded.len().saturating_sub('\u{FFFD}'.len_utf8());
                    } else if max_stop_len > 1 {
                        let keep_tail = max_stop_len - 1;
                        let target = decoded.len().saturating_sub(keep_tail);
                        let mut p = target;
                        while p > emitted_bytes && !decoded.is_char_boundary(p) {
                            p -= 1;
                        }
                        send_until = p;
                    }
                    if send_until > emitted_bytes {
                        let piece = &decoded[emitted_bytes..send_until];
                        emitted_bytes = send_until;
                        emit_piece(piece);
                    }
                    true
                });

                match res {
                    Ok(info) => {
                        if !hit_stop_str {
                            let decoded = decode_for_mode(&tokenizer, &job.output_mode, &gen_ids);
                            if decoded.len() > emitted_bytes {
                                emit_piece(&decoded[emitted_bytes..]);
                            }
                        }
                        parser.finish(&mut |ev| forward_parsed_event(ev, &job.resp_tx));

                        metrics_worker
                            .tokens_prefilled_total
                            .fetch_add(info.prompt_tokens_evaluated as u64, Ordering::Relaxed);
                        metrics_worker
                            .tokens_generated_total
                            .fetch_add(info.generated_tokens.len() as u64, Ordering::Relaxed);
                        metrics_worker.update_from_engine(
                            &engine.cache_stats(),
                            engine.prefix_reuse_tokens_total(),
                        );

                        let finish_reason = if parser.has_tool_calls() {
                            "tool_calls".to_owned()
                        } else if !hit_stop_str
                            && info.generated_tokens.len() >= job.params.max_tokens
                        {
                            "length".to_owned()
                        } else {
                            "stop".to_owned()
                        };
                        let _ = job.resp_tx.send(StreamEvent::Done {
                            prompt_tokens: info.prompt_tokens_evaluated,
                            completion_tokens: info.generated_tokens.len(),
                            prefix_reused_tokens: info.prefix_reused_tokens,
                            finish_reason,
                            prefill_tps: info.prefill_tps,
                            decode_tps: info.decode_tps,
                            draft_n: info.draft_n,
                            draft_accepted: info.draft_accepted,
                        });
                    }
                    Err(e) => {
                        tracing::error!("infer worker error: {e:#}");
                        let _ = job.resp_tx.send(StreamEvent::Error(e.to_string()));
                    }
                }
            }
        })?;

    let state = AppState {
        model_name,
        is_glm5,
        max_ctx,
        tx,
        metrics: server_metrics,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
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
        extract_tool_schemas, format_glm5_chat, format_inkling2_chat, resolve_glm5_thinking,
        ChatMessage,
    };
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
        let (p_off, exp_off, pf_off) = format_glm5_chat(&msgs, None, None, false, "Low");
        assert!(!exp_off);
        assert!(pf_off.is_empty());
        assert_eq!(
            p_off,
            "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think></think>"
        );

        let (p_on, exp_on, pf_on) = format_glm5_chat(&msgs, None, None, true, "Max");
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
        let (p_t2, _, _) = format_glm5_chat(&msgs2, None, None, false, "Low");
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
                ParsedEvent::ToolCall { .. } => panic!("unexpected tool call"),
            });
        }
        parser.finish(&mut |ev| match ev {
            ParsedEvent::Reasoning(s) => r.push_str(&s),
            ParsedEvent::Token(s) => c.push_str(&s),
            ParsedEvent::ToolCall { .. } => panic!("unexpected tool call"),
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

        let (prompt, expect_think, prefill) =
            format_glm5_chat(&msgs, Some(&tools), None, true, "High");
        assert!(expect_think);
        assert!(prefill.is_empty());
        assert!(prompt.contains("<tools>\n{\"description\":\"Multiply integers\",\"name\":\"calc\""));
        assert!(!prompt.contains("hidden_tool"));
        assert!(!prompt.contains("\"strict\""));
        assert!(prompt.contains("<|observation|><tool_response>42</tool_response><tool_response>6</tool_response><|assistant|><think>"));

        let forced_tc = json!({"type": "function", "function": {"name": "calc"}});
        let (fp, f_think, f_prefill) =
            format_glm5_chat(&[user_msg("go")], Some(&tools), Some(&forced_tc), true, "Max");
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
        let mut calls = Vec::new();
        for ch in raw.as_bytes().chunks(5) {
            let piece = std::str::from_utf8(ch).unwrap();
            p.push(piece, &mut |ev| match ev {
                ParsedEvent::Reasoning(s) => reasoning.push_str(&s),
                ParsedEvent::Token(s) => text.push_str(&s),
                ParsedEvent::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => calls.push((index, id, name, arguments)),
            });
        }
        p.finish(&mut |ev| match ev {
            ParsedEvent::Reasoning(s) => reasoning.push_str(&s),
            ParsedEvent::Token(s) => text.push_str(&s),
            ParsedEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => calls.push((index, id, name, arguments)),
        });
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
}
