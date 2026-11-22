use crate::metrics::ServerMetrics;
use ampllm_engine::SamplingParams;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub is_glm5: bool,
    pub max_ctx: usize,
    pub tx: mpsc::UnboundedSender<EngineJob>,
    pub metrics: Arc<ServerMetrics>,
}

pub enum StreamEvent {
    Token(String),
    Reasoning(String),
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
        prefix_reused_tokens: usize,
        finish_reason: String,
        prefill_tps: f64,
        decode_tps: f64,
        draft_n: usize,
        draft_accepted: usize,
    },
    Error(String),
}

pub struct EngineJob {
    pub prompt: String,
    pub params: SamplingParams,
    pub stop_strings: Vec<String>,
    pub expect_think_close: bool,
    pub resp_tx: mpsc::UnboundedSender<StreamEvent>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_effort: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopParam>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multi(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::Single(s) => vec![s],
            StopParam::Multi(v) => v,
        }
    }
}

#[derive(Serialize)]
struct PromptTokensDetails {
    cached_tokens: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    prefix_cached_tokens: usize,
    prompt_tokens_details: PromptTokensDetails,
}

fn build_usage_timings(
    prompt_tokens: usize,
    completion_tokens: usize,
    prefix_reused_tokens: usize,
    prefill_tps: f64,
    decode_tps: f64,
    draft_n: usize,
    draft_accepted: usize,
) -> (Usage, serde_json::Value) {
    let usage = Usage {
        prompt_tokens: prompt_tokens + prefix_reused_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + prefix_reused_tokens + completion_tokens,
        prefix_cached_tokens: prefix_reused_tokens,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: prefix_reused_tokens,
        },
    };
    let timings = serde_json::json!({
        "prompt_n": prompt_tokens,
        "prompt_per_second": (prefill_tps * 100.0).round() / 100.0,
        "predicted_n": completion_tokens,
        "predicted_per_second": (decode_tps * 100.0).round() / 100.0,
        "draft_n": draft_n,
        "draft_n_accepted": draft_accepted,
    });
    (usage, timings)
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format OpenAI chat messages using Inkling-2-Mini's native chat template tokens so
/// multi-turn conversations strictly extend the exact BPE prefix for KV+conv reuse.
pub fn format_inkling2_chat(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut effort_emitted = false;

    for msg in messages {
        if !effort_emitted && msg.role != "system" {
            out.push_str("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>");
            effort_emitted = true;
        }
        let role_tok = match msg.role.as_str() {
            "system" => "<|message_system|>",
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "tool" => "<|message_tool|>",
            _ => "<|message_user|>",
        };
        let content = msg.content.as_deref().unwrap_or("");
        out.push_str(role_tok);
        out.push_str("<|content_text|>");
        out.push_str(content);
        out.push_str("<|end_message|>");
        if msg.role == "assistant" {
            out.push_str("<|content_model_end_sampling|>");
        }
        if msg.role == "system" {
            effort_emitted = true;
        }
    }
    if !effort_emitted {
        out.push_str("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>");
    }
    out.push_str("<|message_model|><|content_text|>");
    out
}

/// Resolve `(enable_thinking, effort_label)` from OpenAI / GLM-5 request options.
pub fn resolve_glm5_thinking(
    chat_template_kwargs: Option<&serde_json::Value>,
    reasoning_effort: Option<&serde_json::Value>,
) -> (bool, &'static str) {
    let mut off = false;
    let mut effort_str: Option<&str> = None;

    if let Some(kw) = chat_template_kwargs.and_then(|v| v.as_object()) {
        if let Some(et) = kw.get("enable_thinking").and_then(|v| v.as_bool()) {
            if !et {
                off = true;
            }
        }
        if let Some(re) = kw.get("reasoning_effort").and_then(|v| v.as_str()) {
            effort_str = Some(re);
        }
    }
    if let Some(re) = reasoning_effort {
        if re == &serde_json::Value::Bool(false) {
            off = true;
        } else if let Some(s) = re.as_str() {
            effort_str = Some(s);
        }
    }

    if let Some(s) = effort_str {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "minimal" | "disabled" | "false" => off = true,
            _ => {}
        }
    }

    if off {
        (false, "Low")
    } else {
        let label = match effort_str.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("low") => "Low",
            Some("high") => "High",
            _ => "Max",
        };
        (true, label)
    }
}

/// Format OpenAI chat messages using GLM-5.3-Flash's exact GGUF chat_template (`[gMASK]<sop>...`).
/// Returns `(rendered_prompt, expect_think_close)`.
pub fn format_glm5_chat(
    messages: &[ChatMessage],
    enable_thinking: bool,
    effort_label: &str,
) -> (String, bool) {
    let mut out = String::new();
    out.push_str("[gMASK]<sop><|system|>Reasoning Effort: ");
    out.push_str(effort_label);

    for (idx, msg) in messages.iter().enumerate() {
        let raw = msg.content.as_deref().unwrap_or("");
        match msg.role.as_str() {
            "system" => {
                out.push_str("<|system|>");
                out.push_str(raw);
            }
            "user" => {
                out.push_str("<|user|>");
                out.push_str(raw);
            }
            "assistant" => {
                out.push_str("<|assistant|>");
                if let Some(end_idx) = raw.find("</think>") {
                    let think_body = raw[..end_idx]
                        .rsplit("<think>")
                        .next()
                        .unwrap_or(&raw[..end_idx]);
                    let ans = &raw[end_idx + "</think>".len()..];
                    out.push_str("<think>");
                    out.push_str(think_body);
                    out.push_str("</think>");
                    let trimmed = ans.trim();
                    if !trimmed.is_empty() {
                        out.push_str(trimmed);
                    }
                } else {
                    out.push_str("<think></think>");
                    let trimmed = raw.trim();
                    if !trimmed.is_empty() {
                        out.push_str(trimmed);
                    }
                }
            }
            "tool" => {
                if idx == 0 || messages[idx - 1].role != "tool" {
                    out.push_str("<|observation|>");
                }
                out.push_str("<tool_response>");
                out.push_str(raw);
                out.push_str("</tool_response>");
            }
            _ => {
                out.push_str("<|user|>");
                out.push_str(raw);
            }
        }
    }

    if enable_thinking {
        out.push_str("<|assistant|><think>");
        (out, true)
    } else {
        out.push_str("<|assistant|><think></think>");
        (out, false)
    }
}

pub async fn health(State(st): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": st.model_name,
        "max_context": st.max_ctx
    }))
}

pub async fn list_models(State(st): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": st.model_name,
            "object": "model",
            "created": unix_ts(),
            "owned_by": "ampllm"
        }]
    }))
}

pub async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    let body = st.metrics.format_prometheus();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

pub async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let (prompt, expect_think_close) = if st.is_glm5 {
        let (enable_thinking, effort_label) = resolve_glm5_thinking(
            req.chat_template_kwargs.as_ref(),
            req.reasoning_effort.as_ref(),
        );
        format_glm5_chat(&req.messages, enable_thinking, effort_label)
    } else {
        (format_inkling2_chat(&req.messages), false)
    };

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens.unwrap_or(512).min(4096),
        stop_token_ids: Vec::new(),
    };
    let stop_strings = req.stop.map(StopParam::into_vec).unwrap_or_default();
    let stream_flag = req.stream.unwrap_or(false);
    let model = req.model.unwrap_or_else(|| st.model_name.clone());

    st.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt,
            params,
            stop_strings,
            expect_think_close,
            resp_tx,
        })
        .is_err()
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "inference worker stopped").into_response();
    }

    let req_id = format!("chatcmpl-{}", unix_ts());
    let created = unix_ts();

    if stream_flag {
        let model_c = model.clone();
        let req_id_c = req_id.clone();
        let sse_stream = UnboundedReceiverStream::new(resp_rx).map(move |ev| {
            let item = match ev {
                StreamEvent::Token(tok) => {
                    let payload = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": tok },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    Event::default().data(payload.to_string())
                }
                StreamEvent::Reasoning(r) => {
                    let payload = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning_content": r },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    Event::default().data(payload.to_string())
                }
                StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    prefix_reused_tokens,
                    finish_reason,
                    prefill_tps,
                    decode_tps,
                    draft_n,
                    draft_accepted,
                } => {
                    let (usage, timings) = build_usage_timings(
                        prompt_tokens,
                        completion_tokens,
                        prefix_reused_tokens,
                        prefill_tps,
                        decode_tps,
                        draft_n,
                        draft_accepted,
                    );
                    let payload = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }],
                        "usage": usage,
                        "timings": timings
                    });
                    Event::default().data(payload.to_string())
                }
                StreamEvent::Error(e) => {
                    Event::default().data(serde_json::json!({"error": e}).to_string())
                }
            };
            Ok::<Event, Infallible>(item)
        });
        let done_tail = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        });
        let combined = sse_stream.chain(done_tail);
        return Sse::new(combined)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        prefix_cached_tokens: 0,
        prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
    };
    let mut timings = serde_json::json!({});
    let mut finish_reason = "stop".to_owned();

    while let Some(ev) = resp_rx.recv().await {
        match ev {
            StreamEvent::Token(t) => content.push_str(&t),
            StreamEvent::Reasoning(r) => reasoning.push_str(&r),
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens,
                prefix_reused_tokens,
                finish_reason: fr,
                prefill_tps,
                decode_tps,
                draft_n,
                draft_accepted,
            } => {
                (usage, timings) = build_usage_timings(
                    prompt_tokens,
                    completion_tokens,
                    prefix_reused_tokens,
                    prefill_tps,
                    decode_tps,
                    draft_n,
                    draft_accepted,
                );
                finish_reason = fr;
                break;
            }
            StreamEvent::Error(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
            }
        }
    }

    let mut msg_obj = serde_json::json!({
        "role": "assistant",
        "content": content
    });
    if !reasoning.is_empty() {
        msg_obj["reasoning_content"] = serde_json::Value::String(reasoning);
    }

    Json(serde_json::json!({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg_obj,
            "finish_reason": finish_reason
        }],
        "usage": usage,
        "timings": timings
    }))
    .into_response()
}

pub async fn completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens.unwrap_or(256).min(4096),
        stop_token_ids: Vec::new(),
    };
    let stop_strings = req.stop.map(StopParam::into_vec).unwrap_or_default();
    let stream_flag = req.stream.unwrap_or(false);
    let model = req.model.unwrap_or_else(|| st.model_name.clone());

    st.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt: req.prompt,
            params,
            stop_strings,
            expect_think_close: false,
            resp_tx,
        })
        .is_err()
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "inference worker stopped").into_response();
    }

    let req_id = format!("cmpl-{}", unix_ts());
    let created = unix_ts();

    if stream_flag {
        let model_c = model.clone();
        let req_id_c = req_id.clone();
        let sse_stream = UnboundedReceiverStream::new(resp_rx).map(move |ev| {
            let item = match ev {
                StreamEvent::Token(tok) | StreamEvent::Reasoning(tok) => Event::default().data(
                    serde_json::json!({
                        "id": req_id_c,
                        "object": "text_completion",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "text": tok,
                            "index": 0,
                            "finish_reason": serde_json::Value::Null
                        }]
                    })
                    .to_string(),
                ),
                StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    prefix_reused_tokens,
                    finish_reason,
                    prefill_tps,
                    decode_tps,
                    draft_n,
                    draft_accepted,
                } => {
                    let (usage, timings) = build_usage_timings(
                        prompt_tokens,
                        completion_tokens,
                        prefix_reused_tokens,
                        prefill_tps,
                        decode_tps,
                        draft_n,
                        draft_accepted,
                    );
                    Event::default().data(
                        serde_json::json!({
                            "id": req_id_c,
                            "object": "text_completion",
                            "created": created,
                            "model": model_c,
                            "choices": [{
                                "text": "",
                                "index": 0,
                                "finish_reason": finish_reason
                            }],
                            "usage": usage,
                            "timings": timings
                        })
                        .to_string(),
                    )
                }
                StreamEvent::Error(e) => {
                    Event::default().data(serde_json::json!({"error": e}).to_string())
                }
            };
            Ok::<Event, Infallible>(item)
        });
        let done_tail = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        });
        return Sse::new(sse_stream.chain(done_tail))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let mut text = String::new();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        prefix_cached_tokens: 0,
        prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
    };
    let mut timings = serde_json::json!({});
    let mut finish_reason = "stop".to_owned();

    while let Some(ev) = resp_rx.recv().await {
        match ev {
            StreamEvent::Token(t) | StreamEvent::Reasoning(t) => text.push_str(&t),
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens,
                prefix_reused_tokens,
                finish_reason: fr,
                prefill_tps,
                decode_tps,
                draft_n,
                draft_accepted,
            } => {
                (usage, timings) = build_usage_timings(
                    prompt_tokens,
                    completion_tokens,
                    prefix_reused_tokens,
                    prefill_tps,
                    decode_tps,
                    draft_n,
                    draft_accepted,
                );
                finish_reason = fr;
                break;
            }
            StreamEvent::Error(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
            }
        }
    }

    Json(serde_json::json!({
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": finish_reason
        }],
        "usage": usage,
        "timings": timings
    }))
    .into_response()
}
