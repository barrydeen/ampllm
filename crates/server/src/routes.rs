use crate::metrics::ServerMetrics;
use crate::parser::{ChatOutputMode, ToolParamSchema};
use crate::scheduler::SlotPublicStatus;
use ampllm_engine::SamplingParams;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::StreamExt as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

static NEXT_REQ_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerModelKind {
    Inkling2,
    Glm5,
    Hy4,
    Qwen4,
    Mm3,
}

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub model_kind: ServerModelKind,
    pub hy4_tag_suffix: String,
    pub has_vision: bool,
    pub max_ctx: usize,
    pub tx: mpsc::UnboundedSender<EngineJob>,
    pub metrics: Arc<ServerMetrics>,
    pub pending_jobs: Arc<AtomicUsize>,
    pub slot_statuses: Arc<RwLock<Vec<SlotPublicStatus>>>,
}

pub enum StreamEvent {
    Token(String),
    Reasoning(String),
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        index: usize,
        delta: String,
    },
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
    pub image_sources: Vec<String>,
    pub params: SamplingParams,
    pub stop_strings: Vec<String>,
    pub output_mode: ChatOutputMode,
    pub call_id_prefix: String,
    pub resp_tx: mpsc::UnboundedSender<StreamEvent>,
}

#[derive(Deserialize, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
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
    pub max_completion_tokens: Option<usize>,
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
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
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

/// Parsed OpenAI `tool_choice` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
    Function(String),
}

/// Parse an OpenAI `tool_choice` value (`"auto"`, `"none"`, `"required"`, or `{"type":"function","function":{"name":...}}`).
pub fn parse_tool_choice(val: Option<&serde_json::Value>) -> ToolChoiceMode {
    match val {
        None | Some(serde_json::Value::Null) => ToolChoiceMode::Auto,
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "none" => ToolChoiceMode::None,
            "required" => ToolChoiceMode::Required,
            _ => ToolChoiceMode::Auto,
        },
        Some(serde_json::Value::Object(obj)) => {
            if let Some(name) = obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .or_else(|| obj.get("name").and_then(|n| n.as_str()))
            {
                ToolChoiceMode::Function(name.to_string())
            } else {
                ToolChoiceMode::Auto
            }
        }
        _ => ToolChoiceMode::Auto,
    }
}

/// Extract plain visible text from a chat message's `content` (string or OpenAI multipart array).
pub fn extract_visible_text(msg: &ChatMessage) -> String {
    match msg.content.as_ref() {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(s) = p.as_str() {
                    out.push_str(s);
                } else if let Some(txt) = p.get("text").and_then(|v| v.as_str()) {
                    out.push_str(txt);
                } else if let Some(txt) = p.get("content").and_then(|v| v.as_str()) {
                    out.push_str(txt);
                }
            }
            out
        }
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Check whether an OpenAI tool entry should be excluded (`defer_loading: true`).
fn is_deferred_tool(tool: &serde_json::Value) -> bool {
    let fn_obj = tool.get("function").unwrap_or(tool);
    tool.get("defer_loading")
        .and_then(|v| v.as_bool())
        .or_else(|| fn_obj.get("defer_loading").and_then(|v| v.as_bool()))
        == Some(true)
}

/// Build an Inkling/GLM clean function object (unwrapped from `{type, function}` with internal keys removed).
fn sanitize_tool_fn_object(tool: &serde_json::Value) -> Option<serde_json::Value> {
    if is_deferred_tool(tool) {
        return None;
    }
    let mut fn_val = tool.get("function").unwrap_or(tool).clone();
    if let Some(obj) = fn_val.as_object_mut() {
        obj.remove("strict");
        obj.remove("defer_loading");
    }
    Some(fn_val)
}

/// Extract per-function parameter schemas from OpenAI `tools` definitions for typed tool-arg decoding.
pub fn extract_tool_schemas(tools: Option<&[serde_json::Value]>) -> ToolParamSchema {
    let mut out = ToolParamSchema::new();
    let Some(list) = tools else {
        return out;
    };
    for t in list {
        if is_deferred_tool(t) {
            continue;
        }
        let fn_obj = t.get("function").unwrap_or(t);
        let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let mut param_map = HashMap::new();
        if let Some(props) = fn_obj
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (param_name, schema_val) in props {
                let ty = match schema_val.get("type") {
                    Some(serde_json::Value::String(s)) => Some(s.clone()),
                    Some(serde_json::Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|v| v.as_str())
                        .find(|&s| s != "null")
                        .map(|s| s.to_owned()),
                    _ => None,
                };
                if let Some(ty_str) = ty {
                    param_map.insert(param_name.clone(), ty_str);
                }
            }
        }
        out.insert(name.to_owned(), param_map);
    }
    out
}

/// Parse an assistant `tool_calls` entry into `(call_id, function_name, arguments_value)`.
pub fn parse_tool_call_entry(entry: &serde_json::Value) -> Option<(Option<String>, String, serde_json::Value)> {
    let id = entry.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
    let fn_obj = entry.get("function").unwrap_or(entry);
    let name = fn_obj.get("name").and_then(|v| v.as_str())?.to_string();
    let args = match fn_obj.get("arguments") {
        None | Some(serde_json::Value::Null) => serde_json::json!({}),
        Some(serde_json::Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::Value::String(s.clone()))
            }
        }
        Some(other) => other.clone(),
    };
    Some((id, name, args))
}

/// Resolve Inkling-2-Mini thinking effort label (`"0"`, `"0.2"`, `"0.5"`, `"1.0"`, etc.) from `reasoning_effort`.
pub fn inkling2_effort_str(reasoning_effort: Option<&serde_json::Value>) -> String {
    match reasoning_effort {
        None | Some(serde_json::Value::Null) => "0".to_owned(),
        Some(serde_json::Value::Bool(false)) => "0".to_owned(),
        Some(serde_json::Value::Bool(true)) => "0.5".to_owned(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "disabled" | "false" | "0" | "0.0" => "0".to_owned(),
            "low" | "minimal" => "0.2".to_owned(),
            "medium" | "med" => "0.5".to_owned(),
            "high" | "max" => "1.0".to_owned(),
            _ => {
                if s.trim().parse::<f64>().is_ok() {
                    s.trim().to_owned()
                } else {
                    "0".to_owned()
                }
            }
        },
        _ => "0".to_owned(),
    }
}

/// Format OpenAI chat messages using Inkling-2-Mini's native chat template tokens so
/// multi-turn conversations strictly extend the exact BPE prefix for KV+conv reuse.
/// Returns `(rendered_prompt, start_in_text)`.
pub fn format_inkling2_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    reasoning_effort: Option<&serde_json::Value>,
) -> (String, bool) {
    let effort = inkling2_effort_str(reasoning_effort);
    let tc_mode = parse_tool_choice(tool_choice);
    let active_tools: Vec<serde_json::Value> = if tc_mode == ToolChoiceMode::None {
        Vec::new()
    } else {
        tools
            .unwrap_or(&[])
            .iter()
            .filter_map(sanitize_tool_fn_object)
            .collect()
    };
    let has_tools = !active_tools.is_empty();
    let thinking_enabled = effort != "0";

    let mut out = String::new();
    if has_tools {
        if let Ok(json_arr) = serde_json::to_string(&active_tools) {
            out.push_str("<|message_system|>tool_declare<|content_xml|>");
            out.push_str(&json_arr);
            out.push_str("<|end_message|>");
        }
    }

    let mut effort_emitted = false;
    let mut call_id_to_name: HashMap<String, String> = HashMap::new();

    for msg in messages {
        if !effort_emitted && msg.role != "system" && msg.role != "developer" {
            out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
            out.push_str(&effort);
            out.push_str("<|end_message|>");
            effort_emitted = true;
        }

        let visible = extract_visible_text(msg);
        match msg.role.as_str() {
            "system" | "developer" => {
                out.push_str("<|message_system|><|content_text|>");
                if !effort_emitted {
                    out.push_str("Thinking effort level: ");
                    out.push_str(&effort);
                    if !visible.is_empty() {
                        out.push_str("\n\n");
                    }
                    effort_emitted = true;
                }
                out.push_str(&visible);
                out.push_str("<|end_message|>");
            }
            "user" => {
                out.push_str("<|message_user|><|content_text|>");
                out.push_str(&visible);
                out.push_str("<|end_message|>");
            }
            "assistant" => {
                let mut had_segment = false;
                if let Some(rc) = msg.reasoning_content.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                    out.push_str("<|message_model|><|content_thinking|>");
                    out.push_str(rc);
                    out.push_str("<|end_message|>");
                    had_segment = true;
                }
                if !visible.trim().is_empty() || (msg.tool_calls.as_deref().unwrap_or(&[]).is_empty() && !had_segment) {
                    out.push_str("<|message_model|><|content_text|>");
                    out.push_str(&visible);
                    out.push_str("<|end_message|>");
                    had_segment = true;
                }
                if let Some(tcs) = msg.tool_calls.as_deref() {
                    for tc in tcs {
                        if let Some((id_opt, name, args)) = parse_tool_call_entry(tc) {
                            if let Some(cid) = id_opt {
                                call_id_to_name.insert(cid, name.clone());
                            }
                            if let (Ok(name_json), Ok(args_json)) =
                                (serde_json::to_string(&name), serde_json::to_string(&args))
                            {
                                out.push_str("<|message_model|>");
                                out.push_str(&name);
                                out.push_str("<|content_invoke_tool_json|>{\"name\":");
                                out.push_str(&name_json);
                                out.push_str(",\"args\":");
                                out.push_str(&args_json);
                                out.push_str("}<|end_message|>");
                                had_segment = true;
                            }
                        }
                    }
                }
                if had_segment {
                    out.push_str("<|content_model_end_sampling|>");
                }
            }
            "tool" => {
                let fn_name = msg
                    .name
                    .as_deref()
                    .or_else(|| {
                        msg.tool_call_id
                            .as_deref()
                            .and_then(|cid| call_id_to_name.get(cid).map(String::as_str))
                    })
                    .unwrap_or("");
                out.push_str("<|message_tool|>");
                out.push_str(fn_name);
                out.push_str("<|content_text|>");
                out.push_str(&visible);
                out.push_str("<|end_message|>");
            }
            _ => {
                out.push_str("<|message_user|><|content_text|>");
                out.push_str(&visible);
                out.push_str("<|end_message|>");
            }
        }
    }

    if !effort_emitted {
        out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        out.push_str(&effort);
        out.push_str("<|end_message|>");
    }

    let start_in_text = !has_tools && !thinking_enabled;
    out.push_str("<|message_model|>");
    if start_in_text {
        out.push_str("<|content_text|>");
    }

    (out, start_in_text)
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

/// Render GLM-5.3-Flash `<tools>...</tools>` system prompt section from sanitized OpenAI `tools`.
pub fn format_glm5_tool_definitions(tools: &[serde_json::Value], tool_choice: &ToolChoiceMode) -> String {
    let mut out = String::from(
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        if let Some(clean_fn) = sanitize_tool_fn_object(t) {
            if let Ok(line) = serde_json::to_string(&clean_fn) {
                out.push_str(&line);
                out.push('\n');
            }
        }
    }
    out.push_str(
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value></tool_call>",
    );
    match tool_choice {
        ToolChoiceMode::Required => {
            out.push_str("\n\nYou must call at least one of the provided tools in your response.");
        }
        ToolChoiceMode::Function(name) => {
            out.push_str("\n\nYou must call the function `");
            out.push_str(name);
            out.push_str("` in your response.");
        }
        _ => {}
    }
    out
}

/// Format an OpenAI tool argument JSON value for GLM-5 `<arg_value>...</arg_value>`.
fn format_glm5_arg_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Order a contiguous run of `role == "tool"` messages to match the preceding assistant's `tool_calls` IDs.
fn order_tool_responses<'a>(
    run: &'a [impl std::borrow::Borrow<ChatMessage>],
    last_assistant_call_ids: &[String],
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..run.len()).collect();
    if !last_assistant_call_ids.is_empty()
        && last_assistant_call_ids.len() == run.len()
        && run.iter().all(|m| {
            m.borrow()
                .tool_call_id
                .as_deref()
                .is_some_and(|id| last_assistant_call_ids.iter().any(|x| x == id))
        })
    {
        order.sort_by_key(|&i| {
            let cid = run[i].borrow().tool_call_id.as_deref().unwrap_or("");
            last_assistant_call_ids
                .iter()
                .position(|x| x == cid)
                .unwrap_or(usize::MAX)
        });
        // Fall back to original order if IDs are ambiguous/duplicated
        let mut distinct = order.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != run.len() {
            return (0..run.len()).collect();
        }
    }
    order
}

/// Resolve Hy4-Flash `(enable_thinking, effort_label)` (`"no_think"`, `"high"`, or `"max"`).
pub fn resolve_hy4_thinking(
    chat_template_kwargs: Option<&serde_json::Value>,
    reasoning_effort: Option<&serde_json::Value>,
) -> (bool, &'static str) {
    let mut off = false;
    let mut effort_str: Option<&str> = None;

    if let Some(kw) = chat_template_kwargs.and_then(|v| v.as_object()) {
        if kw.get("enable_thinking").and_then(|v| v.as_bool()) == Some(false) {
            off = true;
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
            "none" | "off" | "no_think" | "disabled" | "minimal" | "false" => off = true,
            _ => {}
        }
    }

    if off {
        (false, "no_think")
    } else if effort_str.is_some_and(|s| s.trim().eq_ignore_ascii_case("max")) {
        (true, "max")
    } else {
        (true, "high")
    }
}

fn render_multimodal_content_value(
    val: Option<&serde_json::Value>,
    img_placeholder: &str,
    video_err: &str,
    has_vision: bool,
    is_system: bool,
    image_sources: &mut Vec<String>,
) -> Result<String, String> {
    match val {
        None | Some(serde_json::Value::Null) => Ok(String::new()),
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(serde_json::Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(s) = p.as_str() {
                    out.push_str(s);
                    continue;
                }
                let ty = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if ty == "text" {
                    if let Some(txt) = p.get("text").and_then(|v| v.as_str()) {
                        out.push_str(txt);
                    }
                } else if ty == "image_url"
                    || ty == "image"
                    || p.get("image_url").is_some()
                    || p.get("image").is_some()
                {
                    if is_system {
                        return Err("System message cannot contain images.".to_owned());
                    }
                    if !has_vision {
                        return Err(
                            "Model loaded without vision weights; cannot process image inputs."
                                .to_owned(),
                        );
                    }
                    let url = p
                        .get("image_url")
                        .and_then(|u| u.get("url").and_then(|s| s.as_str()).or_else(|| u.as_str()))
                        .or_else(|| p.get("image").and_then(|s| s.as_str()))
                        .or_else(|| p.get("url").and_then(|s| s.as_str()))
                        .ok_or_else(|| "Malformed image_url part in chat message.".to_owned())?;
                    image_sources.push(url.to_owned());
                    out.push_str(img_placeholder);
                } else if ty == "video_url"
                    || ty == "video"
                    || p.get("video_url").is_some()
                    || p.get("video").is_some()
                {
                    if is_system {
                        return Err("System message cannot contain videos.".to_owned());
                    }
                    return Err(video_err.to_owned());
                } else if let Some(txt) = p.get("text").and_then(|v| v.as_str()) {
                    out.push_str(txt);
                } else if let Some(txt) = p.get("content").and_then(|v| v.as_str()) {
                    out.push_str(txt);
                } else {
                    out.push_str(&p.to_string());
                }
            }
            Ok(out)
        }
        Some(other) => Ok(serde_json::to_string(other).unwrap_or_default()),
    }
}

fn render_hy4_content_value(
    val: Option<&serde_json::Value>,
    hytk: &str,
    has_vision: bool,
    is_system: bool,
    image_sources: &mut Vec<String>,
) -> Result<String, String> {
    let img_placeholder =
        format!("<｜hy_image▁start{hytk}｜><｜hy_image▁pad{hytk}｜><｜hy_image▁end{hytk}｜>");
    render_multimodal_content_value(
        val,
        &img_placeholder,
        "Video inputs are not supported by ampllm Hy4 server.",
        has_vision,
        is_system,
        image_sources,
    )
}

fn format_hy4_tools_prompt(
    tools: &[serde_json::Value],
    tool_choice: &ToolChoiceMode,
    hytk: &str,
) -> String {
    let mut out = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    let mut first = true;
    for t in tools {
        if let Some(clean_fn) = sanitize_tool_fn_object(t) {
            if let Ok(line) = serde_json::to_string(&clean_fn) {
                if !first {
                    out.push('\n');
                }
                first = false;
                out.push_str(&line);
            }
        }
    }
    out.push_str("\n</tools>\n\nFor function call returns, you should first print <tool_calls");
    out.push_str(hytk);
    out.push_str(">\nFor each function call, you should return object like:\n<tool_call");
    out.push_str(hytk);
    out.push_str(">{function-name}<arg_key");
    out.push_str(hytk);
    out.push_str(">{arg-key-1}</arg_key");
    out.push_str(hytk);
    out.push_str("><arg_value");
    out.push_str(hytk);
    out.push_str(">{arg-value-1}</arg_value");
    out.push_str(hytk);
    out.push_str("><arg_key");
    out.push_str(hytk);
    out.push_str(">{arg-key-2}</arg_key");
    out.push_str(hytk);
    out.push_str("><arg_value");
    out.push_str(hytk);
    out.push_str(">{arg-value-2}</arg_value");
    out.push_str(hytk);
    out.push_str(">...</tool_call");
    out.push_str(hytk);
    out.push_str(">\nAt the end of function call returns, you should print </tool_calls");
    out.push_str(hytk);
    out.push('>');
    match tool_choice {
        ToolChoiceMode::Required => {
            out.push_str("\n\nYou must call at least one of the provided tools in your response.");
        }
        ToolChoiceMode::Function(name) => {
            out.push_str("\n\nYou must call the function `");
            out.push_str(name);
            out.push_str("` in your response.");
        }
        _ => {}
    }
    out
}

/// Format OpenAI chat messages using Hy4-Flash's `:opensource` Jinja template semantics
/// (including tool calls, out-of-order tool responses, `reasoning_effort`, and vision image spans).
/// Returns `Ok((rendered_prompt, expect_think_close, forced_prefill, image_sources))`.
pub fn format_hy4_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    enable_thinking: bool,
    effort_label: &str,
    hytk: &str,
    has_vision: bool,
) -> Result<(String, bool, String, Vec<String>), String> {
    let hy_start = format!("<｜hy_start{hytk}｜>");
    let hy_middle = format!("<｜hy_middle{hytk}｜>");
    let hy_end = format!("<｜hy_end{hytk}｜>");
    let think_begin = format!("<think{hytk}>");
    let think_end = format!("</think{hytk}>");
    let toolcalls_begin = format!("<tool_calls{hytk}>");
    let toolcalls_end = format!("</tool_calls{hytk}>");
    let toolcall_begin = format!("<tool_call{hytk}>");
    let toolcall_end = format!("</tool_call{hytk}>");
    let argkey_begin = format!("<arg_key{hytk}>");
    let argkey_end = format!("</arg_key{hytk}>");
    let argvalue_begin = format!("<arg_value{hytk}>");
    let argvalue_end = format!("</arg_value{hytk}>");
    let toolresp_begin = format!("<tool_response{hytk}>");
    let toolresp_end = format!("</tool_response{hytk}>");
    let reasoning_mode = format!("<｜reasoning_mode{hytk}｜>");

    let tc_mode = parse_tool_choice(tool_choice);
    let active_tools: Vec<serde_json::Value> = if tc_mode == ToolChoiceMode::None {
        Vec::new()
    } else {
        tools
            .unwrap_or(&[])
            .iter()
            .filter(|t| !is_deferred_tool(t))
            .cloned()
            .collect()
    };
    let has_tools = !active_tools.is_empty();
    let preserved_thinking = has_tools;

    // Group consecutive tool turns to match `merge_tool_responses` so `last_user_index` matches Jinja
    enum NormItem<'a> {
        Single(&'a ChatMessage),
        ToolCluster(&'a [ChatMessage]),
    }
    let mut normed = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == "tool" {
            let mut j = i + 1;
            while j < messages.len() && messages[j].role == "tool" {
                j += 1;
            }
            normed.push(NormItem::ToolCluster(&messages[i..j]));
            i = j;
        } else {
            normed.push(NormItem::Single(&messages[i]));
            i += 1;
        }
    }

    let has_leading_system = matches!(
        normed.first(),
        Some(NormItem::Single(m)) if m.role == "system" || m.role == "developer"
    );
    let last_user_idx = normed
        .iter()
        .rposition(|item| matches!(item, NormItem::Single(m) if m.role == "user"));

    let mut out = String::new();
    let mut image_sources = Vec::new();

    if !has_leading_system {
        out.push_str(&hy_start);
        out.push_str("system");
        out.push_str(&hy_middle);
        if has_tools {
            out.push_str(&format_hy4_tools_prompt(&active_tools, &tc_mode, hytk));
        }
        out.push_str(&reasoning_mode);
        out.push_str("reasoning_effort:");
        out.push_str(effort_label);
        out.push_str(&hy_end);
    }

    let mut last_assistant_call_ids: Vec<String> = Vec::new();
    for (idx, item) in normed.iter().enumerate() {
        match item {
            NormItem::Single(msg) => {
                let role = if msg.role == "developer" {
                    "system"
                } else {
                    msg.role.as_str()
                };
                match role {
                    "system" => {
                        let body = render_hy4_content_value(
                            msg.content.as_ref(),
                            hytk,
                            has_vision,
                            true,
                            &mut image_sources,
                        )?;
                        out.push_str(&hy_start);
                        out.push_str("system");
                        out.push_str(&hy_middle);
                        if idx == 0 {
                            if has_tools {
                                out.push_str(&format_hy4_tools_prompt(&active_tools, &tc_mode, hytk));
                                if !body.is_empty() {
                                    out.push_str("\n\n");
                                }
                            }
                            out.push_str(&body);
                            out.push_str(&reasoning_mode);
                            out.push_str("reasoning_effort:");
                            out.push_str(effort_label);
                        } else {
                            out.push_str(&body);
                        }
                        out.push_str(&hy_end);
                        last_assistant_call_ids.clear();
                    }
                    "user" => {
                        let body = render_hy4_content_value(
                            msg.content.as_ref(),
                            hytk,
                            has_vision,
                            false,
                            &mut image_sources,
                        )?;
                        out.push_str(&hy_start);
                        out.push_str("user");
                        out.push_str(&hy_middle);
                        out.push_str(&body);
                        out.push_str(&hy_end);
                        last_assistant_call_ids.clear();
                    }
                    "assistant" => {
                        let content_body = render_hy4_content_value(
                            msg.content.as_ref(),
                            hytk,
                            has_vision,
                            false,
                            &mut image_sources,
                        )?;
                        let reasoning_text = msg
                            .reasoning
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .or_else(|| msg.reasoning_content.as_deref().filter(|s| !s.is_empty()));
                        let after_last_user = last_user_idx.map_or(true, |lu| idx > lu);

                        out.push_str(&hy_start);
                        out.push_str("assistant");
                        out.push_str(&hy_middle);
                        out.push_str(&think_begin);
                        if effort_label != "no_think" && (preserved_thinking || after_last_user) {
                            if let Some(rt) = reasoning_text {
                                out.push_str(rt);
                            }
                        }
                        out.push_str(&think_end);
                        out.push_str(&content_body);

                        last_assistant_call_ids.clear();
                        if let Some(tcs) = msg.tool_calls.as_deref().filter(|t| !t.is_empty()) {
                            out.push_str(&toolcalls_begin);
                            for tc in tcs {
                                if let Some((id_opt, name, args)) = parse_tool_call_entry(tc) {
                                    if let Some(cid) = id_opt {
                                        last_assistant_call_ids.push(cid);
                                    }
                                    out.push_str(&toolcall_begin);
                                    out.push_str(&name);
                                    if let Some(map) = args.as_object() {
                                        for (k, v) in map {
                                            out.push_str(&argkey_begin);
                                            out.push_str(k);
                                            out.push_str(&argkey_end);
                                            out.push_str(&argvalue_begin);
                                            out.push_str(&format_glm5_arg_value(v));
                                            out.push_str(&argvalue_end);
                                        }
                                    }
                                    out.push_str(&toolcall_end);
                                }
                            }
                            out.push_str(&toolcalls_end);
                        }
                        out.push_str(&hy_end);
                    }
                    other => {
                        let body = render_hy4_content_value(
                            msg.content.as_ref(),
                            hytk,
                            has_vision,
                            false,
                            &mut image_sources,
                        )?;
                        out.push_str(&hy_start);
                        out.push_str(other);
                        out.push_str(&hy_middle);
                        out.push_str(&body);
                        out.push_str(&hy_end);
                        last_assistant_call_ids.clear();
                    }
                }
            }
            NormItem::ToolCluster(run) => {
                let order = order_tool_responses(run, &last_assistant_call_ids);
                out.push_str(&hy_start);
                out.push_str("tool");
                out.push_str(&hy_middle);
                for &k in &order {
                    out.push_str(&toolresp_begin);
                    let resp_body = match run[k].content.as_ref() {
                        Some(serde_json::Value::Array(items)) => {
                            let mut s = String::new();
                            for item in items {
                                if let Some(output_val) = item.get("output") {
                                    s.push_str(&render_hy4_content_value(
                                        Some(output_val),
                                        hytk,
                                        has_vision,
                                        false,
                                        &mut image_sources,
                                    )?);
                                } else {
                                    let single = serde_json::Value::Array(vec![item.clone()]);
                                    s.push_str(&render_hy4_content_value(
                                        Some(&single),
                                        hytk,
                                        has_vision,
                                        false,
                                        &mut image_sources,
                                    )?);
                                }
                            }
                            s
                        }
                        other => render_hy4_content_value(
                            other,
                            hytk,
                            has_vision,
                            false,
                            &mut image_sources,
                        )?,
                    };
                    out.push_str(&resp_body);
                    out.push_str(&toolresp_end);
                }
                out.push_str(&hy_end);
                last_assistant_call_ids.clear();
            }
        }
    }

    let forced_prefill = match tc_mode {
        ToolChoiceMode::Function(fn_name) => format!("<tool_call>{fn_name}"),
        _ => String::new(),
    };

    if !forced_prefill.is_empty() {
        let fn_name = forced_prefill.strip_prefix("<tool_call>").unwrap_or("");
        out.push_str(&hy_start);
        out.push_str("assistant");
        out.push_str(&hy_middle);
        out.push_str(&think_begin);
        out.push_str(&think_end);
        out.push_str(&toolcalls_begin);
        out.push_str(&toolcall_begin);
        out.push_str(fn_name);
        Ok((out, false, forced_prefill, image_sources))
    } else if enable_thinking {
        out.push_str(&hy_start);
        out.push_str("assistant");
        out.push_str(&hy_middle);
        out.push_str(&think_begin);
        Ok((out, true, String::new(), image_sources))
    } else {
        out.push_str(&hy_start);
        out.push_str("assistant");
        out.push_str(&hy_middle);
        out.push_str(&think_begin);
        out.push_str(&think_end);
        Ok((out, false, String::new(), image_sources))
    }
}

/// Resolve Qwen3.8-Flash-Next `(enable_thinking, effort_label)` (`"xhigh"`, `"medium"`, or `"low"`).
pub fn resolve_qwen4_thinking(
    chat_template_kwargs: Option<&serde_json::Value>,
    reasoning_effort: Option<&serde_json::Value>,
) -> Result<(bool, &'static str), String> {
    let mut enable_thinking = true;
    let mut effort_str: Option<&str> = None;

    if let Some(kw) = chat_template_kwargs.and_then(|v| v.as_object()) {
        if let Some(et) = kw.get("enable_thinking").and_then(|v| v.as_bool()) {
            enable_thinking = et;
        }
        if let Some(re) = kw.get("reasoning_effort").and_then(|v| v.as_str()) {
            effort_str = Some(re);
        }
    }
    if let Some(re) = reasoning_effort {
        if re == &serde_json::Value::Bool(false) {
            enable_thinking = false;
        } else if let Some(s) = re.as_str() {
            effort_str = Some(s);
        }
    }

    if let Some(raw) = effort_str {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "no_think" | "disabled" | "false" => return Ok((false, "")),
            _ if !enable_thinking => return Ok((false, "")),
            "high" | "xhigh" | "max" => return Ok((true, "xhigh")),
            "medium" => return Ok((true, "medium")),
            "low" | "minimal" => return Ok((true, "low")),
            _ => {
                return Err(format!(
                    "Unexpected reasoning effort {raw}. Supported types are xhigh (default), medium, and low."
                ))
            }
        }
    }
    if !enable_thinking {
        Ok((false, ""))
    } else {
        Ok((true, "xhigh"))
    }
}

fn format_qwen4_tool_object(tool: &serde_json::Value) -> Option<serde_json::Value> {
    if is_deferred_tool(tool) {
        return None;
    }
    if tool.get("function").is_some() {
        let mut obj = tool.clone();
        if let Some(m) = obj.as_object_mut() {
            m.remove("defer_loading");
        }
        Some(obj)
    } else {
        let fn_val = sanitize_tool_fn_object(tool)?;
        Some(serde_json::json!({
            "type": "function",
            "function": fn_val
        }))
    }
}

fn format_qwen4_tools_prompt(
    tools: &[serde_json::Value],
    tool_choice: &ToolChoiceMode,
) -> String {
    let mut out = String::from(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>",
    );
    for t in tools {
        if let Some(obj) = format_qwen4_tool_object(t) {
            if let Ok(line) = serde_json::to_string(&obj) {
                out.push('\n');
                out.push_str(&line);
            }
        }
    }
    out.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
    );
    match tool_choice {
        ToolChoiceMode::Required => {
            out.push_str("\n\nYou must call at least one of the provided tools in your response.");
        }
        ToolChoiceMode::Function(name) => {
            out.push_str("\n\nYou must call the function `");
            out.push_str(name);
            out.push_str("` in your response.");
        }
        _ => {}
    }
    out
}

/// Format OpenAI chat messages using Qwen3.8-Flash-Next's GGUF `tokenizer.chat_template`.
/// Returns `Ok((rendered_prompt, expect_think_close, forced_prefill, image_sources))`.
pub fn format_qwen4_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    enable_thinking: bool,
    effort_label: &str,
    has_vision: bool,
) -> Result<(String, bool, String, Vec<String>), String> {
    if messages.is_empty() {
        return Err("No messages provided.".to_owned());
    }
    const IMG_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    const VIDEO_ERR: &str = "Video inputs are not supported by ampllm Qwen4 server.";

    let tc_mode = parse_tool_choice(tool_choice);
    let active_tools: Vec<serde_json::Value> = if tc_mode == ToolChoiceMode::None {
        Vec::new()
    } else {
        tools
            .unwrap_or(&[])
            .iter()
            .filter(|t| !is_deferred_tool(t))
            .cloned()
            .collect()
    };
    let has_tools = !active_tools.is_empty();

    let mut num_sys = 0usize;
    let mut merged_system = String::new();
    let mut dummy_imgs = Vec::new();
    for msg in messages {
        if msg.role == "system" || msg.role == "developer" {
            let rendered = render_multimodal_content_value(
                msg.content.as_ref(),
                IMG_PLACEHOLDER,
                VIDEO_ERR,
                has_vision,
                true,
                &mut dummy_imgs,
            )?;
            let trimmed = rendered.trim();
            if !trimmed.is_empty() {
                if !merged_system.is_empty() {
                    merged_system.push('\n');
                }
                merged_system.push_str(trimmed);
            }
            num_sys += 1;
        } else {
            break;
        }
    }

    let reasoning_instructions = if enable_thinking {
        match effort_label {
            "xhigh" | "high" => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            "low" => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            _ => "",
        }
    } else {
        ""
    };

    let mut out = String::new();
    if has_tools {
        out.push_str("<|im_start|>system\n");
        if !reasoning_instructions.is_empty() {
            out.push_str(reasoning_instructions);
            out.push_str("\n\n");
        }
        out.push_str(&format_qwen4_tools_prompt(&active_tools, &tc_mode));
        if !merged_system.is_empty() {
            out.push_str("\n\n");
            out.push_str(&merged_system);
        }
        out.push_str("<|im_end|>\n");
    } else if !merged_system.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !reasoning_instructions.is_empty() {
            out.push_str(reasoning_instructions);
            out.push_str("\n\n");
        }
        out.push_str(&merged_system);
        out.push_str("<|im_end|>\n");
    } else if !reasoning_instructions.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(reasoning_instructions);
        out.push_str("<|im_end|>\n");
    }

    let mut image_sources = Vec::new();
    let tail = &messages[num_sys..];
    for (ti, msg) in tail.iter().enumerate() {
        let rendered = render_multimodal_content_value(
            msg.content.as_ref(),
            IMG_PLACEHOLDER,
            VIDEO_ERR,
            has_vision,
            false,
            &mut image_sources,
        )?;
        let content = rendered.trim();
        match msg.role.as_str() {
            "system" | "developer" => {
                return Err("System message must be at the beginning.".to_owned());
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                let reasoning_content = msg
                    .reasoning
                    .as_deref()
                    .or(msg.reasoning_content.as_deref())
                    .unwrap_or("")
                    .trim();
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(reasoning_content);
                out.push_str("\n</think>\n\n");
                out.push_str(content);

                if let Some(tcs) = msg.tool_calls.as_deref().filter(|t| !t.is_empty()) {
                    for (ci, tc) in tcs.iter().enumerate() {
                        if let Some((_id_opt, name, args)) = parse_tool_call_entry(tc) {
                            if ci == 0 {
                                if !content.is_empty() {
                                    out.push_str("\n\n");
                                }
                            } else {
                                out.push('\n');
                            }
                            out.push_str("<tool_call>\n<function=");
                            out.push_str(&name);
                            out.push_str(">\n");
                            if let Some(map) = args.as_object() {
                                for (k, v) in map {
                                    out.push_str("<parameter=");
                                    out.push_str(k);
                                    out.push_str(">\n");
                                    out.push_str(&format_glm5_arg_value(v));
                                    out.push_str("\n</parameter>\n");
                                }
                            }
                            out.push_str("</function>\n</tool_call>");
                        }
                    }
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                let prev_is_tool = ti > 0 && tail[ti - 1].role == "tool";
                if !prev_is_tool {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                let next_is_tool = ti + 1 < tail.len() && tail[ti + 1].role == "tool";
                if !next_is_tool {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => return Err("Unexpected message role.".to_owned()),
        }
    }

    let forced_prefill = match tc_mode {
        ToolChoiceMode::Function(fn_name) => format!("<tool_call>\n<function={fn_name}>\n"),
        _ => String::new(),
    };

    out.push_str("<|im_start|>assistant\n");
    if !forced_prefill.is_empty() {
        out.push_str("<think>\n\n</think>\n\n");
        out.push_str(&forced_prefill);
        Ok((out, false, forced_prefill, image_sources))
    } else if enable_thinking {
        out.push_str("<think>\n");
        Ok((out, true, String::new(), image_sources))
    } else {
        out.push_str("<think>\n\n</think>\n\n");
        Ok((out, false, String::new(), image_sources))
    }
}

fn to_mm3_xml(val: &serde_json::Value, ns: &str) -> String {
    match val {
        serde_json::Value::Object(map) => {
            let mut s = String::new();
            for (k, v) in map {
                if v.is_null() {
                    continue;
                }
                s.push_str(ns);
                s.push('<');
                s.push_str(k);
                s.push('>');
                s.push_str(&to_mm3_xml(v, ns));
                s.push_str(ns);
                s.push_str("</");
                s.push_str(k);
                s.push('>');
            }
            s
        }
        serde_json::Value::Array(arr) => {
            let mut s = String::new();
            for item in arr {
                s.push_str(ns);
                s.push_str("<item>");
                s.push_str(&to_mm3_xml(item, ns));
                s.push_str(ns);
                s.push_str("</item>");
            }
            s
        }
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::String(st) => st.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn render_mm3_visible_text(
    val: Option<&serde_json::Value>,
    has_vision: bool,
    is_system: bool,
    image_sources: &mut Vec<String>,
) -> Result<String, String> {
    render_multimodal_content_value(
        val,
        "]<]image[>[",
        "Video inputs are not supported by ampllm MM3 server.",
        has_vision,
        is_system,
        image_sources,
    )
}

/// Resolve MiniMax-M3.1 `(thinking_mode, reasoning_effort)` (`"enabled"`, `"disabled"`, `"adaptive"`).
pub fn resolve_mm3_thinking(
    chat_template_kwargs: Option<&serde_json::Value>,
    reasoning_effort: Option<&serde_json::Value>,
) -> (&'static str, Option<String>) {
    let mut mode: Option<&str> = None;
    let mut enable_thinking: Option<bool> = None;
    let mut effort: Option<String> = None;

    if let Some(kw) = chat_template_kwargs.and_then(|v| v.as_object()) {
        if let Some(tm) = kw.get("thinking_mode").and_then(|v| v.as_str()) {
            mode = Some(tm);
        }
        if let Some(et) = kw.get("enable_thinking").and_then(|v| v.as_bool()) {
            enable_thinking = Some(et);
        }
        if let Some(re) = kw.get("reasoning_effort").and_then(|v| v.as_str()) {
            effort = Some(re.trim().to_owned());
        }
    }
    if let Some(re) = reasoning_effort {
        if re == &serde_json::Value::Bool(false) {
            enable_thinking = Some(false);
        } else if let Some(s) = re.as_str() {
            let low = s.trim().to_ascii_lowercase();
            match low.as_str() {
                "none" | "off" | "disabled" | "false" => enable_thinking = Some(false),
                _ => effort = Some(s.trim().to_owned()),
            }
        }
    }

    let resolved_mode = if let Some(tm) = mode {
        match tm.trim().to_ascii_lowercase().as_str() {
            "disabled" | "off" | "none" => "disabled",
            "enabled" | "on" => "enabled",
            _ => "adaptive",
        }
    } else if let Some(et) = enable_thinking {
        if et { "enabled" } else { "disabled" }
    } else {
        "adaptive"
    };

    (resolved_mode, effort)
}

/// Format OpenAI chat messages using MiniMax-M3.1-Flash's exact `chat_template.jinja`.
/// Returns `Ok((rendered_prompt, expect_think_close, forced_prefill, image_sources))`.
pub fn format_mm3_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    thinking_mode: &str,
    reasoning_effort: Option<&str>,
    has_vision: bool,
) -> Result<(String, bool, String, Vec<String>), String> {
    const NS: &str = "]<]minimax[>[";
    const BOD: &str = "]~!b[";
    const BOS: &str = "]~b]";
    const EOS: &str = "[e~[";
    const TC_BEGIN: &str = "]<]minimax[>[<tool_call>";
    const TC_END: &str = "]<]minimax[>[</tool_call>";
    const THINK_BEGIN: &str = "<mm:think>";
    const THINK_END: &str = "</mm:think>";

    let tc_mode = parse_tool_choice(tool_choice);
    let active_tools: Vec<serde_json::Value> = if tc_mode == ToolChoiceMode::None {
        Vec::new()
    } else {
        tools
            .unwrap_or(&[])
            .iter()
            .filter_map(format_qwen4_tool_object)
            .collect()
    };

    let mut image_sources = Vec::new();
    let mut system_msg: Option<&ChatMessage> = None;
    let mut developer_msg: Option<&ChatMessage> = None;
    let mut conv = messages;

    if let Some(first) = conv.first() {
        if first.role == "root" {
            system_msg = Some(first);
            conv = &conv[1..];
            if let Some(second) = conv.first() {
                if second.role == "system" || second.role == "developer" {
                    developer_msg = Some(second);
                    conv = &conv[1..];
                }
            }
        } else if first.role == "system" || first.role == "developer" {
            developer_msg = Some(first);
            conv = &conv[1..];
        }
    }

    let mut out = String::new();
    out.push_str(BOD);
    out.push_str(BOS);
    out.push_str("system\n");
    if let Some(sm) = system_msg {
        let body = render_mm3_visible_text(sm.content.as_ref(), has_vision, true, &mut image_sources)?;
        if !body.is_empty() {
            out.push_str(&body);
        } else {
            out.push_str("Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.");
        }
    } else {
        out.push_str("Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.");
    }
    out.push_str("\n\n<thinking_instructions>\n");
    out.push_str("You have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n");
    match thinking_mode {
        "enabled" => out.push_str("Current thinking mode: enabled. You MUST think step by step before every response, including after receiving function/tool results.\n"),
        "disabled" => out.push_str("Current thinking mode: disabled. Do not output any thinking process.\n"),
        _ => out.push_str("Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n"),
    }
    out.push_str("</thinking_instructions>\n\n<model_id>8bb8149fe</model_id>");
    if let Some(eff) = reasoning_effort.filter(|s| !s.is_empty()) {
        out.push_str("\n\n<effort>");
        out.push_str(eff);
        out.push_str("</effort>");
    }
    out.push_str(EOS);
    out.push('\n');

    out.push_str(BOS);
    out.push_str("developer\n");
    if let Some(dm) = developer_msg {
        let body = render_mm3_visible_text(dm.content.as_ref(), has_vision, true, &mut image_sources)?;
        if !body.is_empty() {
            out.push_str(&body);
        } else {
            out.push_str("You are a helpful assistant.");
        }
    } else {
        out.push_str("You are a helpful assistant.");
    }
    if !active_tools.is_empty() {
        out.push_str("\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n");
        for t in &active_tools {
            if let Some(fn_obj) = t.get("function") {
                if let Ok(line) = serde_json::to_string(fn_obj) {
                    out.push_str("<tool>");
                    out.push_str(&line);
                    out.push_str("</tool>\n");
                }
            }
        }
        out.push_str("</tools>\n\nTo call tools, wrap all invocations in a single ");
        out.push_str(TC_BEGIN);
        out.push_str(TC_END);
        out.push_str(" block. Parameter values containing nested objects or arrays are recursively expanded into XML elements. Example:\n\n");
        out.push_str(TC_BEGIN);
        out.push('\n');
        out.push_str(NS);
        out.push_str("<invoke name=\"tool-name-1\">");
        out.push_str(NS);
        out.push_str("<param-1>value-1");
        out.push_str(NS);
        out.push_str("</param-1>");
        out.push_str(NS);
        out.push_str("<param-2>");
        out.push_str(NS);
        out.push_str("<item>");
        out.push_str(NS);
        out.push_str("<key-a>val-a");
        out.push_str(NS);
        out.push_str("</key-a>");
        out.push_str(NS);
        out.push_str("<key-b>val-b");
        out.push_str(NS);
        out.push_str("</key-b>");
        out.push_str(NS);
        out.push_str("</item>");
        out.push_str(NS);
        out.push_str("</param-2>");
        out.push_str(NS);
        out.push_str("</invoke>\n");
        out.push_str(NS);
        out.push_str("<invoke name=\"tool-name-2\">");
        out.push_str(NS);
        out.push_str("<param-1>value-1");
        out.push_str(NS);
        out.push_str("</param-1>");
        out.push_str(NS);
        out.push_str("</invoke>\n");
        out.push_str(TC_END);
        match &tc_mode {
            ToolChoiceMode::Required => {
                out.push_str("\nYou must call at least one tool in your response.");
            }
            ToolChoiceMode::Function(name) => {
                out.push_str("\nYou must call the tool `");
                out.push_str(name);
                out.push_str("` in your response.");
            }
            _ => {}
        }
    }
    out.push_str(EOS);
    out.push('\n');

    let mut had_tool_call = false;
    for (ti, msg) in conv.iter().enumerate() {
        match msg.role.as_str() {
            "assistant" => {
                out.push_str(BOS);
                out.push_str("ai\n");
                let mut content =
                    render_mm3_visible_text(msg.content.as_ref(), has_vision, false, &mut image_sources)?;
                let mut reasoning = msg
                    .reasoning_content
                    .as_deref()
                    .or(msg.reasoning.as_deref())
                    .unwrap_or("")
                    .to_owned();
                if reasoning.is_empty() {
                    if let Some(end_idx) = content.find(THINK_END) {
                        let before = content[..end_idx].trim_matches('\n');
                        reasoning = before
                            .rsplit(THINK_BEGIN)
                            .next()
                            .unwrap_or(before)
                            .trim_matches('\n')
                            .to_owned();
                        content = content[end_idx + THINK_END.len()..]
                            .trim_matches('\n')
                            .to_owned();
                    }
                }
                if !reasoning.is_empty() {
                    out.push_str(THINK_BEGIN);
                    out.push_str(&reasoning);
                    out.push_str(THINK_END);
                } else {
                    out.push_str(THINK_END);
                }
                if !content.is_empty() {
                    out.push_str(&content);
                }
                if let Some(tcs) = msg.tool_calls.as_deref().filter(|t| !t.is_empty()) {
                    out.push_str(TC_BEGIN);
                    out.push('\n');
                    for tc in tcs {
                        if let Some((_id_opt, name, args)) = parse_tool_call_entry(tc) {
                            out.push_str(NS);
                            out.push_str("<invoke name=\"");
                            out.push_str(&name);
                            out.push_str("\">");
                            if let Some(map) = args.as_object() {
                                for (k, v) in map {
                                    if v.is_null() {
                                        continue;
                                    }
                                    out.push_str(NS);
                                    out.push('<');
                                    out.push_str(k);
                                    out.push('>');
                                    out.push_str(&to_mm3_xml(v, NS));
                                    out.push_str(NS);
                                    out.push_str("</");
                                    out.push_str(k);
                                    out.push('>');
                                }
                            }
                            out.push_str(NS);
                            out.push_str("</invoke>\n");
                        }
                    }
                    out.push_str(TC_END);
                    had_tool_call = true;
                } else {
                    had_tool_call = false;
                }
                out.push_str(EOS);
                out.push('\n');
            }
            "tool" => {
                if !had_tool_call {
                    return Err("Message has tool role, but there was no previous assistant message with a tool call!".to_owned());
                }
                if ti == 0 || conv[ti - 1].role != "tool" {
                    out.push_str(BOS);
                    out.push_str("tool");
                }
                out.push_str("\n<response>");
                let resp_body = match msg.content.as_ref() {
                    Some(serde_json::Value::Array(items)) => {
                        let mut s = String::new();
                        for tr in items {
                            if let Some(out_val) = tr.get("output") {
                                s.push_str(&render_mm3_visible_text(
                                    Some(out_val),
                                    has_vision,
                                    false,
                                    &mut image_sources,
                                )?);
                            } else {
                                let single = serde_json::Value::Array(vec![tr.clone()]);
                                s.push_str(&render_mm3_visible_text(
                                    Some(&single),
                                    has_vision,
                                    false,
                                    &mut image_sources,
                                )?);
                            }
                        }
                        s
                    }
                    other => render_mm3_visible_text(other, has_vision, false, &mut image_sources)?,
                };
                out.push_str(&resp_body);
                out.push_str("</response>");
                if ti + 1 == conv.len() || conv[ti + 1].role != "tool" {
                    out.push_str(EOS);
                    out.push('\n');
                }
            }
            "system" | "developer" => {
                out.push_str(BOS);
                out.push_str("developer\n");
                let body = render_mm3_visible_text(msg.content.as_ref(), has_vision, true, &mut image_sources)?;
                if !body.is_empty() {
                    out.push_str(&body);
                } else {
                    out.push_str("You are a helpful assistant.");
                }
                out.push_str(EOS);
                out.push('\n');
            }
            "user" => {
                out.push_str(BOS);
                out.push_str("user\n");
                let body = render_mm3_visible_text(msg.content.as_ref(), has_vision, false, &mut image_sources)?;
                out.push_str(&body);
                out.push_str(EOS);
                out.push('\n');
            }
            _ => return Err(format!("Unexpected message role `{}`.", msg.role)),
        }
    }

    let forced_prefill = match tc_mode {
        ToolChoiceMode::Function(fn_name) => {
            format!("{TC_BEGIN}\n{NS}<invoke name=\"{fn_name}\">")
        }
        _ => String::new(),
    };

    out.push_str(BOS);
    out.push_str("ai\n");
    if !forced_prefill.is_empty() {
        out.push_str(THINK_END);
        out.push_str(&forced_prefill);
        Ok((out, false, forced_prefill, image_sources))
    } else if thinking_mode == "disabled" {
        out.push_str(THINK_END);
        Ok((out, false, String::new(), image_sources))
    } else if thinking_mode == "enabled" {
        out.push_str(THINK_BEGIN);
        Ok((out, true, String::new(), image_sources))
    } else {
        Ok((out, false, String::new(), image_sources))
    }
}

/// Format OpenAI chat messages using GLM-5.3-Flash's exact GGUF chat_template (`[gMASK]<sop>...`).
fn render_glm5_content_value(
    val: Option<&serde_json::Value>,
    has_vision: bool,
    is_system: bool,
    image_sources: &mut Vec<String>,
) -> Result<String, String> {
    render_multimodal_content_value(
        val,
        "<|begin_of_image|><|image|><|end_of_image|>",
        "Video inputs are not supported by ampllm GLM-5 server.",
        has_vision,
        is_system,
        image_sources,
    )
}

/// Returns `Ok((rendered_prompt, expect_think_close, forced_prefill, image_sources))`.
pub fn format_glm5_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    enable_thinking: bool,
    effort_label: &str,
    has_vision: bool,
) -> Result<(String, bool, String, Vec<String>), String> {
    let tc_mode = parse_tool_choice(tool_choice);
    let mut out = String::new();
    let mut image_sources = Vec::new();
    out.push_str("[gMASK]<sop><|system|>Reasoning Effort: ");
    out.push_str(effort_label);

    let active_tools: Vec<&serde_json::Value> = if tc_mode == ToolChoiceMode::None {
        Vec::new()
    } else {
        tools
            .unwrap_or(&[])
            .iter()
            .filter(|t| !is_deferred_tool(t))
            .collect()
    };
    if !active_tools.is_empty() {
        let owned: Vec<serde_json::Value> = active_tools.iter().map(|t| (*t).clone()).collect();
        out.push_str(&format_glm5_tool_definitions(&owned, &tc_mode));
    }

    let mut last_assistant_call_ids: Vec<String> = Vec::new();
    let mut idx = 0;
    while idx < messages.len() {
        let msg = &messages[idx];
        match msg.role.as_str() {
            "system" | "developer" => {
                let raw = render_glm5_content_value(
                    msg.content.as_ref(),
                    has_vision,
                    true,
                    &mut image_sources,
                )?;
                out.push_str("<|system|>");
                out.push_str(&raw);
                idx += 1;
            }
            "user" => {
                let raw = render_glm5_content_value(
                    msg.content.as_ref(),
                    has_vision,
                    false,
                    &mut image_sources,
                )?;
                out.push_str("<|user|>");
                out.push_str(&raw);
                idx += 1;
            }
            "assistant" => {
                let raw = render_glm5_content_value(
                    msg.content.as_ref(),
                    has_vision,
                    false,
                    &mut image_sources,
                )?;
                out.push_str("<|assistant|>");
                if let Some(rc) = msg.reasoning_content.as_deref() {
                    out.push_str("<think>");
                    out.push_str(rc);
                    out.push_str("</think>");
                    let trimmed = raw.trim();
                    if !trimmed.is_empty() {
                        out.push_str(trimmed);
                    }
                } else if let Some(end_idx) = raw.find("</think>") {
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

                last_assistant_call_ids.clear();
                if let Some(tcs) = msg.tool_calls.as_deref() {
                    for tc in tcs {
                        if let Some((id_opt, name, args)) = parse_tool_call_entry(tc) {
                            if let Some(cid) = id_opt {
                                last_assistant_call_ids.push(cid);
                            }
                            out.push_str("<tool_call>");
                            out.push_str(&name);
                            if let Some(map) = args.as_object() {
                                for (k, v) in map {
                                    out.push_str("<arg_key>");
                                    out.push_str(k);
                                    out.push_str("</arg_key><arg_value>");
                                    out.push_str(&format_glm5_arg_value(v));
                                    out.push_str("</arg_value>");
                                }
                            }
                            out.push_str("</tool_call>");
                        }
                    }
                }
                idx += 1;
            }
            "tool" => {
                let mut end = idx + 1;
                while end < messages.len() && messages[end].role == "tool" {
                    end += 1;
                }
                let run = &messages[idx..end];
                let order = order_tool_responses(run, &last_assistant_call_ids);
                out.push_str("<|observation|>");
                for &i in &order {
                    let body = render_glm5_content_value(
                        run[i].content.as_ref(),
                        has_vision,
                        false,
                        &mut image_sources,
                    )?;
                    out.push_str("<tool_response>");
                    out.push_str(&body);
                    out.push_str("</tool_response>");
                }
                idx = end;
            }
            _ => {
                let raw = render_glm5_content_value(
                    msg.content.as_ref(),
                    has_vision,
                    false,
                    &mut image_sources,
                )?;
                out.push_str("<|user|>");
                out.push_str(&raw);
                idx += 1;
            }
        }
    }

    let forced_prefill = match tc_mode {
        ToolChoiceMode::Function(fn_name) => format!("<tool_call>{fn_name}"),
        _ => String::new(),
    };

    if !forced_prefill.is_empty() {
        out.push_str("<|assistant|><think></think>");
        out.push_str(&forced_prefill);
        Ok((out, false, forced_prefill, image_sources))
    } else if enable_thinking {
        out.push_str("<|assistant|><think>");
        Ok((out, true, String::new(), image_sources))
    } else {
        out.push_str("<|assistant|><think></think>");
        Ok((out, false, String::new(), image_sources))
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
            "owned_by": "ampllm",
            "context_length": st.max_ctx
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

pub async fn list_slots(State(st): State<AppState>) -> impl IntoResponse {
    let slots = st
        .slot_statuses
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();
    let queue_depth = st.pending_jobs.load(Ordering::Relaxed);
    Json(serde_json::json!({
        "parallel_slots": slots.len(),
        "queue_depth": queue_depth,
        "slots": slots,
    }))
}

pub async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let tc_mode = parse_tool_choice(req.tool_choice.as_ref());
    let has_tools_arg = req.tools.as_ref().is_some_and(|v| !v.is_empty());
    let parse_tools = has_tools_arg && tc_mode != ToolChoiceMode::None;
    let schemas = if parse_tools {
        extract_tool_schemas(req.tools.as_deref())
    } else {
        ToolParamSchema::new()
    };

    let (prompt, image_sources, output_mode) = match st.model_kind {
        ServerModelKind::Mm3 => {
            let (thinking_mode, effort_opt) = resolve_mm3_thinking(
                req.chat_template_kwargs.as_ref(),
                req.reasoning_effort.as_ref(),
            );
            let (p, expect_think_close, prefill, imgs) = match format_mm3_chat(
                &req.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                thinking_mode,
                effort_opt.as_deref(),
                st.has_vision,
            ) {
                Ok(v) => v,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };
            (
                p,
                imgs,
                ChatOutputMode::Mm3 {
                    expect_think_close,
                    parse_tools,
                    prefill,
                    schemas,
                },
            )
        }
        ServerModelKind::Qwen4 => {
            let (enable_thinking, effort_label) = match resolve_qwen4_thinking(
                req.chat_template_kwargs.as_ref(),
                req.reasoning_effort.as_ref(),
            ) {
                Ok(v) => v,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };
            let (p, expect_think_close, prefill, imgs) = match format_qwen4_chat(
                &req.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                enable_thinking,
                effort_label,
                st.has_vision,
            ) {
                Ok(v) => v,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };
            (
                p,
                imgs,
                ChatOutputMode::Qwen4 {
                    expect_think_close,
                    parse_tools,
                    prefill,
                    schemas,
                },
            )
        }
        ServerModelKind::Hy4 => {
            let (enable_thinking, effort_label) = resolve_hy4_thinking(
                req.chat_template_kwargs.as_ref(),
                req.reasoning_effort.as_ref(),
            );
            let (p, expect_think_close, prefill, imgs) = match format_hy4_chat(
                &req.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                enable_thinking,
                effort_label,
                &st.hy4_tag_suffix,
                st.has_vision,
            ) {
                Ok(v) => v,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };
            (
                p,
                imgs,
                ChatOutputMode::Hy4 {
                    expect_think_close,
                    parse_tools,
                    prefill,
                    schemas,
                },
            )
        }
        ServerModelKind::Glm5 => {
            let (enable_thinking, effort_label) = resolve_glm5_thinking(
                req.chat_template_kwargs.as_ref(),
                req.reasoning_effort.as_ref(),
            );
            let (p, expect_think_close, prefill, imgs) = match format_glm5_chat(
                &req.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                enable_thinking,
                effort_label,
                st.has_vision,
            ) {
                Ok(v) => v,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };
            (
                p,
                imgs,
                ChatOutputMode::Glm5 {
                    expect_think_close,
                    parse_tools,
                    prefill,
                    schemas,
                },
            )
        }
        ServerModelKind::Inkling2 => {
            let (p, start_in_text) = format_inkling2_chat(
                &req.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                req.reasoning_effort.as_ref(),
            );
            (
                p,
                Vec::new(),
                ChatOutputMode::Inkling2 {
                    start_in_text,
                    parse_tools,
                },
            )
        }
    };

    let req_max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(4096)
        .min(st.max_ctx);
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        seed: req.seed.unwrap_or(42),
        max_tokens: req_max_tokens,
        stop_token_ids: Vec::new(),
    };
    let stop_strings = req.stop.map(StopParam::into_vec).unwrap_or_default();
    let stream_flag = req.stream.unwrap_or(false);
    let model = req.model.unwrap_or_else(|| st.model_name.clone());
    let seq = NEXT_REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    let created = unix_ts();
    let req_id = format!("chatcmpl-{created}-{seq}");
    let call_id_prefix = format!("call_{seq}");

    st.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    st.pending_jobs.fetch_add(1, Ordering::Relaxed);
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt,
            image_sources,
            params,
            stop_strings,
            output_mode,
            call_id_prefix,
            resp_tx,
        })
        .is_err()
    {
        st.pending_jobs.fetch_sub(1, Ordering::Relaxed);
        return (StatusCode::INTERNAL_SERVER_ERROR, "inference worker stopped").into_response();
    }

    if stream_flag {
        let model_c = model.clone();
        let req_id_c = req_id.clone();
        let mut sent_role = false;
        let sse_stream = UnboundedReceiverStream::new(resp_rx).flat_map(move |ev| {
            let mut events: Vec<Result<Event, Infallible>> = Vec::new();
            match ev {
                StreamEvent::Token(tok) => {
                    let mut delta = serde_json::Map::new();
                    if !sent_role {
                        delta.insert(
                            "role".to_owned(),
                            serde_json::Value::String("assistant".to_owned()),
                        );
                        sent_role = true;
                    }
                    delta.insert("content".to_owned(), serde_json::Value::String(tok));
                    let payload = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": serde_json::Value::Object(delta),
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(payload.to_string())));
                }
                StreamEvent::Reasoning(r) => {
                    let mut delta = serde_json::Map::new();
                    if !sent_role {
                        delta.insert(
                            "role".to_owned(),
                            serde_json::Value::String("assistant".to_owned()),
                        );
                        sent_role = true;
                    }
                    delta.insert(
                        "reasoning_content".to_owned(),
                        serde_json::Value::String(r),
                    );
                    let payload = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": serde_json::Value::Object(delta),
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(payload.to_string())));
                }
                StreamEvent::ToolCall {
                    index,
                    id,
                    name,
                    arguments,
                } => {
                    let mut d1 = serde_json::Map::new();
                    if !sent_role {
                        d1.insert(
                            "role".to_owned(),
                            serde_json::Value::String("assistant".to_owned()),
                        );
                        sent_role = true;
                    }
                    d1.insert(
                        "tool_calls".to_owned(),
                        serde_json::json!([{
                            "index": index,
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": ""
                            }
                        }]),
                    );
                    let p1 = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": serde_json::Value::Object(d1),
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(p1.to_string())));

                    let p2 = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "function": {
                                        "arguments": arguments
                                    }
                                }]
                            },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(p2.to_string())));
                }
                StreamEvent::ToolCallStart { index, id, name } => {
                    let mut d1 = serde_json::Map::new();
                    if !sent_role {
                        d1.insert(
                            "role".to_owned(),
                            serde_json::Value::String("assistant".to_owned()),
                        );
                        sent_role = true;
                    }
                    d1.insert(
                        "tool_calls".to_owned(),
                        serde_json::json!([{
                            "index": index,
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": ""
                            }
                        }]),
                    );
                    let p1 = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": serde_json::Value::Object(d1),
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(p1.to_string())));
                }
                StreamEvent::ToolCallArgsDelta { index, delta } => {
                    let p2 = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_c,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "function": {
                                        "arguments": delta
                                    }
                                }]
                            },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    events.push(Ok(Event::default().data(p2.to_string())));
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
                    events.push(Ok(Event::default().data(payload.to_string())));
                }
                StreamEvent::Error(e) => {
                    events.push(Ok(
                        Event::default().data(openai_error_payload(&e).to_string())
                    ));
                }
            }
            futures::stream::iter(events)
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
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
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
            StreamEvent::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }));
            }
            StreamEvent::ToolCallStart { id, name, .. } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": ""
                    }
                }));
            }
            StreamEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(serde_json::Value::String(args)) = tool_calls
                    .get_mut(index)
                    .and_then(|entry| entry.pointer_mut("/function/arguments"))
                {
                    args.push_str(&delta);
                }
            }
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
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(openai_error_payload(&e)))
                    .into_response();
            }
        }
    }

    let mut msg_obj = serde_json::json!({
        "role": "assistant",
        "content": if !tool_calls.is_empty() && content.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(content)
        }
    });
    if !reasoning.is_empty() {
        msg_obj["reasoning_content"] = serde_json::Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        msg_obj["tool_calls"] = serde_json::Value::Array(tool_calls);
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

fn openai_error_payload(message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null
        }
    })
}

pub async fn completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens.unwrap_or(512).min(st.max_ctx),
        stop_token_ids: Vec::new(),
    };
    let stop_strings = req.stop.map(StopParam::into_vec).unwrap_or_default();
    let stream_flag = req.stream.unwrap_or(false);
    let model = req.model.unwrap_or_else(|| st.model_name.clone());
    let seq = NEXT_REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    let created = unix_ts();
    let req_id = format!("cmpl-{created}-{seq}");

    st.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    st.pending_jobs.fetch_add(1, Ordering::Relaxed);
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt: req.prompt,
            image_sources: Vec::new(),
            params,
            stop_strings,
            output_mode: ChatOutputMode::Plain,
            call_id_prefix: format!("call_{seq}"),
            resp_tx,
        })
        .is_err()
    {
        st.pending_jobs.fetch_sub(1, Ordering::Relaxed);
        return (StatusCode::INTERNAL_SERVER_ERROR, "inference worker stopped").into_response();
    }

    if stream_flag {
        let model_c = model.clone();
        let req_id_c = req_id.clone();
        let sse_stream = UnboundedReceiverStream::new(resp_rx).flat_map(move |ev| {
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
                StreamEvent::ToolCall { .. }
                | StreamEvent::ToolCallStart { .. }
                | StreamEvent::ToolCallArgsDelta { .. } => {
                    return futures::stream::iter(Vec::<Result<Event, Infallible>>::new());
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
                    Event::default().data(openai_error_payload(&e).to_string())
                }
            };
            futures::stream::iter(vec![Ok::<Event, Infallible>(item)])
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
            StreamEvent::ToolCall { .. }
            | StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolCallArgsDelta { .. } => {}
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
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(openai_error_payload(&e)))
                    .into_response();
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
