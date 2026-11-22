use crate::metrics::ServerMetrics;
use crate::parser::{ChatOutputMode, ToolParamSchema};
use ampllm_engine::SamplingParams;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::StreamExt as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

static NEXT_REQ_SEQ: AtomicU64 = AtomicU64::new(1);

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
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
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

/// Format OpenAI chat messages using GLM-5.3-Flash's exact GGUF chat_template (`[gMASK]<sop>...`).
/// Returns `(rendered_prompt, expect_think_close, forced_prefill)`.
pub fn format_glm5_chat(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    enable_thinking: bool,
    effort_label: &str,
) -> (String, bool, String) {
    let tc_mode = parse_tool_choice(tool_choice);
    let mut out = String::new();
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
        let raw = extract_visible_text(msg);
        match msg.role.as_str() {
            "system" | "developer" => {
                out.push_str("<|system|>");
                out.push_str(&raw);
                idx += 1;
            }
            "user" => {
                out.push_str("<|user|>");
                out.push_str(&raw);
                idx += 1;
            }
            "assistant" => {
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
                let mut order: Vec<usize> = (0..run.len()).collect();
                if !last_assistant_call_ids.is_empty()
                    && run.iter().all(|m| {
                        m.tool_call_id
                            .as_deref()
                            .is_some_and(|id| last_assistant_call_ids.iter().any(|x| x == id))
                    })
                {
                    order.sort_by_key(|&i| {
                        let cid = run[i].tool_call_id.as_deref().unwrap_or("");
                        last_assistant_call_ids
                            .iter()
                            .position(|x| x == cid)
                            .unwrap_or(usize::MAX)
                    });
                }
                out.push_str("<|observation|>");
                for &i in &order {
                    out.push_str("<tool_response>");
                    out.push_str(&extract_visible_text(&run[i]));
                    out.push_str("</tool_response>");
                }
                idx = end;
            }
            _ => {
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
        (out, false, forced_prefill)
    } else if enable_thinking {
        out.push_str("<|assistant|><think>");
        (out, true, String::new())
    } else {
        out.push_str("<|assistant|><think></think>");
        (out, false, String::new())
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

    let (prompt, output_mode) = if st.is_glm5 {
        let (enable_thinking, effort_label) = resolve_glm5_thinking(
            req.chat_template_kwargs.as_ref(),
            req.reasoning_effort.as_ref(),
        );
        let (p, expect_think_close, prefill) = format_glm5_chat(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            enable_thinking,
            effort_label,
        );
        (
            p,
            ChatOutputMode::Glm5 {
                expect_think_close,
                parse_tools,
                prefill,
                schemas,
            },
        )
    } else {
        let (p, start_in_text) = format_inkling2_chat(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            req.reasoning_effort.as_ref(),
        );
        (
            p,
            ChatOutputMode::Inkling2 {
                start_in_text,
                parse_tools,
            },
        )
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
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt,
            params,
            stop_strings,
            output_mode,
            call_id_prefix,
            resp_tx,
        })
        .is_err()
    {
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
                        Event::default().data(serde_json::json!({"error": e}).to_string())
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
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
    if st
        .tx
        .send(EngineJob {
            prompt: req.prompt,
            params,
            stop_strings,
            output_mode: ChatOutputMode::Plain,
            call_id_prefix: format!("call_{seq}"),
            resp_tx,
        })
        .is_err()
    {
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
                StreamEvent::ToolCall { .. } => {
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
                    Event::default().data(serde_json::json!({"error": e}).to_string())
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
            StreamEvent::ToolCall { .. } => {}
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
