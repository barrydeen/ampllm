use std::collections::HashMap;

/// Schema map: `tool_name -> (param_name -> json_schema_type)` used for argument coercion.
pub type ToolParamSchema = HashMap<String, HashMap<String, String>>;

#[derive(Clone, Debug)]
pub enum ChatOutputMode {
    Plain,
    Glm5 {
        expect_think_close: bool,
        parse_tools: bool,
        prefill: String,
        schemas: ToolParamSchema,
    },
    Hy4 {
        expect_think_close: bool,
        parse_tools: bool,
        prefill: String,
        schemas: ToolParamSchema,
    },
    Qwen4 {
        expect_think_close: bool,
        parse_tools: bool,
        prefill: String,
        schemas: ToolParamSchema,
    },
    Mm3 {
        expect_think_close: bool,
        parse_tools: bool,
        prefill: String,
        schemas: ToolParamSchema,
    },
    Inkling2 {
        start_in_text: bool,
        parse_tools: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedEvent {
    Token(String),
    Reasoning(String),
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GlmState {
    Think,
    Text,
    InToolCall,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InkSection {
    SeekingHeader,
    Thinking,
    Text,
    ToolJson,
    Done,
}

enum ParserState {
    Plain,
    Glm5 {
        state: GlmState,
        parse_tools: bool,
        qwen4_xml_tools: bool,
        in_code_fence: bool,
        schemas: ToolParamSchema,
    },
    Inkling2 {
        section: InkSection,
        parse_tools: bool,
        pending_fn_name: String,
    },
    Mm3 {
        state: GlmState,
        parse_tools: bool,
        in_code_fence: bool,
        schemas: ToolParamSchema,
    },
}

/// Streaming parser that splits model reasoning blocks and structured `<tool_call>`
/// or Inkling-2 `<|content_invoke_tool_json|>` payloads into OpenAI events.
pub struct OutputParser {
    inner: ParserState,
    buf: String,
    call_id_prefix: String,
    next_call_index: usize,
}

fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Return byte count from the end of `s` that matches any proper non-empty prefix of `delims`.
fn trailing_delim_prefix_len(s: &str, delims: &[&str]) -> usize {
    let max_hold = delims
        .iter()
        .map(|d| d.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .min(s.len());
    for keep in (1..=max_hold).rev() {
        let start = s.len() - keep;
        if !s.is_char_boundary(start) {
            continue;
        }
        let tail = &s[start..];
        if delims.iter().any(|d| d.starts_with(tail)) {
            return keep;
        }
    }
    0
}

fn earliest_match<'a>(s: &str, patterns: &[&'a str]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for &pat in patterns {
        if let Some(idx) = s.find(pat) {
            match best {
                Some((bi, _)) if idx >= bi => {}
                _ => best = Some((idx, pat)),
            }
        }
    }
    best
}

fn coerce_arg_value(raw: &str, expected_type: Option<&str>) -> serde_json::Value {
    if expected_type == Some("string") {
        return serde_json::Value::String(raw.to_owned());
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::Value::String(String::new());
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(val) => val,
        Err(_) => serde_json::Value::String(raw.to_owned()),
    }
}

fn serialize_args_value(args: Option<&serde_json::Value>) -> String {
    match args {
        Some(serde_json::Value::Object(_)) => {
            serde_json::to_string(args.unwrap()).unwrap_or_else(|_| "{}".to_owned())
        }
        Some(serde_json::Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.starts_with('{') && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
            {
                trimmed.to_owned()
            } else {
                serde_json::to_string(&serde_json::json!({ "input": s }))
                    .unwrap_or_else(|_| "{}".to_owned())
            }
        }
        _ => "{}".to_owned(),
    }
}

fn parse_glm5_tool_body(body: &str, schemas: &ToolParamSchema) -> Option<(String, String)> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !body.contains("<arg_key>") && trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(name) = v.get("name").and_then(|s| s.as_str()) {
                let name = name.trim();
                if !name.is_empty() {
                    let args_str = serialize_args_value(
                        v.get("arguments").or_else(|| v.get("args")),
                    );
                    return Some((name.to_owned(), args_str));
                }
            }
        }
    }

    let (name, mut rest) = if let Some(first_k) = body.find("<arg_key>") {
        (body[..first_k].trim(), &body[first_k..])
    } else {
        (trimmed, "")
    };

    if name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let param_schema = schemas.get(name);
    while let Some(ks) = rest.find("<arg_key>") {
        let after_ks = &rest[ks + "<arg_key>".len()..];
        let Some(ke) = after_ks.find("</arg_key>") else {
            break;
        };
        let key = after_ks[..ke].trim();
        let after_ke = &after_ks[ke + "</arg_key>".len()..];
        let Some(vs) = after_ke.find("<arg_value>") else {
            break;
        };
        let after_vs = &after_ke[vs + "<arg_value>".len()..];
        let (raw_val, next_rest) = if let Some(ve) = after_vs.find("</arg_value>") {
            (&after_vs[..ve], &after_vs[ve + "</arg_value>".len()..])
        } else {
            (after_vs, "")
        };
        if !key.is_empty() {
            let expected_ty = param_schema.and_then(|m| m.get(key)).map(String::as_str);
            args.insert(key.to_owned(), coerce_arg_value(raw_val, expected_ty));
        }
        rest = next_rest;
    }

    let args_json =
        serde_json::to_string(&serde_json::Value::Object(args)).unwrap_or_else(|_| "{}".to_owned());
    Some((name.to_owned(), args_json))
}

fn parse_qwen4_tool_body(body: &str, schemas: &ToolParamSchema) -> Option<(String, String)> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !body.contains("<function=") && trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(name) = v.get("name").and_then(|s| s.as_str()) {
                let name = name.trim();
                if !name.is_empty() {
                    let args_str = serialize_args_value(
                        v.get("arguments").or_else(|| v.get("args")),
                    );
                    return Some((name.to_owned(), args_str));
                }
            }
        }
    }

    let fn_pos = body.find("<function=")?;
    let after_fn = &body[fn_pos + "<function=".len()..];
    let gt = after_fn.find('>')?;
    let name = after_fn[..gt].trim();
    if name.is_empty() {
        return None;
    }
    let after_header = &after_fn[gt + 1..];
    let mut rest = if let Some(fn_end) = after_header.find("</function>") {
        &after_header[..fn_end]
    } else {
        after_header
    };

    let mut args = serde_json::Map::new();
    let param_schema = schemas.get(name);
    while let Some(ps) = rest.find("<parameter=") {
        let after_ps = &rest[ps + "<parameter=".len()..];
        let Some(pe) = after_ps.find('>') else {
            break;
        };
        let key = after_ps[..pe].trim();
        let after_pe = &after_ps[pe + 1..];
        let (mut raw_val, next_rest) = if let Some(pend) = after_pe.find("</parameter>") {
            (&after_pe[..pend], &after_pe[pend + "</parameter>".len()..])
        } else {
            (after_pe, "")
        };
        if let Some(s) = raw_val.strip_prefix("\r\n").or_else(|| raw_val.strip_prefix('\n')) {
            raw_val = s;
        }
        if let Some(s) = raw_val.strip_suffix("\r\n").or_else(|| raw_val.strip_suffix('\n')) {
            raw_val = s;
        }
        if !key.is_empty() {
            let expected_ty = param_schema.and_then(|m| m.get(key)).map(String::as_str);
            args.insert(key.to_owned(), coerce_arg_value(raw_val, expected_ty));
        }
        rest = next_rest;
    }

    let args_json =
        serde_json::to_string(&serde_json::Value::Object(args)).unwrap_or_else(|_| "{}".to_owned());
    Some((name.to_owned(), args_json))
}

fn parse_inkling2_tool_json(body: &str, fallback_name: &str) -> Option<(String, String)> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    let v = serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
    let name = v
        .get("name")
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            let f = fallback_name.trim();
            if f.is_empty() {
                None
            } else {
                Some(f.to_owned())
            }
        })?;
    let args_json = serialize_args_value(v.get("args").or_else(|| v.get("arguments")));
    Some((name, args_json))
}

impl OutputParser {
    pub fn new(mode: ChatOutputMode, call_id_prefix: String) -> Self {
        let (inner, init_buf) = match mode {
            ChatOutputMode::Plain => (ParserState::Plain, String::new()),
            ChatOutputMode::Glm5 {
                expect_think_close,
                parse_tools,
                prefill,
                schemas,
            }
            | ChatOutputMode::Hy4 {
                expect_think_close,
                parse_tools,
                prefill,
                schemas,
            } => {
                let state = if expect_think_close {
                    GlmState::Think
                } else {
                    GlmState::Text
                };
                (
                    ParserState::Glm5 {
                        state,
                        parse_tools,
                        qwen4_xml_tools: false,
                        in_code_fence: false,
                        schemas,
                    },
                    prefill,
                )
            }
            ChatOutputMode::Qwen4 {
                expect_think_close,
                parse_tools,
                prefill,
                schemas,
            } => {
                let state = if expect_think_close {
                    GlmState::Think
                } else {
                    GlmState::Text
                };
                (
                    ParserState::Glm5 {
                        state,
                        parse_tools,
                        qwen4_xml_tools: true,
                        in_code_fence: false,
                        schemas,
                    },
                    prefill,
                )
            }
            ChatOutputMode::Mm3 {
                expect_think_close,
                parse_tools,
                prefill,
                schemas,
            } => (
                ParserState::Mm3 {
                    state: if expect_think_close {
                        GlmState::Think
                    } else {
                        GlmState::Text
                    },
                    parse_tools,
                    in_code_fence: false,
                    schemas,
                },
                prefill,
            ),
            ChatOutputMode::Inkling2 {
                start_in_text,
                parse_tools,
            } => (
                ParserState::Inkling2 {
                    section: if start_in_text {
                        InkSection::Text
                    } else {
                        InkSection::SeekingHeader
                    },
                    parse_tools,
                    pending_fn_name: String::new(),
                },
                String::new(),
            ),
        };
        Self {
            inner,
            buf: init_buf,
            call_id_prefix,
            next_call_index: 0,
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        self.next_call_index > 0
    }

    fn emit_tool_call(&mut self, name: String, arguments: String, emit: &mut impl FnMut(ParsedEvent)) {
        let index = self.next_call_index;
        self.next_call_index += 1;
        emit(ParsedEvent::ToolCall {
            index,
            id: format!("{}_{index}", self.call_id_prefix),
            name,
            arguments,
        });
    }

    pub fn push(&mut self, piece: &str, emit: &mut impl FnMut(ParsedEvent)) {
        if piece.is_empty() && self.buf.is_empty() {
            return;
        }
        self.buf.push_str(piece);
        self.process(false, emit);
    }

    pub fn finish(&mut self, emit: &mut impl FnMut(ParsedEvent)) {
        self.process(true, emit);
    }

    fn process(&mut self, eof: bool, emit: &mut impl FnMut(ParsedEvent)) {
        loop {
            match &mut self.inner {
                ParserState::Plain => {
                    if !self.buf.is_empty() {
                        emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                    }
                    return;
                }
                ParserState::Glm5 {
                    state,
                    parse_tools,
                    qwen4_xml_tools,
                    in_code_fence,
                    schemas,
                } => match *state {
                    GlmState::Think => {
                        const THINK_CLOSE: &str = "</think>";
                        if let Some(idx) = self.buf.find(THINK_CLOSE) {
                            if idx > 0 {
                                emit(ParsedEvent::Reasoning(self.buf[..idx].to_owned()));
                            }
                            self.buf.drain(..idx + THINK_CLOSE.len());
                            *state = GlmState::Text;
                            continue;
                        }
                        if eof {
                            if !self.buf.is_empty() {
                                emit(ParsedEvent::Reasoning(std::mem::take(&mut self.buf)));
                            }
                            return;
                        }
                        let keep = trailing_delim_prefix_len(&self.buf, &[THINK_CLOSE]);
                        let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                        if flush > 0 {
                            emit(ParsedEvent::Reasoning(self.buf[..flush].to_owned()));
                            self.buf.drain(..flush);
                        }
                        return;
                    }
                    GlmState::Text => {
                        if !*parse_tools {
                            if let Some(idx) = self.buf.find("<think>") {
                                if idx > 0 {
                                    emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                }
                                self.buf.drain(..idx + "<think>".len());
                                *state = GlmState::Think;
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &["<think>"]);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }

                        if *in_code_fence {
                            const FENCE: &str = "```";
                            if let Some(idx) = self.buf.find(FENCE) {
                                let end = idx + FENCE.len();
                                emit(ParsedEvent::Token(self.buf[..end].to_owned()));
                                self.buf.drain(..end);
                                *in_code_fence = false;
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &[FENCE]);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }

                        const DELIMS: [&str; 3] = ["<tool_call>", "```", "<think>"];
                        if let Some((idx, pat)) = earliest_match(&self.buf, &DELIMS) {
                            match pat {
                                "```" => {
                                    let end = idx + 3;
                                    emit(ParsedEvent::Token(self.buf[..end].to_owned()));
                                    self.buf.drain(..end);
                                    *in_code_fence = true;
                                    continue;
                                }
                                "<think>" => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.drain(..idx + "<think>".len());
                                    *state = GlmState::Think;
                                    continue;
                                }
                                _ => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.drain(..idx + "<tool_call>".len());
                                    *state = GlmState::InToolCall;
                                    continue;
                                }
                            }
                        }

                        if eof {
                            if !self.buf.is_empty() {
                                emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                            }
                            return;
                        }
                        let keep = trailing_delim_prefix_len(&self.buf, &DELIMS);
                        let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                        if flush > 0 {
                            emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                            self.buf.drain(..flush);
                        }
                        return;
                    }
                    GlmState::InToolCall => {
                        const TOOL_CLOSE: &str = "</tool_call>";
                        if let Some(idx) = self.buf.find(TOOL_CLOSE) {
                            let body = self.buf[..idx].to_owned();
                            self.buf.drain(..idx + TOOL_CLOSE.len());
                            *state = GlmState::Text;
                            let schemas_cloned = schemas.clone();
                            let parsed = if *qwen4_xml_tools {
                                parse_qwen4_tool_body(&body, &schemas_cloned)
                            } else {
                                parse_glm5_tool_body(&body, &schemas_cloned)
                            };
                            if let Some((name, arguments)) = parsed {
                                self.emit_tool_call(name, arguments, emit);
                            } else if !body.trim().is_empty() {
                                emit(ParsedEvent::Token(format!("<tool_call>{body}</tool_call>")));
                            }
                            continue;
                        }
                        if eof {
                            let body = std::mem::take(&mut self.buf);
                            *state = GlmState::Text;
                            let schemas_cloned = schemas.clone();
                            let has_xml_end = if *qwen4_xml_tools {
                                body.contains("</parameter>") || body.contains("</function>")
                            } else {
                                body.contains("</arg_value>")
                            };
                            if (has_xml_end || !body.contains('<')) && !body.trim().is_empty() {
                                let parsed = if *qwen4_xml_tools {
                                    parse_qwen4_tool_body(&body, &schemas_cloned)
                                } else {
                                    parse_glm5_tool_body(&body, &schemas_cloned)
                                };
                                if let Some((name, arguments)) = parsed {
                                    self.emit_tool_call(name, arguments, emit);
                                    return;
                                }
                            }
                            if !body.is_empty() {
                                emit(ParsedEvent::Token(format!("<tool_call>{body}")));
                            }
                            return;
                        }
                        return;
                    }
                },
                ParserState::Inkling2 {
                    section,
                    parse_tools,
                    pending_fn_name,
                } => {
                    const INK_STOP: &str = "<|content_model_end_sampling|>";
                    const INK_END: &str = "<|end_message|>";
                    const INK_MODEL: &str = "<|message_model|>";
                    const INK_THINK: &str = "<|content_thinking|>";
                    const INK_TEXT: &str = "<|content_text|>";
                    const INK_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
                    const INK_DELIMS: [&str; 6] = [
                        INK_STOP,
                        INK_END,
                        INK_MODEL,
                        INK_THINK,
                        INK_TEXT,
                        INK_TOOL_JSON,
                    ];

                    match *section {
                        InkSection::Done => {
                            self.buf.clear();
                            return;
                        }
                        InkSection::SeekingHeader => {
                            if let Some((idx, pat)) = earliest_match(&self.buf, &INK_DELIMS) {
                                match pat {
                                    INK_THINK => {
                                        pending_fn_name.clear();
                                        self.buf.drain(..idx + pat.len());
                                        *section = InkSection::Thinking;
                                    }
                                    INK_TEXT => {
                                        pending_fn_name.clear();
                                        self.buf.drain(..idx + pat.len());
                                        *section = InkSection::Text;
                                    }
                                    INK_TOOL_JSON => {
                                        *pending_fn_name = self.buf[..idx].trim().to_owned();
                                        self.buf.drain(..idx + pat.len());
                                        *section = InkSection::ToolJson;
                                    }
                                    INK_MODEL | INK_END => {
                                        self.buf.drain(..idx + pat.len());
                                    }
                                    INK_STOP => {
                                        self.buf.clear();
                                        *section = InkSection::Done;
                                        return;
                                    }
                                    _ => unreachable!(),
                                }
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            return;
                        }
                        InkSection::Thinking => {
                            if let Some((idx, pat)) = earliest_match(&self.buf, &INK_DELIMS) {
                                if idx > 0 {
                                    emit(ParsedEvent::Reasoning(self.buf[..idx].to_owned()));
                                }
                                self.buf.drain(..idx + pat.len());
                                *section = match pat {
                                    INK_THINK => InkSection::Thinking,
                                    INK_TEXT => InkSection::Text,
                                    INK_TOOL_JSON => InkSection::ToolJson,
                                    INK_END | INK_MODEL => InkSection::SeekingHeader,
                                    INK_STOP => InkSection::Done,
                                    _ => unreachable!(),
                                };
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Reasoning(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &INK_DELIMS);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Reasoning(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }
                        InkSection::Text => {
                            if let Some((idx, pat)) = earliest_match(&self.buf, &INK_DELIMS) {
                                if idx > 0 {
                                    emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                }
                                self.buf.drain(..idx + pat.len());
                                *section = match pat {
                                    INK_THINK => InkSection::Thinking,
                                    INK_TEXT => InkSection::Text,
                                    INK_TOOL_JSON => InkSection::ToolJson,
                                    INK_END | INK_MODEL => InkSection::SeekingHeader,
                                    INK_STOP => InkSection::Done,
                                    _ => unreachable!(),
                                };
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &INK_DELIMS);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }
                        InkSection::ToolJson => {
                            if let Some((idx, pat)) = earliest_match(&self.buf, &INK_DELIMS) {
                                let body = self.buf[..idx].to_owned();
                                self.buf.drain(..idx + pat.len());
                                let fallback = std::mem::take(pending_fn_name);
                                let allow_tools = *parse_tools;
                                *section = match pat {
                                    INK_THINK => InkSection::Thinking,
                                    INK_TEXT => InkSection::Text,
                                    INK_TOOL_JSON => InkSection::ToolJson,
                                    INK_END | INK_MODEL => InkSection::SeekingHeader,
                                    INK_STOP => InkSection::Done,
                                    _ => unreachable!(),
                                };
                                if allow_tools {
                                    if let Some((name, arguments)) =
                                        parse_inkling2_tool_json(&body, &fallback)
                                    {
                                        self.emit_tool_call(name, arguments, emit);
                                    }
                                }
                                continue;
                            }
                            if eof {
                                let body = std::mem::take(&mut self.buf);
                                let fallback = std::mem::take(pending_fn_name);
                                let allow_tools = *parse_tools;
                                *section = InkSection::Done;
                                if allow_tools {
                                    if let Some((name, arguments)) =
                                        parse_inkling2_tool_json(&body, &fallback)
                                    {
                                        self.emit_tool_call(name, arguments, emit);
                                    }
                                }
                                return;
                            }
                            return;
                        }
                    }
                }
                ParserState::Mm3 {
                    state,
                    parse_tools,
                    in_code_fence,
                    schemas,
                } => match *state {
                    GlmState::Think => {
                        const THINK_CLOSE: &str = "</mm:think>";
                        if let Some(idx) = self.buf.find(THINK_CLOSE) {
                            if idx > 0 {
                                emit(ParsedEvent::Reasoning(self.buf[..idx].to_owned()));
                            }
                            self.buf.drain(..idx + THINK_CLOSE.len());
                            *state = GlmState::Text;
                            continue;
                        }
                        if eof {
                            if !self.buf.is_empty() {
                                emit(ParsedEvent::Reasoning(std::mem::take(&mut self.buf)));
                            }
                            return;
                        }
                        let keep = trailing_delim_prefix_len(&self.buf, &[THINK_CLOSE]);
                        let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                        if flush > 0 {
                            emit(ParsedEvent::Reasoning(self.buf[..flush].to_owned()));
                            self.buf.drain(..flush);
                        }
                        return;
                    }
                    GlmState::Text => {
                        const MM3_THINK: &str = "<mm:think>";
                        const MM3_THINK_CLOSE: &str = "</mm:think>";
                        const MM3_TOOL_START: &str = "]<]minimax[>[<tool_call>";
                        const MM3_EOS: &str = "[e~[";
                        if !*parse_tools {
                            const DELIMS_NOTOOL: [&str; 3] =
                                [MM3_THINK, MM3_THINK_CLOSE, MM3_EOS];
                            if let Some((idx, pat)) = earliest_match(&self.buf, &DELIMS_NOTOOL) {
                                if idx > 0 {
                                    emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                }
                                self.buf.drain(..idx + pat.len());
                                if pat == MM3_THINK {
                                    *state = GlmState::Think;
                                    continue;
                                }
                                if pat == MM3_THINK_CLOSE {
                                    continue;
                                }
                                self.buf.clear();
                                return;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &DELIMS_NOTOOL);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }

                        if *in_code_fence {
                            const FENCE: &str = "```";
                            if let Some(idx) = self.buf.find(FENCE) {
                                let end = idx + FENCE.len();
                                emit(ParsedEvent::Token(self.buf[..end].to_owned()));
                                self.buf.drain(..end);
                                *in_code_fence = false;
                                continue;
                            }
                            if eof {
                                if !self.buf.is_empty() {
                                    emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                                }
                                return;
                            }
                            let keep = trailing_delim_prefix_len(&self.buf, &[FENCE]);
                            let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                            if flush > 0 {
                                emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                                self.buf.drain(..flush);
                            }
                            return;
                        }

                        const DELIMS: [&str; 5] =
                            [MM3_TOOL_START, "```", MM3_THINK, MM3_THINK_CLOSE, MM3_EOS];
                        if let Some((idx, pat)) = earliest_match(&self.buf, &DELIMS) {
                            match pat {
                                "```" => {
                                    let end = idx + 3;
                                    emit(ParsedEvent::Token(self.buf[..end].to_owned()));
                                    self.buf.drain(..end);
                                    *in_code_fence = true;
                                    continue;
                                }
                                MM3_THINK => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.drain(..idx + pat.len());
                                    *state = GlmState::Think;
                                    continue;
                                }
                                MM3_THINK_CLOSE => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.drain(..idx + pat.len());
                                    continue;
                                }
                                MM3_EOS => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.clear();
                                    return;
                                }
                                _ => {
                                    if idx > 0 {
                                        emit(ParsedEvent::Token(self.buf[..idx].to_owned()));
                                    }
                                    self.buf.drain(..idx + pat.len());
                                    *state = GlmState::InToolCall;
                                    continue;
                                }
                            }
                        }
                        if eof {
                            if !self.buf.is_empty() {
                                emit(ParsedEvent::Token(std::mem::take(&mut self.buf)));
                            }
                            return;
                        }
                        let keep = trailing_delim_prefix_len(&self.buf, &DELIMS);
                        let flush = floor_char_boundary(&self.buf, self.buf.len() - keep);
                        if flush > 0 {
                            emit(ParsedEvent::Token(self.buf[..flush].to_owned()));
                            self.buf.drain(..flush);
                        }
                        return;
                    }
                    GlmState::InToolCall => {
                        const TOOL_END: &str = "]<]minimax[>[</tool_call>";
                        let mut calls: Vec<(String, String)> = Vec::new();
                        if let Some(end_idx) = self.buf.find(TOOL_END) {
                            let body = self.buf[..end_idx].to_owned();
                            self.buf.drain(..end_idx + TOOL_END.len());
                            if *parse_tools {
                                calls = parse_mm3_tool_block(&body, schemas);
                            }
                            *state = GlmState::Text;
                            for (name, arguments) in calls {
                                self.emit_tool_call(name, arguments, emit);
                            }
                            continue;
                        }
                        if eof {
                            let body = std::mem::take(&mut self.buf);
                            if *parse_tools {
                                calls = parse_mm3_tool_block(&body, schemas);
                            }
                            *state = GlmState::Text;
                            for (name, arguments) in calls {
                                self.emit_tool_call(name, arguments, emit);
                            }
                            return;
                        }
                        return;
                    }
                },
            }
        }
    }
}

fn parse_mm3_xml_value(raw: &str, expected_type: Option<&str>) -> serde_json::Value {
    let mut children: Vec<(String, String)> = Vec::new();
    let s = raw.trim();
    let mut cursor = s;
    let mut parse_ok = true;
    while !cursor.is_empty() {
        let trimmed = cursor.trim_start();
        if trimmed.is_empty() {
            break;
        }
        if !trimmed.starts_with('<') || trimmed.starts_with("</") {
            parse_ok = false;
            break;
        }
        let Some(gt) = trimmed.find('>') else {
            parse_ok = false;
            break;
        };
        let tag_full = trimmed[1..gt].trim();
        let tag = tag_full.split_whitespace().next().unwrap_or("");
        if tag.is_empty() {
            parse_ok = false;
            break;
        }
        let close_pat = format!("</{tag}>");
        let after_open = &trimmed[gt + 1..];
        let Some(close_idx) = after_open.find(&close_pat) else {
            parse_ok = false;
            break;
        };
        children.push((tag.to_owned(), after_open[..close_idx].to_owned()));
        cursor = &after_open[close_idx + close_pat.len()..];
    }
    if parse_ok && !children.is_empty() {
        if children.iter().all(|(t, _)| t == "item") {
            return serde_json::Value::Array(
                children
                    .into_iter()
                    .map(|(_, body)| parse_mm3_xml_value(&body, None))
                    .collect(),
            );
        }
        let mut map = serde_json::Map::new();
        for (k, body) in children {
            map.insert(k, parse_mm3_xml_value(&body, None));
        }
        return serde_json::Value::Object(map);
    }
    let _ = s;
    coerce_arg_value(raw, expected_type)
}

fn parse_mm3_tool_block(body: &str, schemas: &ToolParamSchema) -> Vec<(String, String)> {
    let cleaned = body.replace("]<]minimax[>[<", "<");
    let mut out = Vec::new();
    let mut rest = cleaned.as_str();
    while let Some(inv_s) = rest.find("<invoke") {
        let after_inv = &rest[inv_s + "<invoke".len()..];
        let Some(gt) = after_inv.find('>') else {
            break;
        };
        let attr_str = &after_inv[..gt];
        let name = if let Some(ns) = attr_str.find("name=") {
            let after_eq = attr_str[ns + 5..].trim_start();
            if let Some(q) = after_eq.chars().next().filter(|&c| c == '"' || c == '\'') {
                let inner = &after_eq[1..];
                inner
                    .find(q)
                    .map(|end| inner[..end].trim().to_owned())
                    .unwrap_or_default()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        let after_open = &after_inv[gt + 1..];
        let (inv_body, next_rest) = if let Some(close_i) = after_open.find("</invoke>") {
            (&after_open[..close_i], &after_open[close_i + "</invoke>".len()..])
        } else {
            (after_open, "")
        };
        if !name.is_empty() {
            let param_schema = schemas.get(&name);
            let mut args = serde_json::Map::new();
            let mut cur = inv_body;
            while let Some(lt) = cur.find('<') {
                let tail = &cur[lt + 1..];
                if tail.starts_with('/') {
                    if let Some(g) = tail.find('>') {
                        cur = &tail[g + 1..];
                        continue;
                    }
                    break;
                }
                let Some(tag_end) = tail.find('>') else {
                    break;
                };
                let key = tail[..tag_end].split_whitespace().next().unwrap_or("").trim();
                let val_start = &tail[tag_end + 1..];
                let close_tag = format!("</{key}>");
                if let Some(ve) = val_start.find(&close_tag) {
                    if !key.is_empty() {
                        let expected_ty = param_schema.and_then(|m| m.get(key)).map(String::as_str);
                        args.insert(
                            key.to_owned(),
                            parse_mm3_xml_value(&val_start[..ve], expected_ty),
                        );
                    }
                    cur = &val_start[ve + close_tag.len()..];
                } else {
                    break;
                }
            }
            let args_json =
                serde_json::to_string(&serde_json::Value::Object(args)).unwrap_or_else(|_| "{}".to_owned());
            out.push((name, args_json));
        }
        rest = next_rest;
    }
    out
}
