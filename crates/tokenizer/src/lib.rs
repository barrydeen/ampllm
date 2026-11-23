use ampllm_gguf::GgufFile;
use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::cell::RefCell;
use unicode_normalization::UnicodeNormalization;

/// Exact GPT-4o / Inkling-2 pre-tokenization regex from llama.cpp (LLAMA_VOCAB_PRE_TYPE_GPT4O).
const INKLING2_PRE_REGEX: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Exact ChatGLM4 / GLM-4 / GLM-5 pre-tokenization regex (`LLAMA_VOCAB_PRE_TYPE_CHATGLM4`).
const GLM4_PRE_REGEX: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Sequential pre-tokenization stages for `hy_v4` (`HUNYUAN_DENSE` / DeepSeek3 patterns in llama.cpp).
const HY4_PRE_STAGE1: &str = r"\p{N}{1,3}";
const HY4_PRE_STAGE2: &str = r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
const HY4_PRE_STAGE3: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// Exact Qwen3.5 / Qwen3.8 (`qwen35`) pre-tokenizer regex from llama.cpp (`LLAMA_VOCAB_PRE_TYPE_QWEN35`).
const QWEN35_PRE_REGEX: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Exact MiniMax-M2 / MiniMax-M3.1 (`minimax-m2` / `minimax`) pre-tokenizer regex (applied after NFC).
const MINIMAX_PRE_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

enum PreTokenizer {
    Single(Regex),
    MultiStage([Regex; 3]),
}

pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    always_special: Vec<(String, u32)>,
    all_special: Vec<(String, u32)>,
    pre_tokenizer: PreTokenizer,
    hy4_aliases: HashMap<u32, String>,
    hy4_tag_suffix: Option<String>,
    ignore_merges: bool,
    use_nfc: bool,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
    piece_cache: RefCell<HashMap<String, Vec<u32>>>,
    pub pre_kind: String,
    pub eos_token_id: u32,
    pub bos_token_id: Option<u32>,
    pub eot_token_id: Option<u32>,
    pub eom_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

pub fn build_gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut is_direct = [false; 256];
    for b in 33..=126 {
        is_direct[b] = true;
    }
    for b in 161..=172 {
        is_direct[b] = true;
    }
    for b in 174..=255 {
        is_direct[b] = true;
    }
    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::with_capacity(256);
    let mut next_cp = 256u32;
    for b in 0..256 {
        let ch = if is_direct[b] {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(next_cp).unwrap();
            next_cp += 1;
            c
        };
        b2u[b] = ch;
        u2b.insert(ch, b as u8);
    }
    (b2u, u2b)
}

impl BpeTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?;
        let types_arr = gguf
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array());
        let merges_arr = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.merges"))?;

        let pre_kind = gguf
            .get_str("tokenizer.ggml.pre")
            .unwrap_or("gpt4o")
            .to_owned();
        let is_glm = pre_kind == "glm4" || pre_kind == "glm5" || pre_kind == "chatglm4";
        let is_hy4 = pre_kind == "hy_v4" || pre_kind == "deepseek3";
        let is_qwen35 = pre_kind == "qwen35" || pre_kind == "qwen2";
        let is_minimax = pre_kind == "minimax-m2" || pre_kind.starts_with("minimax");
        let use_nfc = is_minimax;
        let ignore_merges = gguf
            .get_bool("tokenizer.ggml.ignore_merges")
            .unwrap_or(is_glm);

        let mut vocab = Vec::with_capacity(tokens_arr.len());
        let mut token_types = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id = HashMap::with_capacity(tokens_arr.len());
        let mut always_special = Vec::new();
        let mut all_special = Vec::new();

        for (idx, v) in tokens_arr.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or_else(|| anyhow!("non-string token at {idx}"))?
                .to_owned();
            let ttype = types_arr
                .and_then(|a| a.get(idx))
                .and_then(|t| t.as_i64())
                .unwrap_or(1) as i32;
            let id = idx as u32;
            if ttype == 4 {
                always_special.push((s.clone(), id));
            }
            if ttype == 3 || ttype == 4 {
                all_special.push((s.clone(), id));
            }
            token_to_id.insert(s.clone(), id);
            vocab.push(s);
            token_types.push(ttype);
        }
        // Match longest special tokens first.
        always_special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        all_special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut merges = HashMap::with_capacity(merges_arr.len());
        for (rank, v) in merges_arr.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or_else(|| anyhow!("non-string merge at {rank}"))?;
            if let Some((a, b)) = s.split_once(' ') {
                merges.insert((a.to_owned(), b.to_owned()), rank);
            }
        }

        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(200006);
        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eot_token_id = gguf.get_u32("tokenizer.ggml.eot_token_id").ok();
        let eom_token_id = gguf.get_u32("tokenizer.ggml.eom_token_id").ok();
        let pad_token_id = gguf.get_u32("tokenizer.ggml.padding_token_id").ok();
        let (byte_to_unicode, unicode_to_byte) = build_gpt2_byte_tables();
        let pre_tokenizer = if is_hy4 {
            PreTokenizer::MultiStage([
                Regex::new(HY4_PRE_STAGE1)?,
                Regex::new(HY4_PRE_STAGE2)?,
                Regex::new(HY4_PRE_STAGE3)?,
            ])
        } else {
            let regex_str = if is_glm {
                GLM4_PRE_REGEX
            } else if is_qwen35 {
                QWEN35_PRE_REGEX
            } else if is_minimax {
                MINIMAX_PRE_REGEX
            } else {
                INKLING2_PRE_REGEX
            };
            PreTokenizer::Single(Regex::new(regex_str)?)
        };

        let mut hy4_aliases = HashMap::new();
        let mut hy4_tag_suffix: Option<String> = None;
        for (idx, tok) in vocab.iter().enumerate() {
            let ttype = token_types.get(idx).copied().unwrap_or(1);
            if ttype == 3 || ttype == 4 {
                if let Some((detected_suffix, alias)) = parse_hy4_alias(tok) {
                    if hy4_tag_suffix.is_none() {
                        if let Some(sfx) = detected_suffix {
                            hy4_tag_suffix = Some(sfx.to_owned());
                        }
                    }
                    hy4_aliases.insert(idx as u32, alias);
                }
            }
        }

        Ok(Self {
            vocab,
            token_types,
            token_to_id,
            merges,
            always_special,
            all_special,
            pre_tokenizer,
            hy4_aliases,
            hy4_tag_suffix,
            ignore_merges,
            use_nfc,
            byte_to_unicode,
            unicode_to_byte,
            piece_cache: RefCell::new(HashMap::with_capacity(16384)),
            pre_kind,
            eos_token_id,
            bos_token_id,
            eot_token_id,
            eom_token_id,
            pad_token_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    pub fn token_types(&self) -> &[i32] {
        &self.token_types
    }

    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let specials = if parse_special {
            &self.all_special
        } else {
            &self.always_special
        };
        let mut ids = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            if let Some((tok_str, id)) = specials
                .iter()
                .find(|(s, _)| !s.is_empty() && text[pos..].starts_with(s.as_str()))
            {
                ids.push(*id);
                pos += tok_str.len();
                continue;
            }
            // Find next special token boundary.
            let next_special = specials
                .iter()
                .filter(|(s, _)| !s.is_empty())
                .filter_map(|(s, _)| text[pos..].find(s.as_str()).map(|off| pos + off))
                .min()
                .unwrap_or(text.len());

            let chunk = &text[pos..next_special];
            if self.use_nfc {
                let norm: String = chunk.nfc().collect();
                self.encode_plain(&norm, &mut ids)?;
            } else {
                self.encode_plain(chunk, &mut ids)?;
            }
            pos = next_special;
        }
        Ok(ids)
    }

    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(cached) = self.piece_cache.borrow().get(piece) {
            out.extend_from_slice(cached);
            return;
        }
        let encoded: String = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_unicode[b as usize])
            .collect();
        let start = out.len();
        self.bpe_word(&encoded, out);
        let mut cache = self.piece_cache.borrow_mut();
        if cache.len() < 200_000 {
            cache.insert(piece.to_owned(), out[start..].to_vec());
        }
    }

    fn encode_plain(&self, chunk: &str, out: &mut Vec<u32>) -> Result<()> {
        match &self.pre_tokenizer {
            PreTokenizer::Single(regex) => {
                for mat in regex.find_iter(chunk) {
                    let m = mat?;
                    self.encode_piece(m.as_str(), out);
                }
            }
            PreTokenizer::MultiStage(stages) => {
                let mut a = vec![chunk];
                let mut b = Vec::new();
                for stage in stages {
                    split_isolated(&a, stage, &mut b)?;
                    std::mem::swap(&mut a, &mut b);
                }
                for piece in a {
                    self.encode_piece(piece, out);
                }
            }
        }
        Ok(())
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        // When `ignore_merges` is set (GLM4) or for single-piece whole tokens, direct LUT hit skips BPE merges.
        let _ = self.ignore_merges;
        if let Some(&id) = self.token_to_id.get(word) {
            out.push(id);
            return;
        }
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts[idx] = merged;
            parts.remove(idx + 1);
        }
        for p in parts {
            if let Some(&id) = self.token_to_id.get(&p) {
                out.push(id);
            } else {
                // Fallback byte by byte
                for ch in p.chars() {
                    let s = ch.to_string();
                    if let Some(&id) = self.token_to_id.get(&s) {
                        out.push(id);
                    }
                }
            }
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.decode_with(ids, skip_special, !skip_special)
    }

    /// Decode token IDs skipping only `token_type == 3` (`CONTROL`) while preserving
    /// `token_type == 4` (`USER_DEFINED` tags such as `<think>`, `</think>`, `<tool_call>`,
    /// `<arg_key>`, and `<arg_value>`).
    pub fn decode_skip_control(&self, ids: &[u32]) -> String {
        self.decode_with(ids, true, true)
    }

    fn decode_with(&self, ids: &[u32], skip_control: bool, keep_user_defined: bool) -> String {
        self.decode_with_aliases(ids, skip_control, keep_user_defined, false)
    }

    fn decode_with_aliases(
        &self,
        ids: &[u32],
        skip_control: bool,
        keep_user_defined: bool,
        apply_hy4_aliases: bool,
    ) -> String {
        let mut raw_bytes = Vec::new();
        let mut out = String::new();
        let flush_bytes = |bytes: &mut Vec<u8>, s: &mut String| {
            if !bytes.is_empty() {
                s.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        for &id in ids {
            let idx = id as usize;
            if idx >= self.vocab.len() {
                continue;
            }
            let ttype = self.token_types[idx];
            if ttype == 3 || ttype == 4 {
                flush_bytes(&mut raw_bytes, &mut out);
                let keep = if ttype == 3 {
                    !skip_control || (apply_hy4_aliases && self.hy4_aliases.contains_key(&id))
                } else {
                    keep_user_defined
                };
                if keep {
                    if apply_hy4_aliases {
                        if let Some(alias) = self.hy4_aliases.get(&id) {
                            out.push_str(alias);
                        } else {
                            out.push_str(&self.vocab[idx]);
                        }
                    } else {
                        out.push_str(&self.vocab[idx]);
                    }
                }
                continue;
            }
            for ch in self.vocab[idx].chars() {
                if let Some(&b) = self.unicode_to_byte.get(&ch) {
                    raw_bytes.push(b);
                } else {
                    flush_bytes(&mut raw_bytes, &mut out);
                    out.push(ch);
                }
            }
        }
        flush_bytes(&mut raw_bytes, &mut out);
        out
    }

    pub fn hy4_tag_suffix(&self) -> Option<&str> {
        self.hy4_tag_suffix.as_deref()
    }

    pub fn decode_hy4_aliased(&self, ids: &[u32]) -> String {
        self.decode_with_aliases(ids, true, true, true)
    }

    pub fn decode_piece(&self, id: u32) -> String {
        self.decode(&[id], false)
    }

    pub fn token_id(&self, token_str: &str) -> Option<u32> {
        self.token_to_id.get(token_str).copied()
    }
}

fn split_isolated<'a>(pieces: &[&'a str], regex: &Regex, out: &mut Vec<&'a str>) -> Result<()> {
    out.clear();
    for &s in pieces {
        if s.is_empty() {
            continue;
        }
        let mut last = 0usize;
        for mat in regex.find_iter(s) {
            let m = mat?;
            if m.start() > last {
                out.push(&s[last..m.start()]);
            }
            if m.end() > m.start() {
                out.push(&s[m.start()..m.end()]);
            }
            last = m.end();
        }
        if last < s.len() {
            out.push(&s[last..]);
        }
    }
    Ok(())
}

fn parse_hy4_alias(tok: &str) -> Option<(Option<&str>, String)> {
    let inner = tok.strip_prefix('<')?.strip_suffix('>')?;
    let (slash, rest) = match inner.strip_prefix('/') {
        Some(r) => ("/", r),
        None => ("", inner),
    };
    let (tag, suffix) = rest.split_once(':')?;
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    match tag {
        "think" | "tool_call" | "arg_key" | "arg_value" => {
            let detected_suffix = if slash.is_empty() && tag == "think" {
                Some(suffix)
            } else {
                None
            };
            Some((detected_suffix, format!("<{slash}{tag}>")))
        }
        "tool_calls" => Some((None, String::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hy4_test_tokenizer(extra_tokens: &[(&str, i32)]) -> BpeTokenizer {
        let mut vocab = Vec::new();
        let mut token_types = Vec::new();
        let mut token_to_id = HashMap::new();
        let (byte_to_unicode, unicode_to_byte) = build_gpt2_byte_tables();

        let mut push_tok = |s: &str, ttype: i32| -> u32 {
            let id = vocab.len() as u32;
            vocab.push(s.to_owned());
            token_types.push(ttype);
            token_to_id.insert(s.to_owned(), id);
            id
        };

        // Single-byte GPT-2 tokens for basic ASCII pieces
        for b in 0..=255u8 {
            let s: String = [byte_to_unicode[b as usize]].into_iter().collect();
            push_tok(&s, 1);
        }
        // Multibyte CJK UTF-8 chars as raw chars (or byte tokens already present)
        for &(tok, ttype) in extra_tokens {
            push_tok(tok, ttype);
        }

        let mut hy4_aliases = HashMap::new();
        let mut hy4_tag_suffix = None;
        for (idx, tok) in vocab.iter().enumerate() {
            let ttype = token_types[idx];
            if ttype == 3 || ttype == 4 {
                if let Some((sfx, alias)) = parse_hy4_alias(tok) {
                    if hy4_tag_suffix.is_none() {
                        if let Some(s) = sfx {
                            hy4_tag_suffix = Some(s.to_owned());
                        }
                    }
                    hy4_aliases.insert(idx as u32, alias);
                }
            }
        }

        BpeTokenizer {
            vocab,
            token_types,
            token_to_id,
            merges: HashMap::new(),
            always_special: Vec::new(),
            all_special: Vec::new(),
            pre_tokenizer: PreTokenizer::MultiStage([
                Regex::new(HY4_PRE_STAGE1).unwrap(),
                Regex::new(HY4_PRE_STAGE2).unwrap(),
                Regex::new(HY4_PRE_STAGE3).unwrap(),
            ]),
            hy4_aliases,
            hy4_tag_suffix,
            ignore_merges: false,
            use_nfc: false,
            byte_to_unicode,
            unicode_to_byte,
            piece_cache: RefCell::new(HashMap::new()),
            pre_kind: "hy_v4".to_owned(),
            eos_token_id: 0,
            bos_token_id: None,
            eot_token_id: None,
            eom_token_id: None,
            pad_token_id: None,
        }
    }

    fn run_hy4_stages(text: &str) -> Vec<String> {
        let r1 = Regex::new(HY4_PRE_STAGE1).unwrap();
        let r2 = Regex::new(HY4_PRE_STAGE2).unwrap();
        let r3 = Regex::new(HY4_PRE_STAGE3).unwrap();
        let mut a = vec![text];
        let mut b = Vec::new();
        split_isolated(&a, &r1, &mut b).unwrap();
        std::mem::swap(&mut a, &mut b);
        split_isolated(&a, &r2, &mut b).unwrap();
        std::mem::swap(&mut a, &mut b);
        split_isolated(&a, &r3, &mut b).unwrap();
        b.into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn hy4_pretokenizer_stages() {
        assert_eq!(
            run_hy4_stages("abc12345def"),
            vec!["abc", "123", "45", "def"]
        );
        assert_eq!(
            run_hy4_stages("hello 你好世界!foo 12"),
            vec!["hello", " ", "你好世界", "!foo", " ", "12"]
        );
        assert_eq!(
            run_hy4_stages("x.bar+99900z"),
            vec!["x", ".bar", "+", "999", "00", "z"]
        );
    }

    #[test]
    fn hy4_aliases_and_suffix() {
        let tok = make_hy4_test_tokenizer(&[
            ("<think:opensource>", 3),
            ("</think:opensource>", 3),
            ("<tool_calls:opensource>", 3),
            ("</tool_calls:opensource>", 3),
            ("<tool_call:opensource>", 3),
            ("</tool_call:opensource>", 3),
            ("<arg_key:opensource>", 3),
            ("</arg_key:opensource>", 3),
            ("<arg_value:opensource>", 3),
            ("</arg_value:opensource>", 3),
            ("<｜hy_start:opensource｜>", 3),
        ]);
        assert_eq!(tok.hy4_tag_suffix(), Some("opensource"));
        let ids = [
            tok.token_id("<think:opensource>").unwrap(),
            b'o' as u32,
            b'k' as u32,
            tok.token_id("</think:opensource>").unwrap(),
            tok.token_id("<tool_calls:opensource>").unwrap(),
            tok.token_id("<tool_call:opensource>").unwrap(),
            tok.token_id("<arg_key:opensource>").unwrap(),
            b'x' as u32,
            tok.token_id("</arg_key:opensource>").unwrap(),
            tok.token_id("<arg_value:opensource>").unwrap(),
            b'1' as u32,
            tok.token_id("</arg_value:opensource>").unwrap(),
            tok.token_id("</tool_call:opensource>").unwrap(),
            tok.token_id("</tool_calls:opensource>").unwrap(),
            tok.token_id("<｜hy_start:opensource｜>").unwrap(),
        ];
        assert_eq!(
            tok.decode_hy4_aliased(&ids),
            "<think>ok</think><tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        );
        assert_eq!(
            tok.decode(&ids[..1], false),
            "<think:opensource>"
        );
    }

    #[test]
    fn qwen35_pretokenizer_splits() {
        let re = Regex::new(QWEN35_PRE_REGEX).unwrap();
        let split = |s: &str| -> Vec<String> {
            re.find_iter(s)
                .map(|m| m.unwrap().as_str().to_owned())
                .collect()
        };
        assert_eq!(split("abc12345"), vec!["abc", "1", "2", "3", "4", "5"]);
        assert_eq!(
            split("It's fine!\n"),
            vec!["It", "'s", " fine", "!\n"]
        );
        // Combining acute accent `\u{0301}` (\p{M}) must stay attached to word and not merge into `_j` punctuation
        assert_eq!(
            split("cafe\u{0301}_j"),
            vec!["cafe\u{0301}", "_j"]
        );
    }

    #[test]
    fn minimax_pretokenizer_and_nfc_splits() {
        let re = Regex::new(MINIMAX_PRE_REGEX).unwrap();
        let norm: String = "cafe\u{0301} ABC1234 -> \n".nfc().collect();
        let parts: Vec<String> = re
            .find_iter(&norm)
            .map(|m| m.unwrap().as_str().to_owned())
            .collect();
        assert_eq!(norm, "caf\u{00e9} ABC1234 -> \n");
        assert_eq!(parts, vec!["caf\u{00e9}", " ABC", "123", "4", " ->", " \n"]);
    }

    #[test]
    #[ignore]
    fn test_minimax_tokenizer_matches_maya_ref() {
        let Ok(gguf_str) = std::env::var("AMPLLM_MM3_MODEL") else {
            return;
        };
        let gguf_path = std::path::Path::new(&gguf_str);
        let ref_dir = std::path::Path::new("/tmp/mm3_ref");
        if !gguf_path.exists() || !ref_dir.exists() {
            return;
        }
        let gguf = ampllm_gguf::GgufFile::open(gguf_path).unwrap();
        let tok = BpeTokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.pre_kind, "minimax-m2");

        let render_mm3_prompt = |user_q: &str| -> String {
            format!(
                "]~!b[]~b]system\n\
Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.\n\n\
<thinking_instructions>\n\
You have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n\
Current thinking mode: disabled. Do not output any thinking process.\n\
</thinking_instructions>\n\n\
<model_id>8bb8149fe</model_id>[e~[\n\
]~b]developer\nYou are a helpful assistant.[e~[\n\
]~b]user\n{user_q}[e~[\n\
]~b]ai\n</mm:think>"
            )
        };
        let load_ref_ids = |name: &str| -> Vec<u32> {
            std::fs::read_to_string(ref_dir.join(name))
                .unwrap()
                .trim()
                .split([',', '\n', ' '])
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<u32>().unwrap())
                .collect()
        };

        let cases = [
            ("warmup_ids.txt", "Say hello in five languages."),
            (
                "topic0_ids.txt",
                "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.",
            ),
            (
                "topic1_ids.txt",
                "Explain how photosynthesis works, step by step, for a high-school student.",
            ),
            (
                "topic2_ids.txt",
                "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
            ),
        ];
        for (fname, q) in cases {
            let prompt = render_mm3_prompt(q);
            let got = tok.encode(&prompt, true).unwrap();
            let expected = load_ref_ids(fname);
            assert_eq!(got, expected, "token ID mismatch for {fname}");
        }
    }
}
