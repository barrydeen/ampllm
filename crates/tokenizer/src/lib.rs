use ampllm_gguf::GgufFile;
use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::cell::RefCell;

/// Exact GPT-4o / Inkling-2 pre-tokenization regex from llama.cpp (LLAMA_VOCAB_PRE_TYPE_GPT4O).
const INKLING2_PRE_REGEX: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Exact ChatGLM4 / GLM-4 / GLM-5 pre-tokenization regex (`LLAMA_VOCAB_PRE_TYPE_CHATGLM4`).
const GLM4_PRE_REGEX: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    always_special: Vec<(String, u32)>,
    all_special: Vec<(String, u32)>,
    pre_regex: Regex,
    ignore_merges: bool,
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

fn build_gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
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
        let regex_str = if is_glm {
            GLM4_PRE_REGEX
        } else {
            INKLING2_PRE_REGEX
        };
        let pre_regex = Regex::new(regex_str)?;

        Ok(Self {
            vocab,
            token_types,
            token_to_id,
            merges,
            always_special,
            all_special,
            pre_regex,
            ignore_merges,
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
            self.encode_plain(chunk, &mut ids)?;
            pos = next_special;
        }
        Ok(ids)
    }

    fn encode_plain(&self, chunk: &str, out: &mut Vec<u32>) -> Result<()> {
        for mat in self.pre_regex.find_iter(chunk) {
            let m = mat?;
            let piece = m.as_str();
            if let Some(cached) = self.piece_cache.borrow().get(piece) {
                out.extend_from_slice(cached);
                continue;
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
                if !skip_special {
                    out.push_str(&self.vocab[idx]);
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

    pub fn decode_piece(&self, id: u32) -> String {
        self.decode(&[id], false)
    }

    pub fn token_id(&self, token_str: &str) -> Option<u32> {
        self.token_to_id.get(token_str).copied()
    }
}
