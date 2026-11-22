pub mod metrics;
pub mod routes;

use crate::metrics::ServerMetrics;
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

const THINK_CLOSE: &str = "</think>";

/// Streaming splitter for models whose prompt opens `<think>` before assistant generation.
/// Emits `Reasoning` deltas until `</think>` is observed, then emits `Token` deltas.
pub struct ThinkSplitter {
    pub in_think: bool,
    buf: String,
}

impl ThinkSplitter {
    pub fn new(expect_think_close: bool) -> Self {
        Self {
            in_think: expect_think_close,
            buf: String::new(),
        }
    }

    pub fn push(&mut self, piece: &str, emit_reasoning: &mut impl FnMut(String), emit_token: &mut impl FnMut(String)) {
        if !self.in_think {
            if !piece.is_empty() {
                emit_token(piece.to_owned());
            }
            return;
        }
        self.buf.push_str(piece);
        if let Some(idx) = self.buf.find(THINK_CLOSE) {
            if idx > 0 {
                emit_reasoning(self.buf[..idx].to_owned());
            }
            let after = self.buf[idx + THINK_CLOSE.len()..].to_owned();
            self.buf.clear();
            self.in_think = false;
            if !after.is_empty() {
                emit_token(after);
            }
        } else {
            // Keep at most `THINK_CLOSE.len() - 1` trailing bytes that might prefix `</think>`
            let keep = THINK_CLOSE.len() - 1;
            if self.buf.len() > keep {
                let mut split = self.buf.len() - keep;
                while split > 0 && !self.buf.is_char_boundary(split) {
                    split -= 1;
                }
                if split > 0 {
                    emit_reasoning(self.buf[..split].to_owned());
                    self.buf.drain(..split);
                }
            }
        }
    }

    pub fn finish(&mut self, emit_reasoning: &mut impl FnMut(String), emit_token: &mut impl FnMut(String)) {
        if self.buf.is_empty() {
            return;
        }
        let rem = std::mem::take(&mut self.buf);
        if self.in_think {
            emit_reasoning(rem);
        } else {
            emit_token(rem);
        }
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
            let mut default_stops = vec![tokenizer.eos_token_id];
            for special in [
                "<|endoftext|>",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
                "<|message_user|>",
                "<|user|>",
                "<|observation|>",
            ] {
                if let Some(id) = tokenizer.token_id(special) {
                    if !default_stops.contains(&id) {
                        default_stops.push(id);
                    }
                }
            }

            while let Some(mut job) = rx.blocking_recv() {
                for &s in &default_stops {
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

                let mut gen_ids = Vec::new();
                let mut emitted_bytes = 0usize;
                let mut hit_stop_str = false;
                let mut splitter = ThinkSplitter::new(job.expect_think_close);
                let max_stop_len = job.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

                let resp_tx_cb = &job.resp_tx;
                let mut emit_piece = |s: &str| {
                    splitter.push(
                        s,
                        &mut |r| {
                            let _ = resp_tx_cb.send(StreamEvent::Reasoning(r));
                        },
                        &mut |t| {
                            let _ = resp_tx_cb.send(StreamEvent::Token(t));
                        },
                    );
                };

                let res = engine.generate_stream(&prompt_ids, &job.params, |tok_id| {
                    gen_ids.push(tok_id);
                    let decoded = tokenizer.decode(&gen_ids, true);
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
                            let decoded = tokenizer.decode(&gen_ids, true);
                            if decoded.len() > emitted_bytes {
                                emit_piece(&decoded[emitted_bytes..]);
                            }
                        }
                        splitter.finish(
                            &mut |r| {
                                let _ = job.resp_tx.send(StreamEvent::Reasoning(r));
                            },
                            &mut |t| {
                                let _ = job.resp_tx.send(StreamEvent::Token(t));
                            },
                        );

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

                        let finish_reason = if !hit_stop_str
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
    use crate::routes::{format_glm5_chat, format_inkling2_chat, resolve_glm5_thinking, ChatMessage};
    use crate::ThinkSplitter;

    #[test]
    fn test_multi_turn_chat_prefix_extension() {
        let t1 = vec![ChatMessage {
            role: "user".to_owned(),
            content: Some("Hello".to_owned()),
        }];
        let t2 = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: Some("Hello".to_owned()),
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: Some("Hi there!".to_owned()),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: Some("How are you?".to_owned()),
            },
        ];
        let p1 = format_inkling2_chat(&t1);
        let p2 = format_inkling2_chat(&t2);
        assert!(p2.starts_with(&p1), "multi-turn chat prompt must strictly extend Turn 1 prefix");
    }

    #[test]
    fn test_glm5_chat_template_and_think_splitter() {
        let kw = serde_json::json!({"enable_thinking": false});
        let (en, eff) = resolve_glm5_thinking(Some(&kw), None);
        assert!(!en);
        assert_eq!(eff, "Low");

        let msgs = vec![ChatMessage {
            role: "user".to_owned(),
            content: Some("Hi".to_owned()),
        }];
        let (p_off, exp_off) = format_glm5_chat(&msgs, false, "Low");
        assert!(!exp_off);
        assert_eq!(
            p_off,
            "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think></think>"
        );

        let (p_on, exp_on) = format_glm5_chat(&msgs, true, "Max");
        assert!(exp_on);
        assert_eq!(
            p_on,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>"
        );

        let msgs2 = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: Some("Hi".to_owned()),
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: Some("Hello!".to_owned()),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: Some("Next?".to_owned()),
            },
        ];
        let (p_t2, _) = format_glm5_chat(&msgs2, false, "Low");
        assert!(p_t2.starts_with(&p_off), "GLM-5 multi-turn should preserve exact prefix");

        let mut sp = ThinkSplitter::new(true);
        let mut r = String::new();
        let mut c = String::new();
        for ch in ["th", "ink", "ing</", "thi", "nk>\nans", "wer"] {
            sp.push(ch, &mut |s| r.push_str(&s), &mut |s| c.push_str(&s));
        }
        sp.finish(&mut |s| r.push_str(&s), &mut |s| c.push_str(&s));
        assert_eq!(r, "thinking");
        assert_eq!(c, "\nanswer");
    }
}
