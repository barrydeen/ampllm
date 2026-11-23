use ampllm_cuda_kernels::{CudaStream, DeviceBuffer};
use ampllm_gguf::Inkling2Config;
use anyhow::Result;

/// Pre-allocated device activation and intermediate buffers (zero allocations during decode/prefill).
pub struct ActivationScratch {
    pub max_batch: usize,
    pub n_head_local: usize,
    pub n_head_kv_local: usize,
    pub n_ff_exp_local: usize,
    pub n_vocab_local: usize,

    pub cur: DeviceBuffer,
    pub inp_sa: DeviceBuffer,
    pub ffn_inp: DeviceBuffer,
    pub tmp_hidden: DeviceBuffer,
    pub prelude: DeviceBuffer,
    pub bank_out: DeviceBuffer,
    pub blocks: Vec<DeviceBuffer>, // up to 6 block summaries [max_batch, n_embd]

    // AttnRes & loop controller scratch
    pub mix_out: DeviceBuffer,       // [4, max_batch, n_embd]
    pub loop_messages: DeviceBuffer, // [4, max_batch, n_embd]
    pub ctrl_concat: DeviceBuffer,   // [max_batch, 2 * n_embd]
    pub ctrl_gate: DeviceBuffer,     // [max_batch, gate_rank]
    pub ctrl_forget: DeviceBuffer,   // [max_batch, n_embd]
    pub ctrl_gain: DeviceBuffer,     // [max_batch, n_embd]
    pub inputs_qkv: [DeviceBuffer; 3],
    pub inputs_norm: [DeviceBuffer; 3],

    // Quantized activation scratch [3, max_batch, max(q_dim, 2 * n_embd) / 32 * 36]
    pub act_q8_1: DeviceBuffer,
    pub act_q8_1_k: DeviceBuffer,
    pub act_q8_1_v: DeviceBuffer,

    // Attention & short conv scratch
    pub q_cur: DeviceBuffer,
    pub k_raw: DeviceBuffer,
    pub k_cur: DeviceBuffer,
    pub v_raw: DeviceBuffer,
    pub v_cur: DeviceBuffer,
    pub r_cur: DeviceBuffer,
    pub dyn_proj: DeviceBuffer,
    pub sinks: DeviceBuffer,
    pub attn_out: DeviceBuffer,
    pub attn_gate_buf: DeviceBuffer,
    pub attn_proj: DeviceBuffer,
    pub rel_table: DeviceBuffer, // [max_batch, n_head_local, rel_extent + 1]
    pub split_out: DeviceBuffer, // [max(max_batch, 8), n_head_local, 64, head_dim]
    pub split_ml: DeviceBuffer,  // [max(max_batch, 8), n_head_local, 64, 2]

    // Fast FP16 staging + Inkling-2 verify/decode fast-path scratch
    pub x16_0: DeviceBuffer,           // [max_batch, max(2 * n_embd, q_dim)] u16
    pub x16_1: DeviceBuffer,           // [max_batch, max(2 * n_embd, q_dim)] u16
    pub x16_2: DeviceBuffer,           // [max_batch, max(2 * n_embd, q_dim)] u16
    pub q16: DeviceBuffer,             // [max_batch, q_dim] u16
    pub out16: DeviceBuffer,           // [max_batch, q_dim] u16
    pub x_limbs: DeviceBuffer,         // [8 * 2 * n_embd] u16
    pub moe_inter_f16: DeviceBuffer,   // [64 * n_ff_exp_local] u16
    pub grp_gu_ptrs: DeviceBuffer,     // [64] u64
    pub grp_dn_ptrs: DeviceBuffer,     // [64] u64
    pub grp_n: DeviceBuffer,           // [64] i32
    pub grp_tok: DeviceBuffer,         // [64 * 8] i32
    pub grp_row: DeviceBuffer,         // [64 * 8] i32
    pub slot_sg: DeviceBuffer,         // [64] f32
    pub slot_su: DeviceBuffer,         // [64] f32
    pub slot_wt: DeviceBuffer,         // [64] f32
    pub argmax_part_vals: DeviceBuffer,// [8 * 328] f32
    pub argmax_part_ids: DeviceBuffer, // [8 * 328] i32
    pub argmax_ids: DeviceBuffer,      // [8] i32
    pub taps: DeviceBuffer,            // [4 * max_batch * n_embd] f32
    pub rec_old_hist: DeviceBuffer,    // [n_layers_logical * n_sconv_channels * 3] f32
    pub rec_x: DeviceBuffer,           // [n_layers_logical * 8 * n_sconv_channels] f32
    pub conv_hist_ptrs_dev: DeviceBuffer, // [n_layers_logical] u64

    // MoE scratch
    pub router_logits: DeviceBuffer,  // [max_batch, n_expert]
    pub expert_ids: DeviceBuffer,     // [max_batch, topk] i32
    pub expert_weights: DeviceBuffer, // [max_batch, topk] f32
    pub moe_inter_f32: DeviceBuffer,  // [max_batch, topk, n_ff_exp_local]
    pub moe_inter_q8_1: DeviceBuffer, // [max_batch, topk, n_ff_exp_local / 32 * 36]
    pub moe_slot_out: DeviceBuffer,   // [max_batch, topk, n_embd]
    pub moe_out: DeviceBuffer,        // [max_batch, n_embd]
    pub ffn_out: DeviceBuffer,        // [max_batch, n_embd]

    // Inputs & output logits
    pub token_ids: DeviceBuffer,  // [max_batch] i32
    pub pos_ids: DeviceBuffer,    // [max_batch] i32
    pub tau: DeviceBuffer,        // [max_batch] f32
    pub logits: DeviceBuffer,     // [n_vocab_local] f32
    pub argmax_id: DeviceBuffer,  // [1] i32
    pub argmax_val: DeviceBuffer, // [1] f32

    pub state_ptrs_scratch: DeviceBuffer, // [16] u64
}

impl ActivationScratch {
    pub fn alloc(cfg: &Inkling2Config, device: i32, max_batch: usize) -> Result<Self> {
        Self::alloc_custom(cfg, device, max_batch, None, None, None, None)
    }

    pub fn alloc_custom(
        cfg: &Inkling2Config,
        device: i32,
        max_batch: usize,
        n_head_override: Option<usize>,
        n_head_kv_override: Option<usize>,
        n_ff_exp_override: Option<usize>,
        n_vocab_override: Option<usize>,
    ) -> Result<Self> {
        let e = cfg.n_embd;
        let n_head = n_head_override.unwrap_or(cfg.n_head);
        let n_head_kv = n_head_kv_override.unwrap_or(cfg.n_head_kv);
        let n_ff_exp = n_ff_exp_override.unwrap_or(cfg.n_ff_exp);
        let n_vocab = n_vocab_override.unwrap_or(cfg.n_vocab_padded);

        let q_dim = n_head * cfg.head_dim;
        let kv_dim = n_head_kv * cfg.head_dim;
        let rel_dim = n_head * cfg.d_rel;
        let dyn_dim = n_head_kv * cfg.shortconv_l_cache;
        let x16_cols = q_dim.max(2 * e);
        let split_slots = (max_batch * 8).max(8 * 64);
        let n_slots = cfg.logical_layers.len().max(48);
        let n_sconv_channels = 2 * kv_dim + 2 * e;

        let hid_bytes = max_batch * e * 4;
        let blocks = (0..6)
            .map(|_| DeviceBuffer::alloc(device, hid_bytes))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            max_batch,
            n_head_local: n_head,
            n_head_kv_local: n_head_kv,
            n_ff_exp_local: n_ff_exp,
            n_vocab_local: n_vocab,
            cur: DeviceBuffer::alloc(device, hid_bytes)?,
            inp_sa: DeviceBuffer::alloc(device, hid_bytes)?,
            ffn_inp: DeviceBuffer::alloc(device, hid_bytes)?,
            tmp_hidden: DeviceBuffer::alloc(device, hid_bytes)?,
            prelude: DeviceBuffer::alloc(device, hid_bytes)?,
            bank_out: DeviceBuffer::alloc(device, hid_bytes)?,
            blocks,
            mix_out: DeviceBuffer::alloc(device, 4 * hid_bytes)?,
            loop_messages: DeviceBuffer::alloc(
                device,
                cfg.loop_end.saturating_sub(cfg.loop_start).max(1) * 4 * hid_bytes,
            )?,
            ctrl_concat: DeviceBuffer::alloc(device, 2 * hid_bytes)?,
            ctrl_gate: DeviceBuffer::alloc(device, 3 * max_batch * (2 * cfg.loop_gate_rank.max(128) + 2) * 4)?,
            ctrl_forget: DeviceBuffer::alloc(device, hid_bytes)?,
            ctrl_gain: DeviceBuffer::alloc(device, hid_bytes)?,
            inputs_qkv: [
                DeviceBuffer::alloc(device, hid_bytes)?,
                DeviceBuffer::alloc(device, hid_bytes)?,
                DeviceBuffer::alloc(device, hid_bytes)?,
            ],
            inputs_norm: [
                DeviceBuffer::alloc(device, hid_bytes)?,
                DeviceBuffer::alloc(device, hid_bytes)?,
                DeviceBuffer::alloc(device, hid_bytes)?,
            ],
            act_q8_1: DeviceBuffer::alloc(device, max_batch * (q_dim.max(2 * e) / 32) * 36)?,
            act_q8_1_k: DeviceBuffer::alloc(device, max_batch * (e / 32) * 36)?,
            act_q8_1_v: DeviceBuffer::alloc(device, max_batch * (e / 32) * 36)?,
            q_cur: DeviceBuffer::alloc(device, max_batch * q_dim * 4)?,
            k_raw: DeviceBuffer::alloc(device, max_batch * kv_dim * 4)?,
            k_cur: DeviceBuffer::alloc(device, max_batch * kv_dim * 4)?,
            v_raw: DeviceBuffer::alloc(device, max_batch * kv_dim * 4)?,
            v_cur: DeviceBuffer::alloc(device, max_batch * kv_dim * 4)?,
            r_cur: DeviceBuffer::alloc(device, max_batch * rel_dim * 4)?,
            dyn_proj: DeviceBuffer::alloc(device, 2 * max_batch * dyn_dim * 4)?,
            sinks: DeviceBuffer::alloc(device, max_batch * n_head * 4)?,
            attn_out: DeviceBuffer::alloc(device, max_batch * q_dim * 4)?,
            attn_gate_buf: DeviceBuffer::alloc(device, max_batch * q_dim * 4)?,
            attn_proj: DeviceBuffer::alloc(device, hid_bytes)?,
            rel_table: DeviceBuffer::alloc(device, max_batch * n_head * (cfg.rel_extent + 1) * 4)?,
            split_out: DeviceBuffer::alloc(device, split_slots * n_head * cfg.head_dim * 4)?,
            split_ml: DeviceBuffer::alloc(device, split_slots * n_head * 2 * 4)?,
            x16_0: DeviceBuffer::alloc(device, max_batch * x16_cols * 2)?,
            x16_1: DeviceBuffer::alloc(device, max_batch * x16_cols * 2)?,
            x16_2: DeviceBuffer::alloc(device, max_batch * x16_cols * 2)?,
            q16: DeviceBuffer::alloc(device, max_batch * q_dim * 2)?,
            out16: DeviceBuffer::alloc(device, max_batch * q_dim * 2)?,
            x_limbs: DeviceBuffer::alloc(device, 8 * 2 * e * 2)?,
            moe_inter_f16: DeviceBuffer::alloc(device, 64 * n_ff_exp * 2)?,
            grp_gu_ptrs: DeviceBuffer::alloc(device, 64 * 8)?,
            grp_dn_ptrs: DeviceBuffer::alloc(device, 64 * 8)?,
            grp_n: DeviceBuffer::alloc(device, 64 * 4)?,
            grp_tok: DeviceBuffer::alloc(device, 64 * 8 * 4)?,
            grp_row: DeviceBuffer::alloc(device, 64 * 8 * 4)?,
            slot_sg: DeviceBuffer::alloc(device, 64 * 4)?,
            slot_su: DeviceBuffer::alloc(device, 64 * 4)?,
            slot_wt: DeviceBuffer::alloc(device, 64 * 4)?,
            argmax_part_vals: DeviceBuffer::alloc(device, 8 * 656 * 4)?,
            argmax_part_ids: DeviceBuffer::alloc(device, 8 * 656 * 4)?,
            argmax_ids: DeviceBuffer::alloc(device, 8 * 4)?,
            taps: DeviceBuffer::alloc(device, 4 * hid_bytes)?,
            rec_old_hist: DeviceBuffer::alloc(device, n_slots * n_sconv_channels * 3 * 4)?,
            rec_x: DeviceBuffer::alloc(device, n_slots * 8 * n_sconv_channels * 4)?,
            conv_hist_ptrs_dev: DeviceBuffer::alloc(device, n_slots * 8)?,
            router_logits: DeviceBuffer::alloc(device, max_batch * cfg.n_expert * 4)?,
            expert_ids: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * 4)?,
            expert_weights: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * 4)?,
            moe_inter_f32: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * n_ff_exp * 4)?,
            moe_inter_q8_1: DeviceBuffer::alloc(
                device,
                max_batch * cfg.n_expert_used * (n_ff_exp / 32) * 36,
            )?,
            moe_slot_out: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * e * 4)?,
            moe_out: DeviceBuffer::alloc(device, hid_bytes)?,
            ffn_out: DeviceBuffer::alloc(device, hid_bytes)?,
            token_ids: DeviceBuffer::alloc(device, max_batch * 4)?,
            pos_ids: DeviceBuffer::alloc(device, max_batch * 4)?,
            tau: DeviceBuffer::alloc(device, max_batch * 4)?,
            logits: DeviceBuffer::alloc(device, n_vocab * 4)?,
            argmax_id: DeviceBuffer::alloc(device, 4)?,
            argmax_val: DeviceBuffer::alloc(device, 4)?,
            state_ptrs_scratch: DeviceBuffer::alloc(device, 16 * 8)?,
        })
    }

    pub fn upload_state_ptrs(&self, ptrs: &[u64], stream: &CudaStream) -> Result<()> {
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, std::mem::size_of_val(ptrs))
        };
        self.state_ptrs_scratch.copy_from_host_async(byte_slice, stream)
    }
}
