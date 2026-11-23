use crate::hy4::forward::{attn_core_layer, decode_tokens_to_host, ffn_core_layer, mm_hy4, Hy4ImageSpan};
use crate::hy4::state::{Hy4State, HY4_ATTN_SUB};
use crate::hy4::weights::Hy4ModelWeights;
use ampllm_cuda_kernels::{
    copy_d2d_raw_async, copy_d2h_raw_async, copy_h2d_raw_async, hy4, set_device, zero_raw_async,
};
use anyhow::Result;

unsafe fn copy_hring_to_mtp_hin(
    weights: &Hy4ModelWeights,
    st: &Hy4State,
    q0: usize,
    n_t: usize,
    mtp_hin: *mut f32,
) -> Result<()> {
    let e = weights.cfg.n_embd;
    let hring_cap = weights.hring_cap;
    let stream = &st.sc[weights.head_gpu_idx].stream;
    let hring_f32 = st.hring.ptr as *const f32;

    let mut t = 0usize;
    while t < n_t {
        let q = q0 + t;
        if q == 0 {
            zero_raw_async(mtp_hin.cast(), e * 4, stream)?;
            t += 1;
            continue;
        }
        let r0 = (q - 1) % hring_cap;
        let run = (n_t - t).min(hring_cap - r0);
        copy_d2d_raw_async(
            mtp_hin.add(t * e).cast(),
            hring_f32.add(r0 * e).cast(),
            run * e * 4,
            stream,
        )?;
        t += run;
    }
    Ok(())
}

/// Forward `T` tokens through the `blk.42` MTP block on `head_gpu_idx`, using `mtp_hin[0..T, E]`.
/// When `want == 0`, runs attention/indexer cache population only (fast prefill catch-up).
pub unsafe fn mtp_forward(
    weights: &Hy4ModelWeights,
    st: &mut Hy4State,
    ids: &[u32],
    p0: usize,
    want: usize,
    imgs: &[Hy4ImageSpan],
) -> Result<()> {
    let Some(ref ml) = weights.mtp_layer else {
        return Ok(());
    };
    let Some(ref mc) = st.mtp_cache else {
        return Ok(());
    };
    let t_len = ids.len();
    if t_len == 0 {
        return Ok(());
    }
    let mtp_sub = HY4_ATTN_SUB.min(weights.tc);
    anyhow::ensure!(t_len <= mtp_sub, "mtp_forward window {t_len} exceeds {mtp_sub}");
    let cfg = &weights.cfg;
    let e = cfg.n_embd;
    let eps = cfg.rms_norm_eps;
    let gi = ml.gpu_idx;
    let sc = &st.sc[gi];
    set_device(sc.device)?;

    let mtp_in = st.mtp_in.as_ref().unwrap().ptr as *mut f32;
    let mtp_cur = st.mtp_cur.as_ref().unwrap().ptr as *mut f32;
    let mtp_hin = st.mtp_hin.as_ref().unwrap().ptr as *mut f32;
    let mtp_out = st.mtp_out.as_ref().unwrap().ptr as *mut f32;

    {
        let dst = std::slice::from_raw_parts_mut(st.h_emb_mtp.as_mut_ptr().cast::<f32>(), t_len * e);
        decode_tokens_to_host(weights, ids, p0, imgs, dst)?;
    }

    copy_h2d_raw_async(
        mtp_cur.cast(),
        std::slice::from_raw_parts(st.h_emb_mtp.as_mut_ptr(), t_len * e * 4),
        &sc.stream,
    )?;

    hy4::rmsnorm_strided(
        mtp_cur,
        e,
        ml.mtp_enorm.as_ref().unwrap().ptr.cast(),
        mtp_in,
        2 * e,
        e as i32,
        t_len as i32,
        eps,
        &sc.stream,
    );
    hy4::rmsnorm_strided(
        mtp_hin,
        e,
        ml.mtp_hnorm.as_ref().unwrap().ptr.cast(),
        mtp_in.add(e),
        2 * e,
        e as i32,
        t_len as i32,
        eps,
        &sc.stream,
    );

    mm_hy4(
        ml.mtp_eh_proj.as_ref().unwrap(),
        mtp_in,
        std::ptr::null(),
        t_len,
        mtp_cur,
        1.0,
        sc,
    );
    hy4::rmsnorm_strided(
        mtp_cur,
        e,
        ml.attn_norm.ptr.cast(),
        sc.xa,
        e,
        e as i32,
        t_len as i32,
        eps,
        &sc.stream,
    );

    if want == 0 {
        return attn_core_layer(weights, ml, mc, sc, t_len, 0, p0, true);
    }

    attn_core_layer(weights, ml, mc, sc, t_len, 0, p0, false)?;
    hy4::rmsnorm_strided(
        sc.y,
        e,
        ml.post_attn.ptr.cast(),
        sc.y,
        e,
        e as i32,
        t_len as i32,
        eps,
        &sc.stream,
    );
    hy4::add_f32(mtp_cur, sc.y, t_len * e, &sc.stream);

    // Only compute FFN & shared_head_norm on the last `want` rows!
    let f0 = t_len - want;
    let tf = want;
    let cf = mtp_cur.add(f0 * e);
    hy4::rmsnorm_strided(
        cf,
        e,
        ml.ffn_norm.ptr.cast(),
        sc.xa,
        e,
        e as i32,
        tf as i32,
        eps,
        &sc.stream,
    );

    let h_ids_s = std::slice::from_raw_parts_mut(st.h_ids.as_mut_ptr().cast::<i32>(), st.h_ids.bytes() / 4);
    let h_w_s = std::slice::from_raw_parts_mut(st.h_w.as_mut_ptr().cast::<f32>(), st.h_w.bytes() / 4);
    let h_lat_s = std::slice::from_raw_parts_mut(st.h_lat.as_mut_ptr().cast::<f32>(), st.h_lat.bytes() / 4);
    let h_cpu_s = std::slice::from_raw_parts_mut(st.h_cpu.as_mut_ptr().cast::<f32>(), st.h_cpu.bytes() / 4);
    ffn_core_layer(
        weights,
        ml,
        sc,
        tf,
        h_ids_s,
        h_w_s,
        h_lat_s,
        h_cpu_s,
        &st.cpu_pool,
        &mut st.cpu_scratch,
        &mut st.profile,
    )?;

    hy4::rmsnorm_strided(
        sc.y,
        e,
        ml.post_ffn.ptr.cast(),
        sc.y,
        e,
        e as i32,
        tf as i32,
        eps,
        &sc.stream,
    );
    hy4::add_f32(cf, sc.y, tf * e, &sc.stream);
    hy4::rmsnorm_strided(
        cf,
        e,
        ml.mtp_head_norm.as_ref().unwrap().ptr.cast(),
        mtp_out.add(f0 * e),
        e,
        e as i32,
        tf as i32,
        eps,
        &sc.stream,
    );

    mm_hy4(
        &weights.head.out_w,
        mtp_out.add(f0 * e),
        std::ptr::null(),
        tf,
        sc.logits,
        1.0,
        sc,
    );
    Ok(())
}

/// Catch up MTP cache `[mtp_done .. upto)` (cache-only) after trunk prefill / decode chunks.
pub unsafe fn mtp_catch_up(
    weights: &Hy4ModelWeights,
    st: &mut Hy4State,
    upto: usize,
    imgs: &[Hy4ImageSpan],
) -> Result<()> {
    if weights.mtp_layer.is_none() || st.mtp_done >= upto {
        return Ok(());
    }
    let hring_cap = weights.hring_cap;
    let tc = HY4_ATTN_SUB.min(weights.tc);
    let hgi = weights.head_gpu_idx;
    let dev = weights.gpus[hgi];
    set_device(dev)?;
    let mtp_hin = st.mtp_hin.as_ref().unwrap().ptr as *mut f32;

    let mut c0 = st.mtp_done;
    if c0 + hring_cap < upto + 1 {
        c0 = upto.saturating_sub(hring_cap - 1);
    }

    let mut a = c0;
    while a < upto {
        let n_t = (upto - a).min(tc);
        {
            let sc = &st.sc[hgi];
            let h_pos_s = std::slice::from_raw_parts_mut(st.h_pos.as_mut_ptr().cast::<i32>(), n_t);
            for (t, slot) in h_pos_s.iter_mut().enumerate() {
                *slot = (a + t) as i32;
            }
            copy_h2d_raw_async(
                sc.pos.cast(),
                std::slice::from_raw_parts(h_pos_s.as_ptr() as *const u8, n_t * 4),
                &sc.stream,
            )?;
        }
        copy_hring_to_mtp_hin(weights, st, a, n_t, mtp_hin)?;
        let chunk_ids = st.history[a..a + n_t].to_vec();
        mtp_forward(weights, st, &chunk_ids, a, 0, imgs)?;
        a += n_t;
    }
    st.sc[hgi].stream.sync()?;
    st.mtp_done = upto;
    Ok(())
}

/// Draft up to `max_draft` tokens starting at position `pos` given the latest committed token `tok`,
/// stopping early if any draft's softmax confidence drops below `floor`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_draft_tokens(
    weights: &Hy4ModelWeights,
    st: &mut Hy4State,
    tok: u32,
    pos: usize,
    max_draft: usize,
    floor: f32,
    imgs: &[Hy4ImageSpan],
    out_drafts: &mut Vec<u32>,
    out_conf: &mut Vec<f32>,
) -> Result<()> {
    out_drafts.clear();
    out_conf.clear();
    if weights.mtp_layer.is_none() || max_draft == 0 || pos + max_draft >= weights.max_ctx {
        return Ok(());
    }

    let hgi = weights.head_gpu_idx;
    let dev = weights.gpus[hgi];
    set_device(dev)?;
    let hring_cap = weights.hring_cap;
    let tc = HY4_ATTN_SUB.min(weights.tc);
    let e = weights.cfg.n_embd;
    let mtp_hin = st.mtp_hin.as_ref().unwrap().ptr as *mut f32;
    let mtp_out = st.mtp_out.as_ref().unwrap().ptr as *mut f32;

    let mut c0 = st.mtp_done.min(pos);
    if c0 + hring_cap < pos + 1 {
        c0 = pos.saturating_sub(hring_cap - 1);
    }

    let mut rows: Vec<u32> = Vec::with_capacity(pos - c0 + 1);
    if c0 < pos {
        rows.extend_from_slice(&st.history[c0..pos]);
    }
    rows.push(tok);

    let mut a = 0usize;
    let mut last_local_row = 0usize;
    while a < rows.len() {
        let n_t = (rows.len() - a).min(tc);
        let q0 = c0 + a;
        let is_last = a + n_t == rows.len();
        {
            let sc = &st.sc[hgi];
            let h_pos_s = std::slice::from_raw_parts_mut(st.h_pos.as_mut_ptr().cast::<i32>(), n_t);
            for (t, slot) in h_pos_s.iter_mut().enumerate() {
                *slot = (q0 + t) as i32;
            }
            copy_h2d_raw_async(
                sc.pos.cast(),
                std::slice::from_raw_parts(h_pos_s.as_ptr() as *const u8, n_t * 4),
                &sc.stream,
            )?;
        }
        copy_hring_to_mtp_hin(weights, st, q0, n_t, mtp_hin)?;
        let slice_ids = rows[a..a + n_t].to_vec();
        mtp_forward(weights, st, &slice_ids, q0, usize::from(is_last), imgs)?;
        if is_last {
            last_local_row = n_t - 1;
        }
        a += n_t;
    }
    st.mtp_done = pos + 1;

    let mut cur = tok;
    let cap = max_draft.min(weights.max_ctx - pos - 1);
    for i in 0..cap {
        if i > 0 {
            {
                let sc = &st.sc[hgi];
                let h_pos_s = std::slice::from_raw_parts_mut(st.h_pos.as_mut_ptr().cast::<i32>(), 1);
                h_pos_s[0] = (pos + i) as i32;
                copy_h2d_raw_async(
                    sc.pos.cast(),
                    std::slice::from_raw_parts(h_pos_s.as_ptr() as *const u8, 4),
                    &sc.stream,
                )?;
            }
            mtp_forward(weights, st, &[cur], pos + i, 1, imgs)?;
        }

        {
            let sc = &st.sc[hgi];
            hy4::top1_prob(
                sc.logits,
                weights.cfg.n_vocab as i32,
                sc.top1_scratch,
                &sc.stream,
            );
            copy_d2h_raw_async(
                std::slice::from_raw_parts_mut(st.h_top1.as_mut_ptr(), 8),
                sc.top1_scratch.cast(),
                &sc.stream,
            )?;
            sc.stream.sync()?;
        }

        let top_pair = std::slice::from_raw_parts(st.h_top1.as_mut_ptr().cast::<f32>(), 2);
        let id = top_pair[0].to_bits();
        let prob = top_pair[1];
        out_drafts.push(id);
        out_conf.push(prob);
        if prob < floor || i + 1 == cap {
            break;
        }

        cur = id;
        let row_src = if i == 0 { last_local_row } else { 0 };
        let sc = &st.sc[hgi];
        copy_d2d_raw_async(
            mtp_hin.cast(),
            mtp_out.add(row_src * e).cast(),
            e * 4,
            &sc.stream,
        )?;
    }

    Ok(())
}
