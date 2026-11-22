use ampllm_cache::{plan_layer_capacities, CacheProfile};
use ampllm_gguf::Inkling2Config;

fn mock_inkling2_cfg() -> Inkling2Config {
    let mut is_swa_phys = vec![false; 24];
    for (i, v) in is_swa_phys.iter_mut().enumerate() {
        *v = i % 2 == 0;
    }
    Inkling2Config {
        arch: "inkling2_loop".to_owned(),
        n_layer_phys: 24,
        n_layer_logical: 48,
        n_embd: 2304,
        n_vocab_padded: 201088,
        n_vocab_unpadded: 201088,
        n_head: 24,
        n_head_kv: 8,
        head_dim: 128,
        rms_norm_eps: 1e-6,
        context_length: 32768,
        sliding_window: 1024,
        is_swa_phys,
        d_rel: 16,
        rel_extent: 1024,
        qk_clip: 12.0,
        rel_clip: 12.0,
        log_scale_floor: 256.0,
        log_scale_alpha: 0.25,
        shortconv_l_cache: 4,
        attn_res_block_size: 6,
        n_expert: 128,
        n_expert_used: 8,
        n_ff_exp: 768,
        expert_weights_scale: 2.0,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
        final_logit_softcapping: 30.0,
        is_looped: true,
        loop_start: 6,
        loop_end: 18,
        loop_count: 3,
        loop_gate_rank: 64,
        logical_layers: Vec::new(),
    }
}

#[test]
fn test_budget_planner_loop_core_priority_and_ceiling() {
    let cfg = mock_inkling2_cfg();
    let mut layer_sizes = Vec::with_capacity(24);
    for i in 0..24 {
        if i < 20 {
            layer_sizes.push((1_990_656, 995_328)); // IQ4_NL ~2.98 MiB total
        } else {
            layer_sizes.push((3_760_128, 1_880_064)); // Q8_0 ~5.64 MiB total
        }
    }

    let total_needed: usize = layer_sizes
        .iter()
        .map(|&(gu, dn)| 128 * (((gu + 255) & !255) + ((dn + 255) & !255)))
        .sum();
    for budget_mb in [4096usize, 8192, 12288] {
        let budget_bytes = budget_mb * 1024 * 1024;
        let caps = plan_layer_capacities(&cfg, &layer_sizes, budget_bytes, None);
        let used_bytes: usize = caps
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let (gu, dn) = layer_sizes[i];
                let gu_a = (gu + 255) & !255;
                let dn_a = (dn + 255) & !255;
                c * (gu_a + dn_a)
            })
            .sum();
        assert!(used_bytes <= budget_bytes);
        assert!(used_bytes >= total_needed.min(budget_bytes * 95 / 100));
        let core_avg: f64 = caps[6..18].iter().sum::<usize>() as f64 / 12.0;
        let prelude_avg: f64 = caps[0..6].iter().sum::<usize>() as f64 / 6.0;
        println!("Budget {budget_mb} MiB -> core_avg={core_avg:.1}/128 prelude_avg={prelude_avg:.1}/128");
        assert!(
            core_avg >= 127.9 || core_avg >= prelude_avg * 2.0,
            "budget {budget_mb}MB expected loop core priority: core={core_avg:.1} prelude={prelude_avg:.1}"
        );
    }

    let tmp = std::env::temp_dir().join("ampllm_cache_profile_test.json");
    let prof = CacheProfile {
        arch: "inkling2_loop".to_owned(),
        n_layer_phys: 2,
        n_expert: 4,
        ranked_experts: vec![vec![3, 1, 0, 2], vec![2, 0, 3, 1]],
        global_ranked: Vec::new(),
    };
    prof.save(&tmp).unwrap();
    let loaded = CacheProfile::load(&tmp).unwrap();
    assert_eq!(prof, loaded);
    let _ = std::fs::remove_file(&tmp);

    // Self-contained synthetic STRP v1 profile parser verification
    let mut strp = Vec::new();
    strp.extend_from_slice(b"STRP");
    strp.extend_from_slice(&1u32.to_le_bytes()); // version
    strp.extend_from_slice(&3u32.to_le_bytes()); // n_layers
    strp.extend_from_slice(&4u32.to_le_bytes()); // n_expert
    strp.extend_from_slice(&1u32.to_le_bytes()); // slots
    let pairs: [(u16, u16); 5] = [(1, 2), (0, 3), (1, 0), (2, 1), (0, 3)]; // includes duplicate
    strp.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
    for (l, e) in pairs {
        strp.extend_from_slice(&l.to_le_bytes());
        strp.extend_from_slice(&e.to_le_bytes());
    }
    let parsed = CacheProfile::from_strp_bytes(&strp).unwrap();
    assert_eq!(parsed.n_layer_phys, 3);
    assert_eq!(parsed.n_expert, 4);
    assert_eq!(parsed.global_ranked, vec![(1, 2), (0, 3), (1, 0), (2, 1)]);
    assert_eq!(parsed.ranked_experts[0], vec![3, 0, 1, 2]);
    assert_eq!(parsed.ranked_experts[1], vec![2, 0, 1, 3]);
    assert_eq!(parsed.ranked_experts[2], vec![1, 0, 2, 3]);
}
