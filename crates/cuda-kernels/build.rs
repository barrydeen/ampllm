use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> String {
    if let Ok(nvcc) = env::var("NVCC") {
        if !nvcc.trim().is_empty() {
            return nvcc;
        }
    }
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(var) {
            let candidate = Path::new(&root).join("bin").join("nvcc");
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    for std_path in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"] {
        if Path::new(std_path).exists() {
            return std_path.to_owned();
        }
    }
    "nvcc".to_owned()
}

fn push_lib_dirs(dirs: &mut BTreeSet<PathBuf>, root: &Path) {
    for sub in ["lib64", "lib"] {
        let p = root.join(sub);
        if p.is_dir() {
            dirs.insert(p);
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cu_files = [
        "quant_gemv.cu",
        "moe_grouped.cu",
        "inkling_ops.cu",
        "flash_attn_rel.cu",
        "doorbell.cu",
        "comm_p2p.cu",
        "glm5_ops.cu",
        "glm5_gemm_mma.cu",
        "glm5_mmq_mma.cu",
        "hy4_ops.cu",
        "qwen4_ops.cu",
        "mm3_ops.cu",
    ];

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/common/sm86_common.cuh");
    for f in &cu_files {
        println!("cargo:rerun-if-changed=csrc/{f}");
    }
    for var in [
        "NVCC",
        "CUDA_HOME",
        "CUDA_PATH",
        "AMPLLM_CUDA_ARCH",
        "AMPLLM_NVCC_VERBOSE",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let nvcc_bin = find_nvcc();
    let cuda_arch = env::var("AMPLLM_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_owned());
    let arch_flag = format!("-arch={cuda_arch}");
    let verbose = env::var("AMPLLM_NVCC_VERBOSE").as_deref() == Ok("1");

    let mut objs = Vec::with_capacity(cu_files.len());
    for f in &cu_files {
        let src = csrc_dir.join(f);
        let obj = out_dir.join(format!("{}.o", f.trim_end_matches(".cu")));
        let out = Command::new(&nvcc_bin)
            .args([
                arch_flag.as_str(),
                "-O3",
                "-std=c++17",
                "-Xcompiler",
                "-fPIC",
                "-Xptxas=-v",
                "-I",
                csrc_dir.to_str().unwrap(),
                "-c",
                src.to_str().unwrap(),
                "-o",
                obj.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn {nvcc_bin} for {f}: {e}"));

        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines() {
            let nonzero_spill = (line.contains("spill stores")
                && !line.contains("0 bytes spill stores"))
                || (line.contains("spill loads") && !line.contains("0 bytes spill loads"));
            let is_diag = line.contains("warning")
                || line.contains("Warning")
                || line.contains("error")
                || line.contains("Error");
            if verbose || nonzero_spill || is_diag {
                println!("cargo:warning=[nvcc {f}] {line}");
            }
        }
        if !out.status.success() {
            panic!("nvcc failed compiling {f}:\n{stderr}");
        }
        objs.push(obj);
    }

    let lib_path = out_dir.join("libampllm_kernels.a");
    if lib_path.exists() {
        let _ = std::fs::remove_file(&lib_path);
    }
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for o in &objs {
        ar_cmd.arg(o);
    }
    let ar_status = ar_cmd.status().expect("failed to run ar");
    assert!(ar_status.success(), "ar failed archiving libampllm_kernels.a");

    let mut lib_dirs = BTreeSet::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(var) {
            push_lib_dirs(&mut lib_dirs, Path::new(&root));
        }
    }
    if let Some(root) = Path::new(&nvcc_bin).parent().and_then(|p| p.parent()) {
        if !root.as_os_str().is_empty() {
            push_lib_dirs(&mut lib_dirs, root);
        }
    }
    for std_root in ["/usr/local/cuda", "/opt/cuda"] {
        push_lib_dirs(&mut lib_dirs, Path::new(std_root));
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=ampllm_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
}
