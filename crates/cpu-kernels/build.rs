use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/iq_tables.hpp");
    println!("cargo:rerun-if-changed=csrc/iq_avx512_glm5.cpp");
    println!("cargo:rerun-if-changed=csrc/iq_avx512_hy4.cpp");
    println!("cargo:rerun-if-changed=csrc/iq_avx512_mm3.cpp");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AMPLLM_CPU_TUNE");

    let cxx = env::var("CXX").unwrap_or_else(|_| "g++".to_owned());
    let tune = env::var("AMPLLM_CPU_TUNE").unwrap_or_else(|_| "native".to_owned());
    let tune_flag = format!("-mtune={tune}");
    let lib = out_dir.join("libampllm_cpu_iq512.a");

    let sources = ["iq_avx512_glm5.cpp", "iq_avx512_hy4.cpp", "iq_avx512_mm3.cpp"];
    let mut objs = Vec::with_capacity(sources.len());
    for file in &sources {
        let src = csrc_dir.join(file);
        let obj = out_dir.join(format!("{}.o", file.trim_end_matches(".cpp")));
        let out = Command::new(&cxx)
            .args([
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-mavx512f",
                "-mavx512bw",
                "-mavx512vl",
                "-mavx512dq",
                "-mavx512vnni",
                "-mavx512vpopcntdq",
                "-mavx2",
                "-mfma",
                "-mf16c",
                "-mbmi2",
                tune_flag.as_str(),
                "-I",
                csrc_dir.to_str().unwrap(),
                "-c",
                src.to_str().unwrap(),
                "-o",
                obj.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn {cxx}: {e}"));

        if !out.status.success() {
            panic!(
                "{cxx} failed compiling {file}:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        objs.push(obj);
    }

    if lib.exists() {
        let _ = std::fs::remove_file(&lib);
    }
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib);
    for o in &objs {
        ar_cmd.arg(o);
    }
    let ar_st = ar_cmd.status().expect("failed to run ar");
    assert!(ar_st.success(), "ar failed archiving libampllm_cpu_iq512.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ampllm_cpu_iq512");
    println!("cargo:rustc-link-lib=stdc++");
}
