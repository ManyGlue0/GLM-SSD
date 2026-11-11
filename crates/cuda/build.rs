use std::path::PathBuf;

fn main() {
    let cuda = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .expect("CUDA_PATH not set (install the CUDA toolkit)");
    let cuda = PathBuf::from(cuda);
    // Target GPU architecture(s). Default "native": the GPU(s) in this machine. GLMSSD_CUDA_ARCH
    // takes one arch or a comma list, e.g. "sm_89" or "80,86,89,120"; with a list, the last one
    // is also embedded as PTX so newer GPUs can JIT it (release builds).
    // Ampere (sm_80) or newer: the attention kernel needs >64 KB of shared memory per block.
    let arch = std::env::var("GLMSSD_CUDA_ARCH").unwrap_or_else(|_| "native".into());
    let archs: Vec<String> = arch
        .split(',')
        .map(|a| a.trim().trim_start_matches("sm_").trim_start_matches("compute_").to_string())
        .filter(|a| !a.is_empty())
        .collect();

    let kernels: Vec<PathBuf> = std::fs::read_dir("kernels")
        .expect("kernels dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();

    let mut b = cc::Build::new();
    // nvcc needs MSVC's cl.exe as host compiler; it's usually not on PATH outside a
    // "Developer Prompt", so locate it with vswhere and pass it explicitly.
    // NVCC_CCBIN also covers cc's own `nvcc -E` probes, which don't get our flags.
    if let Some(ccbin) = find_msvc_bin() {
        std::env::set_var("NVCC_CCBIN", &ccbin);
    }
    // Never let cc add -G (device debug = no optimization) because the profile has debug info.
    b.debug(false).opt_level(3);
    // cc cannot classify nvcc as a compiler family and warns about it on every build
    b.cargo_warnings(false);
    b.cuda(true)
        .cudart("static")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-std=c++20")
        .flag("-Xcompiler=/MD") // match Rust's MSVC CRT
        .include("kernels");
    // line info for profilers; release builds (GLMSSD_PORTABLE) leave it out, it embeds source paths
    if std::env::var_os("GLMSSD_PORTABLE").is_none() {
        b.flag("-lineinfo");
    }
    if archs.len() == 1 {
        let a = &archs[0];
        b.flag(if a == "native" || a.starts_with("all") { format!("-arch={a}") } else { format!("-arch=sm_{a}") });
    } else {
        for a in &archs {
            b.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        }
        let last = archs.last().expect("GLMSSD_CUDA_ARCH");
        b.flag(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }
    for k in &kernels {
        b.file(k);
    }
    b.compile("glmssd_kernels");

    println!("cargo:rustc-link-search=native={}", cuda.join("lib").join("x64").display());
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=GLMSSD_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=GLMSSD_PORTABLE");
}

/// Directory containing the x64 host cl.exe of the newest Visual Studio install.
fn find_msvc_bin() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let pf86 = std::env::var("ProgramFiles(x86)").ok()?;
    let vswhere = PathBuf::from(pf86).join("Microsoft Visual Studio/Installer/vswhere.exe");
    let out = std::process::Command::new(vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
        .output()
        .ok()?;
    let vs = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let tools = vs.join("VC/Tools/MSVC");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&tools).ok()?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    versions.sort();
    let bin = versions.pop()?.join("bin/Hostx64/x64");
    bin.join("cl.exe").exists().then_some(bin)
}
