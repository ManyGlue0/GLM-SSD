use std::path::PathBuf;

fn main() {
    let cuda = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .expect("CUDA_PATH not set (install the CUDA toolkit)");
    let cuda = PathBuf::from(cuda);
    // Target GPU arch; override with GLMSSD_CUDA_ARCH=sm_89 etc.
    let arch = std::env::var("GLMSSD_CUDA_ARCH").unwrap_or_else(|_| "sm_86".into());

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
    b.cuda(true)
        .cudart("static")
        .flag(format!("-arch={arch}"))
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-std=c++17")
        .flag("-lineinfo")
        .flag("-Xcompiler=/MD") // match Rust's MSVC CRT
        .include("kernels");
    for k in &kernels {
        b.file(k);
    }
    b.compile("glmssd_kernels");

    println!("cargo:rustc-link-search=native={}", cuda.join("lib").join("x64").display());
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=GLMSSD_CUDA_ARCH");
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
