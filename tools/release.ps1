# Builds the Windows release zip: target\dist\glm-ssd-v<version>-windows-x64.zip
#
# - GLMSSD_PORTABLE: the exe looks for models next to itself (or in GLMSSD_MODELS), never in
#   this repository's folder.
# - GLMSSD_CUDA_ARCH: native code for Ampere (sm_80/86), Ada (sm_89), Hopper (sm_90) and
#   Blackwell (sm_120), plus sm_120 PTX that newer GPUs can JIT-compile.
# Uses its own target dir so it doesn't invalidate the normal (native-arch) build.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:GLMSSD_PORTABLE = "1"
$env:GLMSSD_CUDA_ARCH = "80,86,89,90,120"
cargo build --release --locked -p glm-ssd --target-dir target\portable
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Remove-Item Env:GLMSSD_PORTABLE, Env:GLMSSD_CUDA_ARCH

$version = (Select-String -Path crates\engine\Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$name = "glm-ssd-v$version-windows-x64"
$stage = "target\dist\$name"
Remove-Item -Recurse -Force $stage, "target\dist\$name.zip" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item target\portable\release\glm-ssd.exe, README.md, LICENSE, THIRD-PARTY-NOTICES.md $stage
Compress-Archive -Path "$stage\*" -DestinationPath "target\dist\$name.zip"
$hash = (Get-FileHash "target\dist\$name.zip" -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Set-Content -NoNewline "target\dist\$name.zip.sha256"
Write-Host "target\dist\$name.zip"
Write-Host "sha256 $hash"
