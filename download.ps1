# Downloads GLM-5.3-Flash UD-IQ2_XXS (~102 GB) into .\models using the parallel Rust downloader.
# Resumable: just run it again after an interruption. Extra args are passed through, e.g.:
#   .\download.ps1 -c 32                          # more connections
#   .\download.ps1 --include "UD-Q2_K_XL/*"       # a different quant
Set-Location $PSScriptRoot
cargo run --release -p hfdl -- @args
