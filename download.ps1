# Downloads GLM-5.3-Flash UD-IQ2_XXS (~102 GB) into .\models using the parallel Rust downloader.
# Resumable: just run it again after an interruption. Extra args are passed through, e.g.:
#   .\download.ps1 -c 32                          # more connections
#   .\download.ps1 --include "UD-Q2_K_XL/*"       # a different quant
#   .\download.ps1 --url https://huggingface.co/owner/name               # another repo (GGUF picked automatically)
#   .\download.ps1 --url https://huggingface.co/owner/name/tree/main/UD-IQ2_XXS   # a folder of it
#   .\download.ps1 --repo owner/name --include "*IQ2_XXS*"                # same, by repo id
# Other repos go to .\models\<owner>--<name>\ (never over the official files); at the end the
# script prints the glm-ssd command (-m takes the folder, or just its name: -m owner--name).
Set-Location $PSScriptRoot
cargo run --release -p hfdl -- @args
