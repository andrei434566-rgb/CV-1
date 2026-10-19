# CV-1 local VLM setup (Windows PowerShell). Requires NVIDIA GPU, internet and disk space.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$root = Join-Path (Split-Path $PSScriptRoot -Parent) "vlm_core_v1"
$venv = Join-Path $root ".venv"
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required. Install it, then rerun." }
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA installation failed" }
& $python -m pip install "transformers>=4.57,<5" accelerate peft safetensors pillow huggingface_hub
if ($LASTEXITCODE -ne 0) { throw "VLM dependencies installation failed" }
& $python -c "import torch; assert torch.cuda.is_available(), 'NVIDIA CUDA GPU is not available'; from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print('CUDA:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA / Qwen compatibility check failed" }
$cache = Join-Path $root "model_cache\hub"
& $python -c "from huggingface_hub import snapshot_download; import sys; print('MODEL_PATH=' + snapshot_download(repo_id='Qwen/Qwen3-VL-2B-Instruct', cache_dir=sys.argv[1]))" $cache
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
Write-Host "VLM downloaded. Restart CV-1; in Settings > VLM select Python and base model if not autodetected." -ForegroundColor Green
Write-Host "Custom LoRA adapters are optional and NOT included."
