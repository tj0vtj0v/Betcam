$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found at .venv. Create it first with: py -3.12 -m venv .venv"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "CUDA-enabled PyTorch and project dependencies installed."
& $venvPython -c "import torch; print(torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
