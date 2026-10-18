param(
    [string]$AssetsDirectory = "",
    [switch]$SkipPythonInstall
)

$ErrorActionPreference = 'Stop'
$Repo = 'andrei434566-rgb/CV-1'
$ReleaseTag = 'cv1-workspace-v1'
$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Workspace = Split-Path -Parent $AppRoot
$VlmRoot = Join-Path $Workspace 'vlm_core_v1'
if (-not $AssetsDirectory) { $AssetsDirectory = Join-Path $PSScriptRoot 'downloads' }
New-Item -ItemType Directory -Path $AssetsDirectory -Force | Out-Null
$AssetsDirectory = (Resolve-Path $AssetsDirectory).Path

function Get-GitHubCredential {
    $credentialText = "protocol=https`nhost=github.com`n`n" | git credential fill 2>$null
    if ($LASTEXITCODE -ne 0) { return '' }
    foreach ($line in $credentialText) {
        if ($line -like 'password=*') { return $line.Substring(9) }
    }
    return ''
}

function Get-ReleaseAsset {
    param($Asset, [string]$Destination, [hashtable]$Headers)
    if (Test-Path -LiteralPath $Destination) { return }
    Write-Host "Скачиваю $($Asset.name) ..."
    $downloadHeaders = $Headers.Clone()
    $downloadHeaders['Accept'] = 'application/octet-stream'
    Invoke-WebRequest -Uri $Asset.url -Headers $downloadHeaders -OutFile $Destination -UseBasicParsing
}

$manifestPath = Join-Path $AssetsDirectory 'cv1-runtime-manifest.json'
$appZip = Join-Path $AssetsDirectory 'Kern_Analyzer-Windows.zip'
$needDownload = -not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $appZip)
if (-not $needDownload) {
    $existingManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $required = @($existingManifest.runtime_archive.name) + @($existingManifest.parts | ForEach-Object { $_.name })
    $needDownload = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $AssetsDirectory $_)) }).Count -gt 0
}
if ($needDownload) {
    $token = Get-GitHubCredential
    if (-not $token) {
        throw "Войдите в GitHub для git или скачайте файлы релиза $ReleaseTag вручную в $AssetsDirectory"
    }
    $headers = @{ Authorization = "Bearer $token"; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28'; 'User-Agent' = 'CV-1-installer' }
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$ReleaseTag" -Headers $headers
    $assetNames = @('cv1-runtime-manifest.json', 'cv1-runtime.zip', 'Kern_Analyzer-Windows.zip')
    $assetNames += @($release.assets | Where-Object { $_.name -like 'cv1-qwen3vl-2b-model.part*' } | ForEach-Object { $_.name })
    foreach ($name in $assetNames) {
        $asset = $release.assets | Where-Object { $_.name -eq $name } | Select-Object -First 1
        if (-not $asset) { throw "В релизе отсутствует $name" }
        Get-ReleaseAsset -Asset $asset -Destination (Join-Path $AssetsDirectory $name) -Headers $headers
    }
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 'cv1_runtime_v1') { throw 'Неизвестный формат пакета модели.' }
if (-not $manifest.app_archive) { throw 'В пакете отсутствует описание проверенного EXE.' }
function Confirm-Checksum {
    param([string]$Name, [string]$Expected, [long]$ExpectedBytes)
    $path = Join-Path $AssetsDirectory $Name
    if (-not (Test-Path -LiteralPath $path)) { throw "Не найден $Name" }
    if ((Get-Item -LiteralPath $path).Length -ne $ExpectedBytes) { throw "Неверный размер $Name" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) { throw "Неверная контрольная сумма $Name" }
}
Confirm-Checksum -Name $manifest.runtime_archive.name -Expected $manifest.runtime_archive.sha256 -ExpectedBytes $manifest.runtime_archive.bytes
Confirm-Checksum -Name $manifest.app_archive.name -Expected $manifest.app_archive.sha256 -ExpectedBytes $manifest.app_archive.bytes
foreach ($part in $manifest.parts) {
    Confirm-Checksum -Name $part.name -Expected $part.sha256 -ExpectedBytes $part.bytes
}

Write-Host 'Распаковываю локальные модели и справочник...'
Expand-Archive -LiteralPath (Join-Path $AssetsDirectory $manifest.runtime_archive.name) -DestinationPath $Workspace -Force
if ((Split-Path -Leaf $AppRoot) -ne 'kern_analyzer_app') {
    $sourceModels = Join-Path $Workspace 'kern_analyzer_app\models'
    if (Test-Path -LiteralPath $sourceModels) {
        New-Item -ItemType Directory -Path (Join-Path $AppRoot 'models') -Force | Out-Null
        Copy-Item -Path (Join-Path $sourceModels '*') -Destination (Join-Path $AppRoot 'models') -Recurse -Force
    }
}

$modelPath = Join-Path $Workspace ($manifest.model_relative_path -replace '/', '\')
New-Item -ItemType Directory -Path (Split-Path -Parent $modelPath) -Force | Out-Null
$modelReady = $false
if (Test-Path -LiteralPath $modelPath) {
    $modelReady = ((Get-Item -LiteralPath $modelPath).Length -eq $manifest.model_bytes) -and ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $manifest.model_sha256.ToLowerInvariant())
    if (-not $modelReady) { throw "В $modelPath уже есть другой файл модели. Сохраните его отдельно и запустите установку снова." }
}
if (-not $modelReady) {
    $partialPath = "$modelPath.partial"
    $output = [System.IO.File]::Create($partialPath)
    try {
        foreach ($part in $manifest.parts) {
            $input = [System.IO.File]::OpenRead((Join-Path $AssetsDirectory $part.name))
            try { $input.CopyTo($output, 8MB) } finally { $input.Dispose() }
        }
    } finally { $output.Dispose() }
    if ((Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.model_sha256.ToLowerInvariant()) {
        throw 'Собранная модель не прошла проверку SHA256.'
    }
    Move-Item -LiteralPath $partialPath -Destination $modelPath
}

Write-Host 'Распаковываю приложение...'
Expand-Archive -LiteralPath $appZip -DestinationPath $AppRoot -Force
if (-not (Test-Path -LiteralPath (Join-Path $AppRoot 'Kern_Analyzer.exe'))) {
    throw 'В архиве отсутствует Kern_Analyzer.exe.'
}

if (-not $SkipPythonInstall) {
    $uv = Join-Path $VlmRoot 'portable_trainer\vendor\uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) { throw 'Не найден uv.exe для установки Python VLM.' }
    Write-Host 'Устанавливаю Python и библиотеки локальной VLM; потребуется интернет и несколько ГБ места...'
    & $uv python install 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось скачать Python 3.11.' }
    & $uv venv (Join-Path $VlmRoot '.venv') --python 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать Python-окружение.' }
    $vlmPython = Join-Path $VlmRoot '.venv\Scripts\python.exe'
    & $uv pip install --python $vlmPython -r (Join-Path $VlmRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости локальной VLM.' }
}

Write-Host "Готово. Запустите $(Join-Path $AppRoot 'Kern_Analyzer.exe')"
