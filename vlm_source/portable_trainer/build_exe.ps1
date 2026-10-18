$ErrorActionPreference = "Stop"

$packageDir = $PSScriptRoot
$vendorDir = Join-Path $packageDir "vendor"
$distDir = Join-Path $packageDir "dist"
$buildDir = Join-Path $packageDir "build"
$uvVersion = "0.12.20"
$uvArchiveName = "uv-x86_64-pc-windows-msvc.zip"
$uvUrl = "https://releases.astral.sh/github/uv/releases/download/$uvVersion/$uvArchiveName"
$uvExpectedSha256 = "95f9bc30fbb3574d276e28ac4a6de932d25153645853d13da8c21eec3bc88d06"
$uvExpectedExeSha256 = "a0d2742d49564a32488753b02e76276e7b5ef1b1ea8cf30bcbf06ee28f60cd73"
$archivePath = Join-Path $vendorDir $uvArchiveName
$uvPath = Join-Path $vendorDir "uv.exe"
$guiPath = Join-Path $packageDir "core_trainer_gui.py"
$trainerPath = Join-Path $packageDir "train_qwen3vl.py"

if (-not (Test-Path $guiPath)) { throw "Не найден исходник GUI: $guiPath" }
if (-not (Test-Path $trainerPath)) { throw "Не найден скрипт обучения: $trainerPath" }
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    throw "PyInstaller не найден. Установите его в Python сборочного компьютера: python -m pip install pyinstaller"
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
if (-not (Test-Path $uvPath)) {
    Write-Host "Загружаю официальный uv $uvVersion для Windows x64..."
    Invoke-WebRequest -Uri $uvUrl -OutFile $archivePath
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $uvExpectedSha256) {
        throw "SHA-256 скачанного архива uv не совпал с опубликованной суммой. Сборку остановил."
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath (Join-Path $vendorDir "uv_extract") -Force
    $foundUv = Get-ChildItem -LiteralPath (Join-Path $vendorDir "uv_extract") -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $foundUv) { throw "В официальном архиве не найден uv.exe" }
    Copy-Item -LiteralPath $foundUv.FullName -Destination $uvPath -Force
}

$uvHash = (Get-FileHash -LiteralPath $uvPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($uvHash -ne $uvExpectedExeSha256) { throw "SHA-256 uv.exe не совпал с проверенной официальной сборкой: $uvHash" }
if ((Get-AuthenticodeSignature -LiteralPath $uvPath).Status -ne "Valid") { throw "Подпись uv.exe не прошла проверку Authenticode." }
Write-Host "uv.exe SHA-256: $uvHash"

$pyinstaller = (Get-Command pyinstaller).Source
& $pyinstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "CoreSedTrainer" `
    --paths $packageDir `
    --add-data "$trainerPath;." `
    --add-binary "$uvPath;." `
    --collect-all openpyxl `
    --hidden-import openpyxl.cell._writer `
    --workpath $buildDir `
    --distpath $distDir `
    --specpath $buildDir `
    $guiPath

if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с кодом $LASTEXITCODE" }
$exePath = Join-Path $distDir "CoreSedTrainer.exe"
if (-not (Test-Path $exePath)) { throw "PyInstaller не создал ожидаемый файл: $exePath" }
$exeHash = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
Copy-Item -LiteralPath (Join-Path $packageDir "README_RU.md") -Destination $distDir -Force
Copy-Item -LiteralPath (Join-Path $packageDir "THIRD_PARTY_NOTICES.md") -Destination $distDir -Force
Copy-Item -LiteralPath (Join-Path $packageDir "LICENSE-MIT-UV.txt") -Destination $distDir -Force
Write-Host "Сборка готова: $exePath"
Write-Host "Размер: $([math]::Round((Get-Item -LiteralPath $exePath).Length / 1MB, 1)) МБ"
Write-Host "SHA-256: $exeHash"
