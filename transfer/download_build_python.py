"""Fetch the pinned official CPython portable runtime for the offline installer."""
from pathlib import Path
import requests
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1] / "outputs/build_cv1_210"
ROOT.mkdir(parents=True, exist_ok=True)
archive = ROOT / "python-3.11.9-embed-amd64.zip"
if not archive.is_file():
    response = requests.get("https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip", timeout=120)
    response.raise_for_status()
    archive.write_bytes(response.content)
target = ROOT / "python_runtime"
target.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as package:
    if package.testzip() is not None: raise ValueError("Повреждён официальный архив CPython")
    package.extractall(target)
(target / "python311._pth").write_text("python311.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
print("Official offline Python ready:", target)

ocr=ROOT/'ocr/rus.traineddata'
ocr.parent.mkdir(exist_ok=True)
url='https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/rus.traineddata'
if not ocr.is_file():
    response=requests.get(url,timeout=120);response.raise_for_status();ocr.write_bytes(response.content)
if hashlib.sha256(ocr.read_bytes()).hexdigest()!='e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225':
    raise ValueError('Russian OCR checksum mismatch')
print('Pinned Russian OCR ready')
