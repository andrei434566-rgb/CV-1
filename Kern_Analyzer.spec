# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from pathlib import Path

datas = [
    ('MODEL_GUIDE.md', '.'),
    ('CV1_ARCHITECTURE_RU.md', '.'),
    ('CV1_VISION_RU.md', '.'),
    ('README_OPEN_DATASETS_RU.md', '.'),
    ('requirements.txt', '.'),
    ('README_EXCEL_JPG_IMPORT.md', '.'),
    ('README_WORD_PHOTO_TRAINING_RU.md', '.'),
    ('README_PRIMARY_SOURCE_RU.md', '.'),
    ('docs/USER_GUIDE_RU.md', 'docs'),
    ('PORTABLE_README.txt', '.'),
    ('model_blueprint', 'model_blueprint'),
    # Offline OCR used by the photo-interval agent.  Ship it beside the
    # executable so a target PC does not need a separate Tesseract install.
    ('tools/tesseract', 'tools/tesseract'),
]
binaries = []
hiddenimports = [
    'app.ui.widgets.workspace_canvas',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    # run.py imports this only when a BAT invokes the no-dialog pipeline.
    'build_core_tape',
    'excel_photo_model_studio.description_model',
]
binaries += collect_dynamic_libs('shiboken6')
hiddenimports += collect_submodules('app')
# External Python training tools need the source tree, but caches and weights
# must enter the distribution only through the explicit model allowlist below.
for source in Path('app').rglob('*'):
    if source.is_file() and '__pycache__' not in source.parts and source.suffix.lower() in {
        '.py', '.json', '.md', '.txt', '.xlsx', '.png', '.jpg', '.jpeg', '.svg', '.ico', '.qss', '.ui'
    }:
        datas.append((str(source), source.parent.as_posix()))

# Ship the selected classifier only, never training runs or pretrained caches.
vision_root = Path('models/vision')
if (vision_root / 'current.json').is_file():
    import json
    folder = json.loads((vision_root / 'current.json').read_text(encoding='utf-8'))['folder']
    selected = (vision_root / folder).resolve()
    if not selected.is_relative_to(vision_root.resolve()):
        raise ValueError('Vision model path is outside models/vision')
    # The model folder may also contain research datasets and cached features.
    # They are neither runtime dependencies nor public application resources.
    datas += [(str(vision_root / 'current.json'), 'models/vision')]
    for name in ('config.json', 'model.pt'):
        datas.append((str(selected / name), 'models/vision/' + folder))
# The PySide hooks collect the imported Qt modules and their plugins.
# Collecting all of PySide6 also pulls in unrelated Qt modules and tools.
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# PyInstaller's official hooks already collect the required torch binaries.
# collect_all(torch/torchvision) duplicated several gigabytes in old releases
# and pushed the ZIP over GitHub's 2 GiB asset limit.
hiddenimports += collect_submodules('torchvision')
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openpyxl')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('docx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pytesseract')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pypdf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('xlrd')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['run.py'],
    pathex=['.', 'excel_photo_model_studio/src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['app/qt_runtime_hook.py'],
    excludes=['transformers', 'peft', 'accelerate', 'llama_cpp', 'bitsandbytes'],
    noarchive=False,
    optimize=0,
)
# PyInstaller can accidentally pick Poppler ICU DLLs from the build machine
# PATH.  Qt on supported Windows uses the system ICU runtime; the bundled
# Poppler copies make QtWidgets fail before the interface can start.
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name.casefold() not in {"icuuc.dll", "icudt78.dll"}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Kern_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Kern_Analyzer',
)
