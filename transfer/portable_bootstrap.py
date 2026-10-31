"""Small Windows launcher using the complete offline Python shipped beside it."""
import ctypes
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    root = Path(__file__).resolve().parents[1]
    # The full embedded interpreter, not PyInstaller's pruned stdlib, runs Qt/Torch.
    sys._cv1_python_executable = sys.executable
    sys.frozen = True
    sys.executable = str(root / 'Kern_Analyzer.exe')
    runtime = root / "python_runtime"
    packages = runtime / "Lib/site-packages"
    internal = root / "_internal"
    sys.path[:0] = [str(internal), str(runtime / "python311.zip"), str(runtime), str(packages), str(internal / "excel_photo_model_studio/src")]
    sys.dont_write_bytecode = True
    os.chdir(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["QT_PLUGIN_PATH"] = str(packages / "PySide6/plugins")
    handles = []
    for path in (runtime, packages / "PySide6", packages / "shiboken6", packages / "torch/lib", packages / "cv2"):
        if path.is_dir() and hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(path)))
    sys._cv1_dll_handles = handles
    logs = root / "Kern Analyzer Data/logs"
    logs.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:
        stream = (logs / "application.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
    if sys.stdin is None:
        sys.stdin = open(os.devnull, encoding="utf-8")
    runpy.run_path(str(internal / "run.py"), run_name="__main__")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        message = traceback.format_exc()
        root = Path(__file__).resolve().parents[1]
        try: (root / "launch_error.log").write_text(message, encoding="utf-8")
        except OSError: pass
        if not any(value in sys.argv for value in ("--portable-smoke-test", "--hub-sync-job")):
            ctypes.windll.user32.MessageBoxW(0, "Не удалось запустить CV-1. Проверьте целостность установленного комплекта.\n\n" + message[-1600:], "CV-1", 0x10)
        raise SystemExit(1)
