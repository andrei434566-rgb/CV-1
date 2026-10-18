"""One offline round-trip to the local Qwen runner for troubleshooting."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VLM_ROOT = ROOT / "vlm_core_v1"
PYTHON = VLM_ROOT / ".venv" / "Scripts" / "python.exe"
SNAPSHOTS = VLM_ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
BASE = next(path for path in SNAPSHOTS.iterdir() if (path / "config.json").is_file())
ADAPTER = VLM_ROOT / "outputs" / "core_vlm_v8_single_column_lora"
RUNNER = Path(__file__).resolve().parent / "app" / "infrastructure" / "local_vlm_runner.py"

request = {
    "id": "diagnostic", "operation": "chat", "question": "Какие признаки песчаника видны на керне?",
    "base": str(BASE), "adapter": str(ADAPTER), "theory_adapter": "", "image": "",
    "folders": [], "history": [], "use_knowledge": False,
}
if "--photo" in sys.argv:
    request.update({
        "operation": "photo_chat",
        "question": "Кратко опиши видимые признаки керна на фото. Не придумывай глубины.",
        "image": str(next((VLM_ROOT / "data" / "images").glob("*.jpg"))),
    })
started = time.monotonic()
result = subprocess.run(
    [str(PYTHON), "-u", str(RUNNER)],
    input=json.dumps(request, ensure_ascii=False) + "\n",
    text=True, encoding="utf-8", capture_output=True, timeout=120,
    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
)
line = next((line for line in result.stdout.splitlines() if line.startswith("{")), "")
answer = json.loads(line) if line else {}
print(json.dumps({"seconds": round(time.monotonic() - started, 1), "exit": result.returncode,
                  "ok": answer.get("ok"), "error": answer.get("error"),
                  "answer_excerpt": str(answer.get("answer") or "")[:250],
                  "stderr_tail": result.stderr[-400:]}, ensure_ascii=True))
