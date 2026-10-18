"""Exercise the same Qt transport used by the assistant panel."""

import json
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

from app.infrastructure.local_vlm_client import LocalVlmClient


ROOT = Path(__file__).resolve().parent.parent / "vlm_core_v1"
BASE = next(path for path in (ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots").iterdir() if (path / "config.json").is_file())
ADAPTER = ROOT / "outputs" / "core_vlm_v8_single_column_lora"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

application = QCoreApplication([])
client = LocalVlmClient()
started = time.monotonic()


def received(result):
    print(json.dumps({"seconds": round(time.monotonic() - started, 1), "ok": result.get("ok"),
                      "error": result.get("error"), "answer_chars": len(result.get("answer") or "")}, ensure_ascii=True))
    client.stop()
    application.quit()


client.reply.connect(received)
client.send(str(PYTHON), {
    "operation": "chat", "question": "Какие признаки песчаника видны на керне?",
    "base": str(BASE), "adapter": str(ADAPTER), "theory_adapter": "",
    "image": "", "folders": [], "history": [], "use_knowledge": False,
})
QTimer.singleShot(120000, application.quit)
application.exec()
