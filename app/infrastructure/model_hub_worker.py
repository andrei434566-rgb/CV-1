"""Cancellable subprocess for slow or disconnected Windows shared folders."""
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.infrastructure.model_hub import atomic_json, materialize_packet, read_json, rollback_cache, submit_reviewed_packet, sync_models


def main(job_file: str) -> int:
    job = read_json(Path(job_file))
    result_path = Path(job["result"])
    try:
        operation = job["operation"]
        if operation == "sync":
            result = sync_models(Path(job["hub"]), Path(job["cache"]))
        elif operation == "rollback":
            result = {"active": rollback_cache(Path(job["cache"])), "changed": True}
        elif operation == "submit":
            packet = materialize_packet(job["snapshot"], Path(job["outbox"]))
            result = submit_reviewed_packet(Path(job["hub"]), packet)
        else:
            raise ValueError("Неизвестная операция общего хранилища")
        atomic_json(result_path, {"ok": True, **result})
        return 0
    except Exception as exc:
        atomic_json(result_path, {"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
