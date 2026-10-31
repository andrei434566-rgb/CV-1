"""Administrator-only file operations; Windows share ACLs decide write access."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.infrastructure.model_hub import initialize_hub, approve_packet, make_registry_model, publish_model, read_json, atomic_json, identifier, validate_registry


def main():
    parser = argparse.ArgumentParser(description="CV-1: управление общими моделями и принятыми данными")
    parser.add_argument("--hub", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    approve = commands.add_parser("approve"); approve.add_argument("batch")
    dataset = commands.add_parser("datasets"); dataset.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish")
    for name in ("model-id", "role", "version", "entrypoint", "source", "validation-report"):
        publish.add_argument("--" + name, required=True)
    publish.add_argument("--base-model-id", default=""); publish.add_argument("--base-version", default="")
    rollback = commands.add_parser("rollback"); rollback.add_argument("release")
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "init": initialize_hub(args.hub)
    elif args.command == "approve": print(approve_packet(args.hub, args.batch))
    elif args.command == "datasets":
        from app.infrastructure.model_hub_datasets import prepare_datasets
        print(json.dumps(prepare_datasets(args.hub, args.output), ensure_ascii=False, indent=2))
    elif args.command == "publish":
        report = read_json(Path(args.validation_report))
        if report.get("accepted") is not True or not report.get("independent_wells"):
            raise ValueError("Нужен принятый отчёт с независимыми контрольными скважинами")
        initialize_hub(args.hub)
        report_name = "reports/validation/" + identifier(args.model_id) + "-" + identifier(args.version) + ".json"
        if (args.hub / report_name).exists() and read_json(args.hub / report_name) != report:
            raise ValueError("Для этой версии уже сохранён другой отчёт")
        atomic_json(args.hub / report_name, report)
        model = make_registry_model(Path(args.source), model_id=args.model_id, role=args.role, version=args.version,
            entrypoint=args.entrypoint, base_model_id=args.base_model_id, base_version=args.base_version,
            approval={"approved": True, "scope": "Независимые скважины", "validation_report": report_name,
                      "independent_wells": report["independent_wells"]})
        publish_model(args.hub, Path(args.source), model)
    elif args.command == "rollback":
        previous = validate_registry(read_json(args.hub / "registry/history" / (identifier(args.release) + ".json")))
        current = read_json(args.hub / "registry/models.json")
        atomic_json(args.hub / "registry/history" / (identifier(current["release_id"]) + ".json"), current)
        atomic_json(args.hub / "registry/models.json", previous)
    elif args.command == "status":
        path = args.hub / "registry/models.json"
        print(json.dumps(read_json(path), ensure_ascii=False, indent=2) if path.exists() else "Модели ещё не опубликованы")
        print("Входящих пакетов:", len(list((args.hub / "training/incoming").glob("*/receipt.json"))))
        print("Принятых пакетов:", len(list((args.hub / "training/approved").glob("*/approval.json"))))
    return 0


if __name__ == "__main__": raise SystemExit(main())
