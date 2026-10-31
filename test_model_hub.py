"""Real file transport: atomic versions, offline fallback and expert-only datasets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from app.infrastructure.model_hub import (
    approve_packet, atomic_json, digest, initialize_hub, make_registry_model, materialize_packet,
    publish_model, read_json, relative, rollback_cache, submit_reviewed_packet, sync_models, validate_registry,
)
from app.infrastructure.model_hub_datasets import prepare_datasets


class ModelHubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hub, self.cache = self.root / "hub", self.root / "cache"
        initialize_hub(self.hub)

    def publish(self, version="v1", data=b"model", role="geometry"):
        source = self.root / (role + version)
        source.mkdir()
        (source / "best.pt").write_bytes(data)
        model = make_registry_model(source, model_id=role, role=role, version=version,
            entrypoint="best.pt", approval={"approved": True, "scope": "test fixture"})
        publish_model(self.hub, source, model, baseline=True)
        return model

    def snapshot(self, *, well="A", label="F1", text_approved=True, project_key="p1"):
        photo = self.root / (well + ".png")
        if not photo.exists(): Image.new("RGB", (100, 100), "gray" if well == "A" else "tan").save(photo)
        return {"computer": "PC1", "project": "Test", "project_key": project_key, "photos": [{
            "source": str(photo), "canvas_size": [100, 100], "annotations": [{"label": label, "well": well,
            "depth_from": 1, "depth_to": 2, "geometry_approved": True, "text_approved": text_approved,
            "polygon": [[10,10],[80,10],[80,90],[10,90]], "attributes": {"__vlm_description": "Серый слоистый керн."}}]}]}

    def accept(self, snapshot):
        packet = materialize_packet(snapshot, self.root / "outbox")
        receipt = submit_reviewed_packet(self.hub, packet)
        approve_packet(self.hub, receipt["batch_id"])
        return packet, receipt

    def test_copies_locally_and_keeps_active_state_when_server_disappears(self):
        self.publish()
        first = sync_models(self.hub, self.cache)
        self.assertTrue(first["changed"])
        entry = Path(first["active"]["models"][0]["local_entrypoint"])
        self.assertEqual(entry.read_bytes(), b"model")
        self.assertFalse(sync_models(self.hub, self.cache)["changed"])
        saved = (self.cache / "active.json").read_bytes()
        with self.assertRaises(OSError): sync_models(self.root / "offline", self.cache)
        self.assertEqual((self.cache / "active.json").read_bytes(), saved)
        self.assertEqual(entry.read_bytes(), b"model")

    def test_failed_new_checksum_never_replaces_old_active_model(self):
        self.publish(); sync_models(self.hub, self.cache)
        before = (self.cache / "active.json").read_bytes()
        self.publish("v2", b"new model")
        (self.hub / "models/geometry/v2/best.pt").write_bytes(b"broken")
        with self.assertRaises(ValueError): sync_models(self.hub, self.cache)
        self.assertEqual((self.cache / "active.json").read_bytes(), before)

    def test_rollback_keeps_both_immutable_versions(self):
        self.publish(); sync_models(self.hub, self.cache)
        self.publish("v2", b"new"); sync_models(self.hub, self.cache)
        previous = rollback_cache(self.cache)
        self.assertEqual(previous["models"][0]["version"], "v1")
        self.assertTrue((self.cache / "models/geometry/v2/best.pt").exists())

    def test_existing_version_and_unaccepted_candidate_cannot_be_published(self):
        self.publish()
        source = self.root / "changed"; source.mkdir(); (source / "best.pt").write_bytes(b"changed")
        model = make_registry_model(source, model_id="geometry", role="geometry", version="v1", entrypoint="best.pt", approval={"approved":True})
        before = (self.hub / "registry/models.json").read_bytes()
        with self.assertRaises(ValueError): publish_model(self.hub, source, model, baseline=True)
        with self.assertRaises(ValueError): publish_model(self.hub, source, model)
        self.assertEqual(before, (self.hub / "registry/models.json").read_bytes())

    def test_registry_rejects_traversal_duplicate_roles_and_incompatible_lora(self):
        model = self.publish()
        registry = read_json(self.hub / "registry/models.json")
        for value in ("../outside", "C:/secret", "/root", "file:stream", "a\\b"):
            with self.subTest(value=value), self.assertRaises(ValueError): relative(value)
        invalid = deepcopy(registry); invalid["models"].append(deepcopy(model))
        with self.assertRaises(ValueError): validate_registry(invalid)
        invalid = deepcopy(registry); adapter = deepcopy(model); adapter.update(id="adapter", role="vlm_adapter")
        invalid["models"].append(adapter)
        with self.assertRaises(ValueError): validate_registry(invalid)

    def test_packet_deduplication_and_predictions_are_not_accepted(self):
        packet = materialize_packet(self.snapshot(), self.root / "outbox")
        first = submit_reviewed_packet(self.hub, packet)
        second = submit_reviewed_packet(self.hub, packet)
        self.assertFalse(first["duplicate"]); self.assertTrue(second["duplicate"])
        bad = self.snapshot(); bad["photos"][0]["annotations"][0]["geometry_approved"] = False
        with self.assertRaises(ValueError): materialize_packet(bad, self.root / "outbox")

    def test_only_centrally_approved_text_enters_vlm_with_well_separation(self):
        unapproved = materialize_packet(self.snapshot(), self.root / "outbox")
        submit_reviewed_packet(self.hub, unapproved)
        with self.assertRaises(ValueError): prepare_datasets(self.hub, self.root / "not-ready")
        self.accept(self.snapshot())
        self.accept(self.snapshot(well="B", text_approved=False, project_key="p2"))
        report = prepare_datasets(self.hub, self.root / "dataset")
        self.assertTrue(report["independent_validation_available"])
        self.assertEqual(report["vlm_train"] + report["vlm_val"], 1)
        train = {row["group"] for row in report["samples"] if row["split"] == "train"}
        val = {row["group"] for row in report["samples"] if row["split"] == "val"}
        self.assertFalse(train & val)

    def test_new_accepted_project_snapshot_supersedes_old_wrong_answer(self):
        _, old = self.accept(self.snapshot(label="old"))
        self.accept(self.snapshot(label="corrected"))
        self.assertFalse(read_json(self.hub / "training/approved" / old["batch_id"] / "approval.json")["central_approved"])
        report = prepare_datasets(self.hub, self.root / "corrected-data")
        self.assertEqual([row["label"] for row in report["samples"]], ["corrected"])

    def test_conflicting_independent_expert_labels_block_training(self):
        self.accept(self.snapshot(label="F1"))
        self.accept(self.snapshot(label="F2", project_key="other-project"))
        with self.assertRaisesRegex(ValueError, "разные подтверждённые фации"):
            prepare_datasets(self.hub, self.root / "conflict")


if __name__ == "__main__": unittest.main()
