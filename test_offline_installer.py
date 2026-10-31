"""The actual installer format works across boundaries and rejects corrupt media."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import runpy
import sys

from transfer.build_offline_release import VolumeWriter
from transfer.offline_installer import Volumes, install, safe_target


class OfflineInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def media(self, kind="client"):
        files = {"Kern_Analyzer.exe": b"example app", "bundled_models/weights.bin": bytes(range(256)) * 20}
        writer = VolumeWriter(self.root, "data", limit=300)
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items(): archive.writestr(name, content)
        control = {"version":"test", "kind":kind, "parts":writer.finish(), "files":[
            {"path":name,"size":len(content),"sha256":hashlib.sha256(content).hexdigest()} for name,content in files.items()],
            "unpacked_bytes":sum(map(len,files.values()))}
        return control, files

    def test_complete_installer_roundtrip_and_resume_preserve_user_files(self):
        control, files = self.media()
        self.assertGreater(len(control["parts"]), 1)
        target = self.root / "installed"; target.mkdir()
        (target / "my_project.json").write_text("important")
        install(control, self.root, target)
        install(control, self.root, target)
        for name, data in files.items(): self.assertEqual((target / name).read_bytes(), data)
        self.assertEqual((target / "my_project.json").read_text(), "important")

    def test_missing_or_corrupt_part_is_rejected_before_installation(self):
        control, _ = self.media()
        (self.root / control["parts"][0]["name"]).write_bytes(b"wrong")
        with self.assertRaises(ValueError): install(control, self.root, self.root / "target")
        self.assertFalse((self.root / "target").exists())

    def test_virtual_reader_cross_part_seeks_are_exact(self):
        control, _ = self.media()
        paths = [self.root / item["name"] for item in control["parts"]]
        expected = b"".join(path.read_bytes() for path in paths)
        with Volumes(paths) as reader:
            reader.seek(290); self.assertEqual(reader.read(100), expected[290:390])
            reader.seek(-15, 2); self.assertEqual(reader.read(), expected[-15:])
            reader.seek(0); self.assertEqual(reader.read(25), expected[:25])

    def test_client_installs_dll_search_setup_for_separate_python_workers(self):
        control,_=self.media()
        target=self.root/'installed'
        runtime=target/'python_runtime'
        (runtime/'Lib/site-packages/PySide6').mkdir(parents=True)
        (runtime/'python311._pth').write_text('import site\n')
        install(control,self.root,target)
        startup=runtime/'sitecustomize.py'
        self.assertTrue(startup.is_file())
        previous=getattr(sys,'_cv1_runtime_dll_handles',None)
        try:
            loaded=runpy.run_path(str(startup))
            self.assertEqual(loaded['_runtime'],runtime.resolve())
            if hasattr(__import__('os'),'add_dll_directory'):
                self.assertGreaterEqual(len(sys._cv1_runtime_dll_handles),2)
        finally:
            for handle in getattr(sys,'_cv1_runtime_dll_handles',[]):handle.close()
            if previous is None:del sys._cv1_runtime_dll_handles
            else:sys._cv1_runtime_dll_handles=previous

    def test_zip_paths_cannot_escape_chosen_directory(self):
        for name in ("../outside", "/absolute", "C:/system", "a\\b", "a:stream"):
            with self.subTest(name=name), self.assertRaises(ValueError): safe_target(self.root, name)


if __name__ == "__main__": unittest.main()
