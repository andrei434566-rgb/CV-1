"""Configuring an unavailable share must not block the GUI or contact it."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMainWindow

from app.infrastructure.model_hub import atomic_json, initialize_hub, probe_hub
from app.infrastructure.model_hub_client import ModelHubClient
from app.infrastructure.network_settings import network_enabled, normalize_hub_path
from app.ui.dialogs.model_hub_dialog import ModelHubDialog

APP = QApplication.instance() or QApplication([])


class NetworkFolderSetupTests(unittest.TestCase):
    def test_unavailable_unc_and_mapped_paths_are_validated_without_filesystem_access(self):
        with patch.object(Path, 'exists', side_effect=AssertionError('Network access')), patch.object(Path, 'resolve', side_effect=AssertionError('Network access')):
            self.assertEqual(normalize_hub_path('  "\\\\OFFLINE\\Геология\\CV1"  '), r'\\OFFLINE\Геология\CV1')
            self.assertEqual(normalize_hub_path('Z:/CV1'), 'Z:\\CV1')
            self.assertEqual(normalize_hub_path(''), '')
        for path in ('CV1', 'Z:CV1', r'\\server', 'https://server/CV1', 'Z:\\bad?name'):
            with self.subTest(path=path), self.assertRaises(ValueError): normalize_hub_path(path)

    def test_save_offline_path_and_disconnect_keep_the_address_without_running_worker(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); parent = QMainWindow(); parent._settings = QSettings(str(root/'private.ini'), QSettings.IniFormat)
            parent._hub_client = ModelHubClient(parent); parent._network_action = Mock()
            with patch('app.ui.dialogs.model_hub_dialog.user_data_root', return_value=root):
                dialog = ModelHubDialog(parent)
                try:
                    dialog.path.setText(r'\\OFFLINE\Общая папка\CV1')
                    parent._settings.setValue('network/last_submitted_signature', 'old-server')
                    with patch.object(Path, 'exists', side_effect=AssertionError('Saving must not contact server')):
                        self.assertTrue(dialog._persist())
                    self.assertEqual(parent._settings.value('network/hub'), r'\\OFFLINE\Общая папка\CV1')
                    self.assertFalse(parent._settings.contains('network/last_submitted_signature'))
                    parent._network_action.assert_not_called()
                    dialog.enabled.setChecked(False); self.assertTrue(dialog._persist())
                    self.assertFalse(network_enabled(parent._settings))
                    self.assertTrue(parent._settings.value('network/hub'))
                    dialog._run('sync'); parent._network_action.assert_not_called()
                    dialog._run('probe'); parent._network_action.assert_called_once_with('probe')
                finally: dialog.close(); parent.close()

    def test_disabled_network_does_not_attempt_startup_or_submit(self):
        from app.ui.windows.main_window import MainWindow
        with TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder)/'private.ini'), QSettings.IniFormat)
            settings.setValue('network/hub', r'\\OFFLINE\CV1'); settings.setValue('network/enabled', False)
            fake = SimpleNamespace(_settings=settings, _hub_closing=False, _network_action=Mock(), statusBar=lambda:Mock())
            MainWindow._network_startup(fake); fake._network_action.assert_not_called()
            MainWindow._network_action(fake,'submit')  # returns before touching any worker or snapshot

    def test_read_only_probe_and_explicit_initialize_do_not_publish_or_overwrite_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); original = root/'my_document.txt'; original.write_text('keep')
            before = set(root.rglob('*')); self.assertFalse(probe_hub(root)['prepared'])
            self.assertEqual(set(root.rglob('*')), before)
            initialize_hub(root); first = (root/'hub.json').read_bytes(); initialize_hub(root)
            self.assertEqual((root/'hub.json').read_bytes(),first)
            self.assertTrue(probe_hub(root)['prepared']); self.assertFalse(probe_hub(root)['models_published'])
            self.assertTrue((root/'training/incoming').is_dir())
            self.assertEqual(original.read_text(),'keep')
            self.assertFalse((root/'registry/models.json').exists())

    def test_wrong_marker_does_not_get_replaced_or_create_exchange_folders(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); atomic_json(root/'hub.json',{'schema':'other-application'})
            before = (root/'hub.json').read_bytes()
            with self.assertRaises(ValueError): initialize_hub(root)
            self.assertEqual((root/'hub.json').read_bytes(), before)
            self.assertFalse((root/'training').exists())


if __name__ == '__main__': unittest.main()
