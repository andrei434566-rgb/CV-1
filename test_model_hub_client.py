"""The portable GUI must own its actual cancellable network worker."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
from app.infrastructure.model_hub_client import ModelHubClient

APP=QApplication.instance() or QApplication([])

class ModelHubClientTests(unittest.TestCase):
    def test_portable_client_starts_worker_python_directly(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            process=Mock()
            process.state.return_value=QProcess.ProcessState.NotRunning
            process.waitForStarted.return_value=True
            with patch('app.infrastructure.model_hub_client.QProcess',return_value=process) as constructor, \
                 patch('app.infrastructure.model_hub_client.user_data_root',return_value=root), \
                 patch('app.infrastructure.model_hub_client.bundled_root',return_value=root/'_internal'), \
                 patch('app.infrastructure.model_hub_client.bundled_runtime_defaults',return_value={'python':str(root/'python.exe')}):
                constructor.ProcessState=QProcess.ProcessState
                client=ModelHubClient()
                self.assertTrue(client.start('sync',str(root/'hub')))
                executable,arguments=process.start.call_args.args
                self.assertEqual(executable,str(root/'python.exe'))
                self.assertEqual(arguments[0],'-B')
                self.assertTrue(arguments[1].endswith('model_hub_worker.py'))
                self.assertTrue(Path(arguments[2]).is_file())
                client.timer.stop()

if __name__=='__main__':unittest.main()
