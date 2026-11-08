"""Run desktop tests with private default settings instead of user preferences."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6 import QtCore


def main():
    native=QtCore.QSettings
    original=native(native.NativeFormat,native.UserScope,'Kern Analyzer','KernAnalyzer')
    before={key:original.value(key) for key in original.allKeys()}
    with TemporaryDirectory(prefix='cv1_desktop_checks_') as folder:
        class Settings(native):
            def __init__(self,*args,**kwargs):
                if args==('Kern Analyzer','KernAnalyzer'):
                    super().__init__(str(Path(folder)/'private.ini'),native.IniFormat)
                else:super().__init__(*args,**kwargs)
        QtCore.QSettings=Settings
        try:
            suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
            result=unittest.TextTestRunner(verbosity=1).run(suite)
            from app.infrastructure.attribute_tasks import wait_for_tasks
            from PySide6.QtWidgets import QApplication
            wait_for_tasks()
            if QApplication.instance():QApplication.instance().processEvents()
            original.sync()
            unchanged=before=={key:original.value(key) for key in original.allKeys()}
            print(f'native_settings_unchanged={unchanged}',flush=True)
            return 0 if result.wasSuccessful() and unchanged else 1
        finally:QtCore.QSettings=native


if __name__=='__main__':raise SystemExit(main())
