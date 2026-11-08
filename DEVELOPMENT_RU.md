# Разработка CV-1 Kern Analyzer

1. Откройте репозиторий в VS Code и создайте окружение Python 3.11.
2. Установите пакеты через `python -m pip install -r requirements.txt`.
3. Запустите `python launch_latest.py`.
4. Для распознавания пород установите модель из актуального релиза в `models/vision`.
5. Укажите собственные веса YOLO через меню модели сегментации.

Для проверок используйте `python -m unittest discover -q` с `QT_QPA_PLATFORM=offscreen`.
Для переносимой Windows-сборки используйте `python -m PyInstaller Kern_Analyzer.spec`.
Собранное приложение проверяется запуском `Kern_Analyzer.exe --portable-smoke-test`.
Порядок подготовки OCR и архивов показан в рабочих процессах GitHub Actions.
