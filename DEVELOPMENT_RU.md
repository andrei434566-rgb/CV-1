# Разработка Kern Analyzer в VS Code

Приватный репозиторий CV-1 содержит исходный код приложения в `app/`, точку
запуска `run.py`, зависимости `requirements.txt`, сборку в `.github/workflows/`
и исходники обучения VLM в `vlm_source/`. Обученные веса, базовая Qwen и
локальный справочник распространяются отдельным релизом. Исходные фото и
учебные таблицы в репозиторий не входят.

## Получить проект на другом ПК

Войдите в GitHub под аккаунтом с доступом к приватному CV-1. В VS Code
выберите **Clone Repository** и вставьте
`https://github.com/andrei434566-rgb/CV-1.git`. Откройте полученную папку
в VS Code. То же самое в терминале:

```powershell
git clone https://github.com/andrei434566-rgb/CV-1.git kern_analyzer_app
cd kern_analyzer_app
code .
```

При первом открытии VS Code спросит, разрешить ли автоматические задачи этой
папки. Разрешите задачу **CV-1: получать изменения с GitHub**. Пока папка
открыта, она проверяет `main` каждые 30 секунд и автоматически получает новые
коммиты в файлы VS Code. Если есть локальные правки или ветки разошлись,
автоматическое обновление ждёт и не перезаписывает работу. Состояние видно в
терминале задачи. Для работы нужен доступ к приватному GitHub из Git.

Чтобы получить модели и готовый EXE, запустите
`powershell -ExecutionPolicy Bypass -File .\transfer\install_cv1.ps1`.
Подробности и ручной способ загрузки приватного релиза — в
[`transfer/README_RU.md`](transfer/README_RU.md).

## Менять и запускать код

Для разработки приложения создайте отдельное окружение в папке репозитория:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

В VS Code выберите интерпретатор `.venv\Scripts\python.exe`. Локальная Qwen
использует собственное окружение `..\vlm_core_v1\.venv`, которое создаёт
установщик. Оно не хранится в Git.

Для ускоренной локальной Qwen из уже скачанных весов запустите
`..\vlm_core_v1\.venv\Scripts\python.exe scripts\prepare_qwen_turbo_local.py`.
Порядок подготовки, разделение чата и обработки и измеренная скорость описаны
в [LOCAL_VLM_PERFORMANCE_RU.md](LOCAL_VLM_PERFORMANCE_RU.md).

Код обучения VLM редактируйте в `vlm_source/`. После получения изменений
перенесите этот код в соседнюю рабочую папку модели:

```powershell
.\.venv\Scripts\python.exe transfer\sync_vlm_source.py
```

Скрипт копирует только текстовые исходники; веса и справочник сохраняются.
Не редактируйте файлы напрямую в `..\vlm_core_v1`, если хотите, чтобы правки
попали в GitHub.

## Получать и отправлять изменения

```powershell
git status
git add <изменённые-файлы>
git commit -m "Краткое описание изменения"
git push origin main
```

Новые коммиты с GitHub подтягиваются автоматически, когда VS Code открыт и
рабочая папка чистая. При необходимости одноразового обновления без VS Code
запустите `powershell -ExecutionPolicy Bypass -File .\transfer\watch_github.ps1 -Once`.
После обновления `vlm_source/` повторите `transfer\sync_vlm_source.py`, чтобы
скопировать код в рабочую папку локальной VLM. Большие модели и приложение EXE лежат в
[релизе CV-1](https://github.com/andrei434566-rgb/CV-1/releases/tag/cv1-workspace-v1),
поэтому автообновление получает только исходный код. Чтобы обновить EXE
после изменения приложения, запустите workflow
**Build CV-1 portable application** во вкладке Actions.
