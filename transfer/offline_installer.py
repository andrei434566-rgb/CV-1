"""Offline installer for a multi-volume ZIP with embedded SHA256 manifest."""
from bisect import bisect_right
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


class Volumes(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.offsets, self.position, total = list(paths), [], 0, 0
        for path in paths:
            self.offsets.append(total); total += Path(path).stat().st_size
        self.length = total
        self.handle, self.part = None, -1
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if position < 0: raise ValueError("Negative seek")
        self.position = position; return position
    def read(self, size=-1):
        size = min(self.length - self.position, size if size >= 0 else self.length)
        result = []
        while size > 0:
            index = bisect_right(self.offsets, self.position) - 1
            if index != self.part:
                if self.handle: self.handle.close()
                self.handle = Path(self.paths[index]).open("rb"); self.part = index
            self.handle.seek(self.position - self.offsets[index])
            block = self.handle.read(size)
            if not block: raise OSError("Неполный файл данных установки")
            result.append(block); self.position += len(block); size -= len(block)
        return b"".join(result)
    def close(self):
        if self.handle: self.handle.close()
        super().close()


def safe_target(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
        raise ValueError("Небезопасный путь в установочном комплекте")
    path = Path(root).joinpath(*pure.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()): raise ValueError("Путь вне папки установки")
    return path


def install(control, media, target=None, progress=lambda *args: None, cancelled=lambda: False):
    paths = []
    for part in control["parts"]:
        path = media / part["name"]
        progress("Проверка " + part["name"], 0)
        if not path.is_file() or path.stat().st_size != part["size"] or sha256(path) != part["sha256"]:
            raise ValueError("Отсутствует или повреждён " + part["name"] + ". Скачайте все части из одного релиза.")
        paths.append(path)
    if target is None: return
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if control["kind"] == "server" and (target / "hub.json").exists():
        if json.loads((target / "hub.json").read_text(encoding="utf-8")).get("schema") != "cv1-network-hub-v1":
            raise ValueError("В папке находится другое хранилище. Выберите отдельную папку CV-1")
    missing, largest_replacement = 0, 0
    for item in control["files"]:
        path = safe_target(target, item["path"])
        if path.is_file(): largest_replacement = max(largest_replacement, item["size"])
        else: missing += item["size"]
    if shutil.disk_usage(target).free < missing + largest_replacement + 256 * 1024 * 1024:
        raise OSError("Недостаточно места в выбранной папке установки")
    expected = {item["path"]: item for item in control["files"]}
    completed = 0
    with Volumes(paths) as combined, zipfile.ZipFile(combined) as archive:
        if {entry.filename for entry in archive.infolist()} != set(expected):
            raise ValueError("Состав архива не совпадает с паспортом выпуска")
        for entry in archive.infolist():
            if cancelled(): raise InterruptedError("Установка остановлена. Для продолжения запустите её повторно.")
            destination = safe_target(target, entry.filename)
            item = expected[entry.filename]
            if control["kind"] == "server" and entry.filename in {"hub.json", "registry/models.json"} and destination.exists():
                completed += item["size"]; continue
            if destination.is_file() and destination.stat().st_size == item["size"] and sha256(destination) == item["sha256"]:
                completed += item["size"]; continue
            if control["kind"] == "server" and entry.filename.startswith("models/") and destination.exists():
                raise ValueError("Существующая версия модели отличается. Опубликованные модели не перезаписываются")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".cv1-installing")
            digest = hashlib.sha256()
            with archive.open(entry) as source, temporary.open("wb") as output:
                while block := source.read(4 * 1024 * 1024):
                    if cancelled(): raise InterruptedError("Установка остановлена. Повторный запуск продолжит её.")
                    output.write(block); digest.update(block); completed += len(block)
                    progress(entry.filename, min(100, int(completed * 100 / control["unpacked_bytes"])))
            if digest.hexdigest() != item["sha256"]: raise ValueError("Неверная контрольная сумма " + entry.filename)
            os.replace(temporary, destination)
    (target / ".cv1_install.json").write_text(json.dumps({"version":control["version"],"kind":control["kind"]}), encoding="utf-8")
    if control["kind"] == "server":
        for name in ("training/incoming", "training/approved", "training/datasets", "reports/validation", "registry/history"):
            safe_target(target, name).mkdir(parents=True, exist_ok=True)


def create_shortcut(target):
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    link = desktop / "CV-1.lnk"
    script = target / ".cv1_shortcut.ps1"
    script.write_text("param($Link,$Target,$Work)\n$s=(New-Object -ComObject WScript.Shell).CreateShortcut($Link)\n$s.TargetPath=$Target\n$s.WorkingDirectory=$Work\n$s.Save()\n", encoding="utf-8-sig")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), str(link), str(target / "Kern_Analyzer.exe"), str(target)], creationflags=0x08000000, check=True)


def main():
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    control = json.loads((bundle / "release_control.json").read_text(encoding="utf-8"))
    media = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else bundle
    if "--verify-only" in sys.argv:
        try:
            install(control, media)
            result = {"ok":True,"parts":len(control["parts"]),"files":len(control["files"])}
        except Exception as exc: result = {"ok":False,"error":str(exc)}
        (media / (Path(sys.executable).stem + ".verify.json")).write_text(json.dumps(result), encoding="utf-8")
        return 0 if result["ok"] else 1
    window = tk.Tk(); window.title("CV-1 " + control["version"]); window.geometry("700x340")
    window.configure(bg="#f6faff")
    server = control["kind"] == "server"
    tk.Label(window, text="Модели для общей папки" if server else "Установка CV-1 с локальной Qwen", font=("Segoe UI",18,"bold"), bg="#f6faff", fg="#153e77").pack(anchor="w",padx=20,pady=(18,10))
    tk.Label(window, text="Все части данных должны находиться рядом с этим EXE. Интернет не требуется.\nПроекты и подтверждённые данные не заменяются при установке.",justify="left", bg="#f6faff",fg="#153e77").pack(anchor="w",padx=20)
    folder = tk.StringVar(value=str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / ("CV1-Server" if server else "Programs/CV-1")))
    row = tk.Frame(window,bg="#f6faff"); row.pack(fill="x",padx=20,pady=14)
    ttk.Entry(row,textvariable=folder).pack(side="left",fill="x",expand=True)
    ttk.Button(row,text="Папка…",command=lambda: folder.set(filedialog.askdirectory() or folder.get())).pack(side="right",padx=(8,0))
    status = tk.StringVar(value="Готово к установке")
    tk.Label(window,textvariable=status,wraplength=650,bg="#f6faff",fg="#153e77",justify="left").pack(anchor="w",padx=20)
    bar = ttk.Progressbar(window,maximum=100); bar.pack(fill="x",padx=20,pady=10)
    shortcut = tk.BooleanVar(value=not server)
    if not server: ttk.Checkbutton(window,text="Создать ярлык на рабочем столе",variable=shortcut).pack(anchor="w",padx=20)
    events, cancel, running = queue.Queue(), threading.Event(), [False]
    def start():
        if running[0]: return
        running[0]=True; button.configure(state="disabled")
        selected_target, add_shortcut = Path(folder.get()), shortcut.get()
        def work():
            try:
                install(control,media,selected_target,lambda name,value:events.put(("progress",name,value)),cancel.is_set)
                if add_shortcut: create_shortcut(selected_target)
                events.put(("done",str(selected_target)))
            except Exception as exc: events.put(("error",str(exc)))
        threading.Thread(target=work,daemon=True).start()
    def poll():
        try:
            while True:
                event=events.get_nowait()
                if event[0]=="progress": status.set(event[1]);bar["value"]=event[2]
                elif event[0]=="error": running[0]=False;button.configure(state="normal");messagebox.showerror("Установка не завершена",event[1])
                elif event[0]=="done":
                    running[0]=False;bar["value"]=100;status.set("Установка завершена")
                    if server: messagebox.showinfo("Готово","Модели распакованы. Откройте общий доступ к этой папке по инструкции администратора.")
                    elif messagebox.askyesno("Готово","CV-1 установлен. Открыть приложение?"):
                        subprocess.Popen([str(Path(event[1])/"Kern_Analyzer.exe")],cwd=event[1],creationflags=0x08000000)
        except queue.Empty: pass
        window.after(100,poll)
    def close():
        if running[0]: cancel.set();status.set("Останавливаю установку…")
        else:window.destroy()
    button=ttk.Button(window,text="Установить",command=start);button.pack(anchor="e",padx=20,pady=12)
    window.protocol("WM_DELETE_WINDOW",close);poll();window.mainloop();return 0


if __name__ == "__main__": raise SystemExit(main())
