"""Small Windows interface for LAN exchange of core source files."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import queue
import socket
import threading
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import lan_sync


class CoreSedSyncApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("CoreSed — обмен данными в локальной сети")
        root.geometry("800x550")
        root.minsize(680, 460)
        self.hub = tk.StringVar()
        self.source = tk.StringVar()
        self.computer = tk.StringVar(value=lan_sync.safe_id(
            "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "_"
                    for c in socket.gethostname())[:40] or "PC"
        ))
        self.source_id = tk.StringVar()
        self.kind = tk.StringVar(value="raw")
        self.status = tk.StringVar(value="Укажите общий сетевой каталог и папку с новыми материалами.")
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False
        self.buttons: list[ttk.Button] = []
        self.build()
        self.root.after(120, self.poll)

    def build(self) -> None:
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Сбор данных керна с нескольких ПК",
                  font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Фото, Excel и документы поступают в общую очередь. Обучение автоматически не запускается.",
                  wraplength=740).pack(anchor="w", pady=(5, 15))
        ttk.Label(outer, text=f"Имя этого компьютера в сети: {socket.gethostname()}",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 9))

        paths = ttk.LabelFrame(outer, text="Папки", padding=12)
        paths.pack(fill="x")
        self.path_row(paths, 0, "Общее хранилище", self.hub, self.choose_hub)
        self.path_row(paths, 1, "Новые материалы", self.source, self.choose_source)

        details = ttk.Frame(outer)
        details.pack(fill="x", pady=12)
        ttk.Label(details, text="Имя ПК").grid(row=0, column=0, sticky="w")
        ttk.Entry(details, textvariable=self.computer, width=17).grid(row=1, column=0, padx=(0, 12), sticky="w")
        ttk.Label(details, text="Название партии / скважины").grid(row=0, column=1, sticky="w")
        ttk.Entry(details, textvariable=self.source_id, width=29).grid(row=1, column=1, padx=(0, 12), sticky="w")
        ttk.Label(details, text="Тип материалов").grid(row=0, column=2, sticky="w")
        ttk.Combobox(details, textvariable=self.kind, values=("raw", "app_sessions", "prepared"),
                     state="readonly", width=17).grid(row=1, column=2, sticky="w")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 10))
        for label, operation in (
            ("1. Создать хранилище", "init"),
            ("2. Передать папку", "submit"),
            ("3. Принять пакеты", "ingest"),
            ("Показать состояние", "status"),
        ):
            button = ttk.Button(actions, text=label, command=lambda op=operation: self.run(op))
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        ttk.Label(outer, textvariable=self.status, wraplength=740).pack(anchor="w", pady=(0, 8))
        self.log = tk.Text(outer, wrap="word", height=14, font=("Consolas", 9),
                           bg="#142426", fg="#eaf3ef", state="disabled")
        self.log.pack(fill="both", expand=True)
        ttk.Label(outer, text="На центральном ПК выполните шаги 1 и 3. На остальных ПК — шаг 2. "
                  "Сетевую папку Windows откройте только доверенным учётным записям.",
                  wraplength=740).pack(anchor="w", pady=(8, 0))

    def path_row(self, parent: ttk.Frame, row: int, label: str,
                 variable: tk.StringVar, choose) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Выбрать…", command=choose).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    def choose_hub(self) -> None:
        path = filedialog.askdirectory(title="Выберите локальную или сетевую папку хранилища")
        if path:
            self.hub.set(path)

    def choose_source(self) -> None:
        path = filedialog.askdirectory(title="Выберите папку новых фото и Excel")
        if path:
            self.source.set(path)

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self, operation: str) -> None:
        if self.busy:
            return
        hub = self.hub.get().strip()
        if not hub:
            messagebox.showerror("Нужна папка", "Укажите путь к общему хранилищу.")
            return
        source = self.source.get().strip()
        if operation == "submit" and (not source or not Path(source).is_dir()):
            messagebox.showerror("Нужны материалы", "Выберите папку с фото, Excel или документами.")
            return
        args = SimpleNamespace(hub=hub, source=source, pc=self.computer.get().strip(),
                               source_id=self.source_id.get().strip(), kind=self.kind.get())
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.status.set("Выполняется: " + operation)
        self.append_log("→ " + operation)

        def worker() -> None:
            output = io.StringIO()
            try:
                with redirect_stdout(output):
                    if operation == "init":
                        lan_sync.initialize(args.hub)
                    elif operation == "submit":
                        lan_sync.submit(args)
                    elif operation == "ingest":
                        lan_sync.ingest(args)
                    else:
                        lan_sync.status(args)
                self.events.put(("ok", output.getvalue()))
            except Exception as error:
                self.events.put(("error", output.getvalue() + str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self) -> None:
        try:
            while True:
                kind, text = self.events.get_nowait()
                self.append_log(text)
                self.status.set("Готово" if kind == "ok" else "Ошибка — подробности в журнале")
                for button in self.buttons:
                    button.configure(state="normal")
                self.busy = False
        except queue.Empty:
            pass
        self.root.after(120, self.poll)


def main() -> None:
    root = tk.Tk()
    CoreSedSyncApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
