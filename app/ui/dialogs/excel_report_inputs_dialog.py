"""Specialist-entered well passport for the fixed Excel description form."""

from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class ExcelReportInputsDialog(QDialog):
    FIELDS = (
        ("field", "Месторождение"),
        ("well_number", "№ скважины"),
        ("core_top", "Общий интервал — кровля, м (если нет долблений)"),
        ("core_base", "Общий интервал — подошва, м (если нет долблений)"),
        ("stratigraphy", "Стратиграфия"),
        ("gis_shift", "Смещение по ГИС, м"),
        ("penetration", "Проходка, м"),
        ("core_recovered", "Вынос керна, м"),
    )
    NUMERIC_FIELDS = frozenset({"core_top", "core_base", "gis_shift", "penetration", "core_recovered"})

    def __init__(self, well_name: str, values: dict | None = None, parent=None):
        super().__init__(parent)
        values = dict(values or {})
        self.setWindowTitle(f"Паспорт для итогового Excel — {well_name}")
        self.setMinimumWidth(550)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Общие поля вводятся один раз на скважину. Долбления укажите ниже с их "
            "интервалами; общая кровля и подошва возьмутся из минимума и максимума "
            "этих интервалов. Смещение ГИС может быть отрицательным."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.inputs: dict[str, QLineEdit] = {}
        for key, label in self.FIELDS:
            current = values.get(key)
            initial = well_name if key == "well_number" and current is None else "" if current is None else str(current)
            editor = QLineEdit(initial)
            if key in self.NUMERIC_FIELDS:
                editor.setPlaceholderText("Число, можно с запятой" if key != "gis_shift" else "Например, -1 или 0")
                editor.textChanged.connect(self.update_preview)
            form.addRow(label + ":", editor)
            self.inputs[key] = editor
        layout.addLayout(form)
        layout.addWidget(QLabel("Долбления: номер и интервал отбора керна, м"))
        self.runs = QTableWidget(0, 3)
        self.runs.setHorizontalHeaderLabels(("№ долбления", "Кровля", "Подошва"))
        self.runs.horizontalHeader().setStretchLastSection(True)
        self.runs.setMinimumHeight(155)
        layout.addWidget(self.runs)
        run_buttons = QHBoxLayout()
        add_run = QPushButton("Добавить долбление")
        add_run.clicked.connect(lambda _checked=False: self._append_run())
        remove_run = QPushButton("Удалить строку")
        remove_run.clicked.connect(lambda _checked=False: self._remove_selected_run())
        run_buttons.addWidget(add_run)
        run_buttons.addWidget(remove_run)
        run_buttons.addStretch(1)
        layout.addLayout(run_buttons)
        source_runs = values.get("drilling_runs")
        if isinstance(source_runs, list) and source_runs:
            for run in source_runs:
                if isinstance(run, dict):
                    self._append_run(run.get("number", ""), run.get("top", ""), run.get("base", ""))
        elif values.get("drilling_number"):
            self._append_run(values.get("drilling_number", ""), values.get("core_top", ""), values.get("core_base", ""))
        if not self.runs.rowCount():
            self._append_run()
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        self.runs.itemChanged.connect(self.update_preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._result: dict = {}
        self.update_preview()

    @staticmethod
    def _number(text: str, label: str) -> float | None:
        text = text.strip().replace(" ", "").replace(",", ".")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"«{label}»: укажите число") from exc
        if not math.isfinite(number):
            raise ValueError(f"«{label}»: укажите конечное число")
        return number

    def values(self) -> dict:
        result: dict = {}
        for key, label in self.FIELDS:
            text = self.inputs[key].text().strip()
            if key in self.NUMERIC_FIELDS:
                value = self._number(text, label)
                if value is not None:
                    result[key] = value
            elif text:
                result[key] = text
        runs = self._read_runs()
        if runs:
            result["drilling_runs"] = runs
            result["core_top"] = min(run["top"] for run in runs)
            result["core_base"] = max(run["base"] for run in runs)
        top, base = result.get("core_top"), result.get("core_base")
        if (top is None) != (base is None):
            raise ValueError("Для интервала отбора керна укажите и кровлю, и подошву")
        if top is not None and base is not None and base <= top:
            raise ValueError("Подошва интервала отбора керна должна быть глубже кровли")
        for key, label in (("penetration", "Проходка"), ("core_recovered", "Вынос керна")):
            if key in result and result[key] < 0:
                raise ValueError(f"«{label}» не может быть отрицательным")
        return result

    def _append_run(self, number: object = "", top: object = "", base: object = "") -> None:
        row = self.runs.rowCount()
        self.runs.insertRow(row)
        for column, value in enumerate((number, top, base)):
            self.runs.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))
        if hasattr(self, "preview"):
            self.update_preview()

    def _remove_selected_run(self) -> None:
        row = self.runs.currentRow()
        if row >= 0:
            self.runs.removeRow(row)
        self.update_preview()

    def _read_runs(self) -> list[dict[str, str | float]]:
        runs: list[dict[str, str | float]] = []
        for row in range(self.runs.rowCount()):
            cells = [self.runs.item(row, column) for column in range(3)]
            number, top_text, base_text = [item.text().strip() if item else "" for item in cells]
            if not any((number, top_text, base_text)):
                continue
            if not all((number, top_text, base_text)):
                raise ValueError(f"Долбление, строка {row + 1}: укажите номер, кровлю и подошву")
            top = self._number(top_text, f"Кровля долбления {number}")
            base = self._number(base_text, f"Подошва долбления {number}")
            if top is None or base is None or base <= top:
                raise ValueError(f"Долбление {number}: подошва должна быть глубже кровли")
            runs.append({"number": number, "top": top, "base": base})
        runs.sort(key=lambda run: (run["top"], run["base"]))
        for previous, following in zip(runs, runs[1:]):
            if following["top"] < previous["base"]:
                raise ValueError(f"Интервалы долблений {previous['number']} и {following['number']} перекрываются")
        return runs

    def update_preview(self, *_args) -> None:
        try:
            top = self._number(self.inputs["core_top"].text(), "Кровля")
            base = self._number(self.inputs["core_base"].text(), "Подошва")
            shift = self._number(self.inputs["gis_shift"].text(), "Смещение")
            penetration = self._number(self.inputs["penetration"].text(), "Проходка")
            recovered = self._number(self.inputs["core_recovered"].text(), "Вынос")
            runs = self._read_runs()
        except ValueError:
            self.preview.setText("Проверьте числовые поля перед сохранением.")
            return
        if runs:
            top = min(run["top"] for run in runs)
            base = max(run["base"] for run in runs)
        gis = f"{top + shift:g}–{base + shift:g} м" if None not in (top, base, shift) else "после ввода кровли, подошвы и смещения"
        recovery = f"{recovered / penetration * 100:g}%" if penetration is not None and penetration > 0 and recovered is not None else "после ввода проходки и выноса"
        overall = f"{top:g}–{base:g} м" if None not in (top, base) else "не задан"
        self.preview.setText(f"Общий интервал: {overall}. По ГИС: {gis}. Вынос: {recovery}.")

    def accept(self) -> None:
        try:
            self._result = self.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте паспорт", str(exc))
            return
        super().accept()

    def result_values(self) -> dict:
        return dict(self._result)
