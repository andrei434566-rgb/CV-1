"""Readable local cards under either Windows system palette."""
from PySide6.QtGui import QColor, QPalette


def apply_light_card(widget):
    from app.runtime_paths import bundled_root
    check_icon = (bundled_root() / 'app/resources/icons/card-check.svg').as_posix()
    palette = QPalette(widget.palette())
    colors = {QPalette.ColorRole.Window: "#f7faff", QPalette.ColorRole.WindowText: "#163660",
              QPalette.ColorRole.Base: "#ffffff", QPalette.ColorRole.AlternateBase: "#eef4fc",
              QPalette.ColorRole.Text: "#163660", QPalette.ColorRole.Button: "#edf5ff",
              QPalette.ColorRole.ButtonText: "#163660", QPalette.ColorRole.Highlight: "#007de0",
              QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.PlaceholderText: "#5c718b"}
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for role, color in colors.items():
            palette.setColor(group, role, QColor(color))
    widget.setPalette(palette)
    widget.setStyleSheet('''
        QWidget { background-color: #f7faff; color: #163660; }
        QLabel, QCheckBox { background: transparent; }
        QCheckBox { spacing: 7px; }
        QCheckBox::indicator, QTableView::indicator { width: 16px; height: 16px;
            background: white; border: 1px solid #839ab5; border-radius: 3px; }
        QCheckBox::indicator:checked, QTableView::indicator:checked {
            background: #007de0; border: 1px solid #007de0; image: url("__CHECK__"); }
        QCheckBox::indicator:disabled { border-color: #b9c6d7; background: #e8edf3; }
        QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QTableWidget, QListWidget {
            background: #ffffff; color: #163660; border: 1px solid #b9cce3;
            border-radius: 5px; padding: 5px; selection-background-color: #007de0; selection-color: white; }
        QComboBox QAbstractItemView { background: white; color: #163660;
            selection-background-color: #007de0; selection-color: white; }
        QScrollArea { border: 1px solid #dbe5f2; border-radius: 5px; }
        QPushButton { background: #e9f3ff; color: #12518b; border: 1px solid #b9cce3;
            border-radius: 5px; padding: 6px 12px; }
        QPushButton:hover { background: #d7eaff; }
        QPushButton:disabled { color: #69778a; background: #e8edf3; }
        QTabWidget::pane { background: #f7faff; border: 1px solid #dbe5f2; }
        QTabBar::tab { background: #eaf1fb; color: #395779; min-width: 0; padding: 7px 9px; }
        QTabBar::tab:selected { background: white; color: #164d90; border-top: 2px solid #6c5bff; }
        QHeaderView::section { background: #eaf1fb; color: #163660; padding: 5px; }
    '''.replace('__CHECK__', check_icon))
