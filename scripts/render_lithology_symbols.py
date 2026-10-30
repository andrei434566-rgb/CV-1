"""Render the real Qt legend for visual review without opening a project."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication
from app.domain.lithology import default_lithology_palette
from app.ui.widgets.workspace_canvas import StackColumnItem

app = QApplication.instance() or QApplication([])
if not QFontDatabase.families():
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
app.setFont(QFont("Arial", 10))
palette = default_lithology_palette()
image = QImage(1056, 128 + ((len(palette) + 2) // 3) * 258, QImage.Format.Format_ARGB32)
image.fill(QColor("#edf2f8"))
painter = QPainter(image)
painter.setRenderHint(QPainter.RenderHint.Antialiasing)
font = QFont("Arial", 18)
font.setBold(True)
painter.setFont(font)
painter.setPen(QColor("#163354"))
painter.drawText(QRectF(30, 18, 996, 38), Qt.AlignmentFlag.AlignLeft, "Литология · условные обозначения")
font.setPointSize(10)
font.setBold(False)
painter.setFont(font)
painter.setPen(QColor("#65778d"))
painter.drawText(QRectF(30, 62, 996, 25), "Равномерный узор · компактная подпись · ≈ предварительное определение")
for index, item in enumerate(palette):
    row, column = divmod(index, 3)
    card = QRectF(24 + column * 344, 104 + row * 258, 320, 238)
    painter.setPen(QPen(QColor("#d7e1ed"), 1))
    painter.setBrush(QColor("white"))
    painter.drawRoundedRect(card, 12, 12)
    title = QFont("Arial", 10)
    title.setBold(True)
    painter.setFont(title)
    painter.setPen(QColor("#163354"))
    painter.drawText(card.adjusted(14, 10, -12, -194), Qt.TextFlag.TextWordWrap, item["name"])
    track = QRectF(card.left() + 18, card.top() + 48, 108, 172)
    color = QColor(item["color"])
    painter.fillRect(track, color)
    StackColumnItem._fill_legend_pattern(painter, track, color, item["pattern"])
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(color.darker(130), .8))
    painter.drawRect(track)
    StackColumnItem._draw_badge_caption(painter, track, item["symbol"], item["name"], tentative=index == 1)
    painter.setFont(QFont("Arial", 10))
    painter.setPen(QColor("#526880"))
    painter.drawText(QRectF(card.left() + 146, card.top() + 68, 155, 135), Qt.TextFlag.TextWordWrap,
                     "Код: " + item["symbol"] + "\n\nУзор повторяется без растяжения.")
painter.end()
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/lithology_symbols_preview.png")
output.parent.mkdir(parents=True, exist_ok=True)
if not image.save(str(output)):
    raise RuntimeError("Could not save symbol preview")
print(str(output.resolve()))
