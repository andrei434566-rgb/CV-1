"""Quick visual preview of the first UKCS image from a partially downloaded ZIP."""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external_raw" / "ukcs_q204_q205_coremdlr"
ZIP = RAW / "Q204_data.zip"
IMAGE_HEADER_OFFSET = 1713464
ROW_LIMIT = 10000
WIDTH = 783
LABELS = {"s": "песчаник", "is": "глинистый песчаник", "ih": "песчанистый аргиллит", "os": "нефтенасыщенный песчаник", "sh": "аргиллит"}


def member_payload(path: Path) -> bytes:
    data = path.read_bytes()
    name_size, extra_size = struct.unpack_from("<HH", data, 26)
    compressed_size = struct.unpack_from("<I", data, 18)[0]
    return zlib.decompress(data[30 + name_size + extra_size:30 + name_size + extra_size + compressed_size], -15)


def image_prefix() -> np.ndarray:
    with ZIP.open("rb") as stream:
        stream.seek(IMAGE_HEADER_OFFSET)
        header = stream.read(30)
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        stream.read(name_size + extra_size)
        decompressor = zlib.decompressobj(-15)
        pieces = []
        length = 0
        target = ROW_LIMIT * WIDTH * 3 + 1024
        while length < target:
            compressed = stream.read(65536)
            if not compressed:
                raise EOFError("Partial ZIP does not yet contain preview rows")
            chunk = decompressor.decompress(compressed)
            pieces.append(chunk)
            length += len(chunk)
    raw = b"".join(pieces)
    header_stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(header_stream)
    reader = {1: np.lib.format.read_array_header_1_0, 2: np.lib.format.read_array_header_2_0}[version[0]]
    shape, fortran, dtype = reader(header_stream)
    if fortran or shape[1:] != (WIDTH, 3) or dtype != np.dtype("uint8"):
        raise ValueError((shape, fortran, dtype))
    start = header_stream.tell()
    available = (len(raw) - start) // (WIDTH * 3)
    return np.frombuffer(raw, dtype=np.uint8, count=available * WIDTH * 3, offset=start).reshape(available, WIDTH, 3)


def main() -> None:
    labels = np.load(io.BytesIO(member_payload(RAW / "sample_labels_member.bin")), allow_pickle=False)
    pixels = image_prefix()
    cropped = []
    metadata = []
    for code, title in LABELS.items():
        mask = labels[:min(len(pixels), ROW_LIMIT)] == code.encode("ascii")
        starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == -1)
        candidates = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 400]
        if not candidates:
            continue
        top, bottom = max(candidates, key=lambda pair: pair[1] - pair[0])
        center = (top + bottom) // 2
        image = Image.fromarray(pixels[center - 200:center + 200].copy(), "RGB")
        cropped.append((title, image))
        metadata.append({"label": code, "top_row": center - 200, "bottom_row": center + 200})
    sheet = Image.new("RGB", (WIDTH, len(cropped) * 450), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(cropped):
        y = index * 450
        draw.text((10, y + 10), title, fill="black")
        sheet.paste(image, (0, y + 50))
    output = RAW / "first_well_label_preview.jpg"
    sheet.save(output, quality=90)
    (RAW / "first_well_label_preview.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == "__main__":
    main()
