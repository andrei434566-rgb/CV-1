"""Estimate depths of photo links from printed log ticks for human review.

This is a layout-based estimate. Some cores have missing recovery and the PDF
contains misleading link positions; no output from this script is a training
label. The original PDF coordinates and all flags remain in the manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_kootenai_river_2011"
PDF = ROOT / "sir20115006_appendixA.pdf"
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def printed_ticks(page: pdfplumber.page.Page) -> tuple[list[dict], list[str]]:
    ticks = [{"depth_m": float(word["text"]), "top_pt": round(word["top"], 2)}
             for word in page.extract_words()
             if word["x0"] < 120 and word["top"] > 160 and NUMBER.fullmatch(word["text"])]
    ticks.sort(key=lambda tick: (tick["top_pt"], tick["depth_m"]))
    warnings = []
    if len(ticks) < 2 or ticks[0]["depth_m"] != 0:
        warnings.append("missing_zero_or_end_tick")
    if any(b["depth_m"] <= a["depth_m"] or b["top_pt"] <= a["top_pt"]
           for a, b in zip(ticks, ticks[1:])):
        warnings.append("conflicting_or_nonmonotonic_printed_ticks")
    return ticks, warnings


def interpolate(y: float, ticks: list[dict]) -> tuple[float | None, str | None]:
    if not ticks:
        return None, "no_depth_ticks"
    if y < ticks[0]["top_pt"]:
        return (ticks[0]["depth_m"], "above_first_tick") if y >= ticks[0]["top_pt"] - 10 else (None, "above_scale")
    if y > ticks[-1]["top_pt"]:
        return (ticks[-1]["depth_m"], "below_last_tick") if y <= ticks[-1]["top_pt"] + 25 else (None, "below_scale")
    for a, b in zip(ticks, ticks[1:]):
        if a["top_pt"] <= y <= b["top_pt"]:
            fraction = (y - a["top_pt"]) / (b["top_pt"] - a["top_pt"])
            return round(a["depth_m"] + fraction * (b["depth_m"] - a["depth_m"]), 3), None
    return None, "tick_interpolation_failure"


def main() -> None:
    index = json.loads((ROOT / "photo_link_index.json").read_text(encoding="utf-8"))
    downloads = json.loads((ROOT / "photo_download_inventory.json").read_text(encoding="utf-8"))
    by_url = {record["url"]: record for record in downloads["photos"]}
    reader = PdfReader(PDF)
    output_pages = []
    with pdfplumber.open(PDF) as pages:
        for page_index, (pdf_page, layout_page) in enumerate(zip(reader.pages, pages.pages), 1):
            if page_index == 1:
                continue
            core_id = index["pages"][page_index - 1]["core_id"]
            ticks, warnings = printed_ticks(layout_page)
            source_text = index["pages"][page_index - 1]["source_text_en"]
            if "No core recovered" in source_text or "no core recovered" in source_text:
                warnings.append("missing_core_recovery_text")
            link_positions = []
            for annotation in pdf_page.get("/Annots", []):
                obj = annotation.get_object()
                raw_url = obj.get("/A", {}).get("/URI")
                if not raw_url:
                    continue
                url = str(raw_url).replace("http://pubs.usgs.gov/", "https://pubs.usgs.gov/", 1)
                rect = [float(v) for v in obj["/Rect"]]
                link_positions.append({"url": url, "link_top_pt": round(float(layout_page.height) - rect[3], 2),
                                       "link_rect_pdf_points": [round(v, 2) for v in rect]})
            link_positions.sort(key=lambda link: link["link_top_pt"])
            unique_positions = []
            for link in link_positions:
                if any(existing["url"] == link["url"] for existing in unique_positions):
                    continue
                unique_positions.append(link)
            for ordinal, link in enumerate(unique_positions):
                record = by_url[link["url"]]
                link["local_photo"] = record.get("local_file") if record["status"] == "downloaded" else None
                link["photo_sha256"] = record.get("sha256")
                link["source_link_corrected"] = record.get("source_link_corrected", False)
                next_y = (unique_positions[ordinal + 1]["link_top_pt"]
                          if ordinal + 1 < len(unique_positions) else ticks[-1]["top_pt"] if ticks else None)
                link["next_link_top_pt"] = next_y
                if warnings or next_y is None:
                    link["layout_estimate_m"] = None
                    link["estimate_flags"] = warnings.copy()
                else:
                    start, flag_a = interpolate(link["link_top_pt"], ticks)
                    end, flag_b = interpolate(next_y, ticks)
                    flags = [flag for flag in (flag_a, flag_b) if flag]
                    if start is None or end is None or end <= start:
                        flags.append("invalid_or_zero_estimate")
                    # The first hyperlink can sit a few points above the
                    # printed zero tick. Keep that estimate but expose the
                    # flag; any other ambiguity suppresses the range.
                    link["layout_estimate_m"] = ([start, end] if set(flags) <= {"above_first_tick"} else None)
                    link["estimate_flags"] = flags
                link["interval_pixel_aligned"] = False
                link["russian_visual_target_reviewed"] = False
                link["training_approved"] = False
            output_pages.append({"pdf_page": page_index, "core_id": core_id,
                                 "printed_depth_ticks": ticks, "page_flags": warnings,
                                 "linked_photos": unique_positions,
                                 "training_approved": False})
    candidates = [photo for page in output_pages for photo in page["linked_photos"]]
    summary = {"core_log_pages": len(output_pages), "link_instances": len(candidates),
               "available_photo_link_instances": sum(p["local_photo"] is not None for p in candidates),
               "layout_estimates": sum(p["layout_estimate_m"] is not None for p in candidates),
               "approved_training_pairs": 0}
    result = {"source_pdf_sha256": index["source_pdf_sha256"], "pages": output_pages,
              "summary": summary,
              "warning": "Layout estimates are only search aids. They are not source-certified image depth ranges; no pairs may enter training without ruler and lithology review."}
    (ROOT / "depth_review_candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
