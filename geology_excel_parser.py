"""Flexible Excel core-description parser with provenance and conservative matching."""
from __future__ import annotations
import re

def _s(v):
    return re.sub(r"\s+", " ", str(v or "")).strip().casefold()

def _num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        t=v.strip().replace(" ", "").replace(",", ".")
        if re.fullmatch(r"\d{2,5}(?:\.\d{1,3})?",t): return float(t)
    return None

def _classify(label):
    s=_s(label)
    if any(t in s for t in ("литологическое описание", "описание керна", "геологическое описание")): return "description"
    if any(t in s for t in ("индекс фации", "индекс фац.")): return "facies_index"
    if any(t in s for t in ("код фации", "код фац.")): return "facies_code"
    if any(t in s for t in ("толщина слоя", "мощность слоя", "толщина фации")): return "thickness"
    if any(t in s for t in ("название фации", "наименование фации")): return "facies"
    if any(t in s for t in ("название литологии", "литология", "название породы")): return "lithology"
    if any(t in s for t in ("кровля", "начало интервала", "глубина от", "интервал от")): return "top"
    if any(t in s for t in ("подошва", "конец интервала", "глубина до", "интервал до")): return "bottom"
    return None

def excel_intervals(path):
    from openpyxl import load_workbook
    results=[]
    try:
        wb=load_workbook(path, read_only=False, data_only=True)
        for ws in wb.worksheets:
            # Expand merged headings in memory only, never change source workbook.
            merged={}
            for rng in ws.merged_cells.ranges:
                val=ws.cell(rng.min_row,rng.min_col).value
                if val is not None:
                    for rr in range(rng.min_row,rng.max_row+1):
                        for cc in range(rng.min_col,rng.max_col+1):
                            merged[(rr,cc)]=val
            def cell(r,c):
                return merged.get((r,c),ws.cell(r,c).value)
            max_col=min(ws.max_column,180)
            header_rows=min(ws.max_row,35)
            fields={}
            for c in range(1,max_col+1):
                parts=[_s(cell(r,c)) for r in range(1,header_rows+1) if cell(r,c) is not None]
                # Prefer the most specific leaf heading over parent group labels.
                for label in reversed(parts):
                    kind=_classify(label)
                    if kind and kind not in fields:
                        fields[kind]=c
                        break
            # If two depths are in the same parent group, preserve their distinct columns.
            if not all(k in fields for k in ("top","bottom")):
                continue
            top_c,bottom_c=fields["top"],fields["bottom"]
            if top_c==bottom_c: continue
            desc_c=fields.get("description")
            facies_c=fields.get("facies")
            lith_c=fields.get("lithology")
            index_c=fields.get("facies_index")
            code_c=fields.get("facies_code")
            thick_c=fields.get("thickness")
            if not any((desc_c,facies_c,lith_c)): continue
            for r in range(1,ws.max_row+1):
                top=_num(ws.cell(r,top_c).value)
                bottom=_num(ws.cell(r,bottom_c).value)
                if top is None or bottom is None or not (0 < bottom-top < 1000): continue
                description=str(ws.cell(r,desc_c).value or "").strip() if desc_c else ""
                facies=str(ws.cell(r,facies_c).value or "").strip() if facies_c else ""
                lithology=str(ws.cell(r,lith_c).value or "").strip() if lith_c else ""
                facies_index=str(ws.cell(r,index_c).value or "").strip() if index_c else ""
                facies_code=str(ws.cell(r,code_c).value or "").strip() if code_c else ""
                thickness=_num(ws.cell(r,thick_c).value) if thick_c else None
                if not (description or facies or lithology or facies_index or facies_code): continue
                results.append(dict(file=str(path),sheet=ws.title,row=r,top=top,bottom=bottom,
                                    description=description,facies=facies,lithology=lithology,
                                    facies_index=facies_index,facies_code=facies_code,
                                    thickness=thickness if thickness is not None else round(bottom-top,3),
                                    source_columns={k:ws.cell(r,c).coordinate for k,c in fields.items()}))
        wb.close()
        return results,None
    except Exception as exc:
        return [],str(exc)
