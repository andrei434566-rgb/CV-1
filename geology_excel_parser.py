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
            header_labels={}
            for c in range(1,max_col+1):
                parts=[_s(cell(r,c)) for r in range(1,header_rows+1) if cell(r,c) is not None]
                header_labels[c]=parts
                for label in reversed(parts):
                    kind=_classify(label)
                    if kind and kind not in ("top","bottom") and kind not in fields:
                        fields[kind]=c
                        break
            # Depth pairs must belong to the FACIES/LAYER interval group.
            # Never fall back to the core recovery/collection interval.
            candidates=[]
            for c in range(1,max_col):
                labels=header_labels[c]
                if not any(_classify(v)=="top" for v in labels):
                    continue
                for d in range(c+1,min(c+4,max_col+1)):
                    right=header_labels[d]
                    if not any(_classify(v)=="bottom" for v in right):
                        continue
                    context=" ".join(labels+right)
                    if any(word in context for word in ("отбора керна","выноса керна","интервал керна","core recovery")):
                        continue
                    score=0
                    if "интервал фации по бурению" in context or ("фаци" in context and "бурению" in context):
                        score+=100
                    if "по гис" in context or "гис" in context:
                        score-=50
                    if any(word in context for word in ("фаци","слоя","слой","послой")):
                        score+=10
                    if d==c+1: score+=2
                    candidates.append((score,c,d))
            score,top_c,bottom_c=max(candidates) if candidates else (-999,None,None)
            fields["top"]=top_c
            fields["bottom"]=bottom_c
            desc_c=fields.get("description")
            facies_c=fields.get("facies")
            lith_c=fields.get("lithology")
            index_c=fields.get("facies_index")
            code_c=fields.get("facies_code")
            thick_c=fields.get("thickness")
            # Standard sedimentological form: numbered columns 14-15 are
            # FACIES DRILLING depths, 19 thickness, 20 index, 21 name.
            # Only apply when those exact numbered columns are present.
            numbered={}
            for rr in range(1,min(ws.max_row,80)+1):
                mapping={}
                for cc in range(1,max_col+1):
                    v=ws.cell(rr,cc).value
                    try:
                        number=int(str(v).strip())
                    except (TypeError,ValueError):
                        continue
                    if str(v).strip() in (str(number),str(number)+".0") and 1<=number<=max_col:
                        mapping[number]=cc
                if all(n in mapping for n in (14,15,19,20,21)):
                    numbered=mapping
                    break
            if numbered:
                top_c,bottom_c=numbered[14],numbered[15]
                thick_c,index_c,facies_c=(numbered[n] for n in (19,20,21))
                fields.update(top=top_c,bottom=bottom_c,thickness=thick_c,
                              facies_index=index_c,facies=facies_c)
                if 22 in numbered: code_c=numbered[22]

            # Some workbooks lack a machine-readable numbered header.
            # Discover the facies-drilling pair by DATA, not by a merged
            # group header which may describe the entire core run.
            # Restrict candidate pairs to 14/15 and their numbered mapping,
            # plus explicitly classified depth pairs. Do not silently use
            # the core-recovery interval as a substitute.
            depth_pairs=[]
            if top_c and bottom_c:
                depth_pairs.append((top_c,bottom_c,"header"))
            if max_col >= 15:
                depth_pairs.append((14,15,"physical_14_15"))
            if numbered:
                depth_pairs.insert(0,(numbered[14],numbered[15],"numbered_14_15"))
            for _,a,b in sorted(candidates,reverse=True)[:5]:
                depth_pairs.append((a,b,"facies_header"))
            unique_pairs=list(dict.fromkeys((a,b) for a,b,_ in depth_pairs))
            # Look for plausible short layers rather than long photo/core
            # intervals. Candidate count alone is not sufficient.
            def pair_quality(a,b):
                good=[]
                for rr in range(1,ws.max_row+1):
                    x=_num(ws.cell(rr,a).value)
                    y=_num(ws.cell(rr,b).value)
                    if x is not None and y is not None and 0 < y-x <= 30:
                        good.append((x,y))
                short=sum((y-x)<3 for x,y in good)
                distinct=len(set(good))
                # Prefer genuinely subdivided layers; reject repeating
                # identical aggregate intervals with no short layers.
                return (short*3+distinct,short,distinct),good
            if unique_pairs:
                rated=[(pair_quality(a,b)[0],a,b) for a,b in unique_pairs]
                best,a,b=max(rated)
                if best[1] > 0 or not (top_c and bottom_c):
                    top_c,bottom_c=a,b
                    fields.update(top=top_c,bottom=bottom_c)
            # Fallback field locations are ONLY used when their numbered
            # column headers have been positively identified.
            # A wrong field is worse than a missing one.
            if top_c is None or bottom_c is None or not any((desc_c,facies_c,lith_c,index_c)): continue
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
                if not (facies or facies_index or facies_code): continue
                if facies_index.replace(".","",1).isdigit() and not facies: continue
                results.append(dict(file=str(path),sheet=ws.title,row=r,top=top,bottom=bottom,
                                    description=description,facies=facies,lithology=lithology,
                                    facies_index=facies_index,facies_code=facies_code,
                                    thickness=thickness if thickness is not None else round(bottom-top,3),
                                    thickness_error_m=round((bottom-top)-thickness,4) if thickness is not None else None,
                                    depth_basis="facies_layer_not_core_collection",
                                    source_columns={k:ws.cell(r,c).coordinate for k,c in fields.items()}))
        wb.close()
        return results,None
    except Exception as exc:
        return [],str(exc)
