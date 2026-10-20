"""CV-1 archive self-test. No changes to source photographs or spreadsheets."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

def diagnose(root: Path, index_dir: Path):
    from geology_excel_parser import excel_intervals
    from index_geology_archive import uv_filename, likely_uv_visual
    from review_geology_pairs import __file__ as review_source
    result={"runtime": {"python":sys.executable, "diagnostic_file":str(Path(__file__).resolve()),
             "review_file":str(Path(review_source).resolve()),
             "parser_file":str(Path(sys.modules["geology_excel_parser"].__file__).resolve()),
             "index_file":str(Path(sys.modules["index_geology_archive"].__file__).resolve())}}
    index_path=index_dir/"archive_index.json"
    if index_path.exists():
        data=json.loads(index_path.read_text(encoding="utf-8"))
        pairs=data.get("pairs",[])
        uv_named=[p.get("photo") for p in pairs if p.get("photo") and uv_filename(Path(p["photo"]))]
        result["index"]={"path":str(index_path),"pairs":len(pairs),"uv_named_still_in_index":len(uv_named),
                         "uv_examples":uv_named[:8],"match_counts":dict(Counter(str(len(p.get("matches",[]))) for p in pairs))}
    else:
        result["index"]={"error":"Index not found; press Scan first"}
    files=list(root.rglob("*.xlsx"))+list(root.rglob("*.xlsm"))
    result["excel"]=[]
    for path in files[:25]:
        if path.name.startswith("~$"): continue
        rows,error=excel_intervals(path)
        result["excel"].append({"file":str(path),"facies_rows":len(rows),"error":error,
            "examples":[{k:r.get(k) for k in ("sheet","row","top","bottom","thickness","facies_index","facies")} for r in rows[:5]]})
    result["filename_uv_checks"]={name:uv_filename(Path(name)) for name in (
        "Core_UV_1111_4_3672.00-3683.75.jpg",
        "Core_day_1111_4_3672.00-3683.75.jpg",
        "Core_ультрафиолет_1111.jpg")}
    return result

def save_report(root:Path,index_dir:Path):
    report=diagnose(root,index_dir)
    index_dir.mkdir(parents=True,exist_ok=True)
    dest=index_dir/"cv1_diagnostic.json"
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return dest,report

if __name__=="__main__":
    if len(sys.argv)<3:
        raise SystemExit('Usage: python diagnose_geology.py "archive_root" "index_directory"')
    dest,report=save_report(Path(sys.argv[1]),Path(sys.argv[2]))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print("Report:",dest)
