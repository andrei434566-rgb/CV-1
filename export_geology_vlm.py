"""Export human-reviewed photo/Excel matches into VLM-ready JSONL.

Only confirmed/corrected matches are exported. Source photos are referenced,
not copied, and original geological files are never modified.
"""
from __future__ import annotations
import json
from pathlib import Path

def export(index_dir: Path):
    index_dir=Path(index_dir)
    archive=json.loads((index_dir/"archive_index.json").read_text(encoding="utf-8"))
    decisions=json.loads((index_dir/"pair_decisions.json").read_text(encoding="utf-8"))["decisions"]
    from index_geology_archive import uv_filename
    excluded_uv=set(archive.get("excluded_uv",[]))
    suspected_uv=set(archive.get("suspected_uv",[]))
    destination=index_dir/"vlm_dataset"
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/"reviewed_examples.jsonl"
    records=[]
    skipped=0
    for pair in archive.get("pairs",[]):
        photo=pair.get("photo")
        if not photo or uv_filename(Path(photo)) or photo in excluded_uv or photo in suspected_uv:
            skipped+=1
            continue
        decision=decisions.get(photo,{})
        if decision.get("decision") not in ("confirmed","corrected_candidate"):
            skipped+=1
            continue
        j=decision.get("candidate_index")
        matches=pair.get("matches",[])
        if not isinstance(j,int) or not 0<=j<len(matches) or not Path(photo).is_file():
            skipped+=1
            continue
        match=matches[j]
        if not any(match.get(key) for key in ("description","facies","lithology")):
            skipped+=1
            continue
        records.append({
            "image":photo,
            "well":pair.get("well"),
            "photo_depth":pair.get("photo_depth"),
            "interval_depth":[match.get("top"),match.get("bottom")],
            "description":match.get("description",""),
            "facies":match.get("facies",""),
            "lithology":match.get("lithology",""),
            "provenance":{"excel":match.get("excel"),"sheet":match.get("sheet"),
                          "row":match.get("row")},
            "review_status":decision["decision"],
            "note":"Interval-level weak supervision: photo may contain multiple intervals. Not pixel labels."
        })
    with output.open("w",encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
    summary={"examples":len(records),"skipped":skipped,"output":str(output),
             "training_started":False}
    (destination/"export_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("index_dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(export(args.index_dir),ensure_ascii=False,indent=2))
