"""Depth mapping for core tray lanes with explicit orientation and provenance.

No depths are inferred from pixel positions unless a human supplies lane
start/end depths and confirms the lane direction. The output is annotation
metadata, not ground-truth facies masks.
"""
from __future__ import annotations
import json
from pathlib import Path

def map_lanes(photo: str, lanes: list[dict], descriptions: list[dict]):
    """lanes: [{x1,y1,x2,y2,top,bottom,axis,reverse}, ...].

    axis='vertical' means top->bottom, axis='horizontal' means left->right.
    reverse=True inverts direction. Coordinates are pixels in original photo.
    """
    result=[]
    for lane_index,lane in enumerate(lanes):
        x1,y1,x2,y2=[float(lane[k]) for k in ("x1","y1","x2","y2")]
        top,bottom=float(lane["top"]),float(lane["bottom"])
        axis=lane.get("axis","vertical")
        if not (x1<x2 and y1<y2 and top<bottom and axis in ("vertical","horizontal")):
            raise ValueError(f"Invalid geometry/depth for lane {lane_index}")
        intervals=[]
        for d in descriptions:
            a,b=float(d["top"]),float(d["bottom"])
            lo,hi=max(top,a),min(bottom,b)
            if hi<=lo: continue
            start=(lo-top)/(bottom-top)
            end=(hi-top)/(bottom-top)
            if lane.get("reverse",False):
                start,end=1-end,1-start
            if axis=="vertical":
                rect=[x1,y1+start*(y2-y1),x2,y1+end*(y2-y1)]
            else:
                rect=[x1+start*(x2-x1),y1,x1+end*(x2-x1),y2]
            intervals.append({"depth":[lo,hi],"rect_xyxy":[round(v,2) for v in rect],
                              "source_excel":d.get("file"),"sheet":d.get("sheet"),
                              "row":d.get("row"),"facies":d.get("facies",""),
                              "lithology":d.get("lithology",""),
                              "description":d.get("description",""),
                              "status":"depth_projected_requires_review"})
        result.append({"photo":photo,"lane_index":lane_index,"lane":lane,"intervals":intervals})
    return result

def generate(index_dir: Path):
    index_dir=Path(index_dir)
    archive=json.loads((index_dir/"archive_index.json").read_text(encoding="utf-8"))
    config_path=index_dir/"lane_annotations.json"
    if not config_path.exists():
        example={"instructions":"Coordinates in ORIGINAL photo pixels. top/bottom in meters. Verify direction and intervals manually.",
                 "photos":{}}
        config_path.write_text(json.dumps(example,ensure_ascii=False,indent=2),encoding="utf-8")
        return {"status":"template_created","file":str(config_path),"mapped":0}
    config=json.loads(config_path.read_text(encoding="utf-8"))
    by_photo={p["file"]:p for p in archive.get("photos",[])}
    output=[]
    for photo,lanes in config.get("photos",{}).items():
        if photo not in by_photo:
            continue
        well=by_photo[photo]["well"]
        descriptions=[d for d in archive.get("descriptions",[]) if d.get("well")==well]
        output.extend(map_lanes(photo,lanes,descriptions))
    destination=index_dir/"depth_mapped_intervals.json"
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"status":"generated","file":str(destination),"mapped":len(output),
            "intervals":sum(len(v["intervals"]) for v in output)}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("index_dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(generate(args.index_dir),ensure_ascii=False,indent=2))
