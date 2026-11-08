"""Real offline runtime inference on existing C-only core crops; no reference labels invented."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.infrastructure.local_vision import VisionEngine

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);args=ap.parse_args()
    engine=VisionEngine();rows=[]
    for path in sorted((ROOT/'models/vision/pipeline_worker_benchmark_20261104').glob('tape*.jpg'))[:3]:
        reply=engine.describe({'image':str(path),'base':str(args.model),'operation':'describe'})
        rows.append({'image':path.name,'answer':reply['answer'],'fields':reply['fields'],
                     'predictions':reply['predictions'],'timing':reply['timing'],'tiles':reply['tiles'],
                     'color':reply['observations']['color'],'model_info':reply['model_info']})
        assert reply['ok'] and len(reply['predictions'])==len(engine.card['classes'])
        assert abs(sum(p['score'] for p in reply['predictions'])-1)<.001
        assert not reply['model_info']['generative'] and not reply['model_info']['facies_trained']
    output={'real_runtime':True,'geological_ground_truth_available':False,'rows':rows}
    dest=ROOT/'research_reports/rock_taxonomy_20261105/runtime_smoke.json'
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([{'image':r['image'],'color':r['color'],'prediction':r['predictions'][0],
                       'timing':r['timing']} for r in rows],ensure_ascii=False))

if __name__=='__main__':main()
