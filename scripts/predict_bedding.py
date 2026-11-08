"""Classify one already selected rock ROI; no geometry changes or auto-labels."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--box',type=int,nargs=4)
    p.add_argument('--model',type=Path,default=ROOT/'models/bedding/rf100_convnext_20261106');args=p.parse_args()
    from PIL import Image,ImageOps
    from app.infrastructure.bedding_classifier import BeddingEngine
    with Image.open(args.image) as source:
        if source.width*source.height>100_000_000:raise ValueError('Изображение слишком велико')
        im=ImageOps.exif_transpose(source).convert('RGB')
        if args.box:
            l,t,r,b=args.box
            if not (0<=l<r<=im.width and 0<=t<b<=im.height):raise ValueError('Неверный фрагмент')
            im=im.crop(args.box)
        result=BeddingEngine(args.model).classify_roi(im)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
