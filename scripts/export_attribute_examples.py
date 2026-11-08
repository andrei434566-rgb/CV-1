"""Export reviewed feature crops; report an empty base instead of fake training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.infrastructure.attribute_examples import export_attribute_dataset

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    parser.add_argument("--root",type=Path)
    args=parser.parse_args()
    print(json.dumps(export_attribute_dataset(args.destination,root=args.root),ensure_ascii=False,indent=2))
