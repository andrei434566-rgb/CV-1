"""Fetch two public CC-BY figure sources and verify the exact source bytes.

Preparation requires Pillow and pypdf; no changes to the ML environment.
No authentication, proxy rotation, CAPTCHA handling or hidden downloads.
"""
import argparse
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_web_facies_20261031 import SOURCE_SHA

FILES={
    'sydney_pdf.pdf':(
        'https://mdpi-res.com/d_attachment/geosciences/geosciences-11-00273/article_deploy/geosciences-11-00273.pdf',
        '10757f410bb2beffbc86f5feec31b2956e4f8deb4664e804dd5f43f76d22a382'),
    'fujian_fig3.webp':(
        'https://www.frontiersin.org/files/Articles/1909600/xml-images/fmars-13-1909600-g003.webp',
        SOURCE_SHA['fujian_fig3.webp']),
}


def digest(data): return sha256(data).hexdigest()


def fetch(output):
    from pypdf import PdfReader
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in FILES.items():
        path=output/name
        if path.exists() and digest(path.read_bytes())==expected:
            print('Verified local '+name,flush=True); continue
        temporary=path.with_name(path.name+'.download')
        with urlopen(url,timeout=45) as response,temporary.open('wb') as stream:
            size=0
            while True:
                block=response.read(1024**2)
                if not block: break
                size+=len(block)
                if size>40*1024**2: raise ValueError('Unexpected source size')
                stream.write(block)
        if digest(temporary.read_bytes())!=expected:
            raise ValueError('Published source changed; review bytes/labels before updating the contract: '+name)
        temporary.replace(path); print('Downloaded and verified '+name,flush=True)
    page=PdfReader(output/'sydney_pdf.pdf').pages[15]
    found={}
    for image in page.images:
        key=digest(image.data)
        for filename,expected in SOURCE_SHA.items():
            if key==expected: found[filename]=image.data
    for filename in ('sydney_p16_image0.jpg','sydney_p16_image1.jpg'):
        if filename not in found: raise ValueError('Exact publisher image not found; use the documented PDF reader version')
        (output/filename).write_bytes(found[filename]); print('Extracted original JPEG '+filename,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'open_datasets/facies_web_20261031/raw')
    args=p.parse_args(); fetch(args.output)
