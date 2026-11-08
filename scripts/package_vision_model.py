"""Export only the selected non-generative model, with attribution and hashes."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.infrastructure.local_vision import default_model_folder,read_model_card

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--name',default='CV1_Vision_Model.zip')
    args=ap.parse_args();folder=args.model.resolve() if args.model else default_model_folder();card=read_model_card(folder)
    if Path(args.name).name!=args.name or not args.name.endswith('.zip'):raise ValueError('Archive name must be a ZIP filename')
    payload=(folder/'model.pt').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==card['model_sha256']
    dest=ROOT/'transfer/out/cv1-vision-v2.2.0';dest.mkdir(parents=True,exist_ok=True)
    target=dest/args.name
    parent=folder.parent
    files=[folder/'config.json',folder/'model.pt']
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in files:z.write(path,'models/vision/'+path.relative_to(parent).as_posix())
        z.writestr('models/vision/current.json',json.dumps({'folder':folder.name,'architecture':card['architecture'],
            'model_sha256':card['model_sha256']},ensure_ascii=False,indent=2))
        lines=['CV-1 Vision 2.2.0. Локальный классификатор без LLM/VLM.',
            'Нужен обновлённый код CV-1 с поддержкой схемы '+card['schema']+'. Старый EXE не получает новый код из архива модели.',
            'Распакуйте папку models рядом с исходным проектом CV-1; в EXE-версии выберите папку модели в настройках.',
            'Обученные группы: '+', '.join(card.get('class_names_ru',card.get('classes_ru',card['classes'])))+'.',
            'Фации, зернистость, биотурбация и гидродинамический режим не обучены.']
        if card.get('noncommercial_only'):lines.append('Только некоммерческие исследования: в обучении использованы данные DCID, CC BY-NC 4.0.')
        sources=card.get('sources') or [{'name':'Mendeley Lithology, Alzubaidi et al.','url':card.get('source_dataset'),'license':card.get('source_license')}]
        lines+=['Источники и лицензии: '+json.dumps(sources,ensure_ascii=False),
            'ImageNet initialization: torchvision official '+card['architecture']+' weights.',
            'Состав выборок и ограничения проверки смотрите в config.json и отчёте.',
            'Все автоматические результаты являются черновиками для проверки геологом.']
        z.writestr('CV1_Vision_README_RU.txt','\n'.join(lines)+'\n')
    with zipfile.ZipFile(target) as z:
        assert hashlib.sha256(z.read('models/vision/'+folder.name+'/model.pt')).hexdigest()==card['model_sha256']
    result={'archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'architecture':card['architecture'],'generative':False,'source_images_included':False}
    (dest/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
