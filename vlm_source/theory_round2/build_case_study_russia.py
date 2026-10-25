"""Short attributed textual case study; images remain unpaired inventory candidates."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE_ID='tpu_yamal_nedolivko_perevertaylo_2025'
PDF_URL='https://izvestiya.tpu.ru/archive/article/download/4976/3406/20807'
POLICY_URL='https://izvestiya.tpu.ru/archive/editorial-policy'
PDF_SHA='ae5fc6def7153e9df30f67b91de149f699ffbaced406349d6424375c76d76d0f'
SOURCE={
 'source_id':SOURCE_ID,
 'title':'Недоливко Н.М., Перевертайло Т.Г. (2025). Особенности осадконакопления терригенных отложений в позднем апте – раннем альбе на севере Ямала',
 'journal':'Известия Томского политехнического университета. Инжиниринг георесурсов',
 'volume':336,'issue':3,'pages':'170–182','doi':'10.18799/24131830/2025/3/4976',
 'url':PDF_URL,'landing_url':'https://izvestiya.tpu.ru/archive/article/view/4976',
 'accessed_date':'2026-10-25', 'pdf_file':'tpu_yamal_2025_4976.pdf','pdf_sha256':PDF_SHA,
 'access':'Read public full article and inspected rendered original Figure 5 on printed page 177/PDF page 8.',
 'license':'CC BY-NC 4.0, stated in the journal editorial policy, not in an article-specific notice in the PDF.',
 'license_basis_url':POLICY_URL,'license_url':'https://creativecommons.org/licenses/by-nc/4.0/',
 'license_verification':'Read official journal policy and followed its CC BY-NC link to version 4.0. No PDF-specific license notice found.',
 'use_scope':'Noncommercial research with author/source attribution and indication of paraphrasing; no commercial permission inferred.',
 'geology_approved':False,
 'limitations':['Case-specific author observations and interpretation; no universal facies label.',
               'Figure panels have no verified crop/depth assignment for training.',
               'Panel D image annotation and caption disagree; English wording for E is unreliable.',
               'The case-study holdout shares its article and concepts with training; not independent knowledge accuracy.'],
}

ROWS=[
 ('sequence','train','В опубликованном разрезе ХМ3 на севере Ямала как меняются породы вверх?',
  'По описанию авторов, глинистая нижняя часть сменяется алеврито-песчаным переслаиванием и песчаниками; верх включает глинисто-сидеритовые осадки.',
  'Printed pages 177–178/PDF pages 8–9, KhM3 section; Figure 5 A–L'),
 ('figure5b','train','В опубликованном разрезе ХМ3 что отмечено на панели B рисунка 5?',
  'Авторы описывают глинистый материал с алевритовыми прослоями, волнисто-линзовидной слоистостью и следами Planolites и Chondrites.',
  'Printed page 177/PDF page 8, Figure 5 panel B, Russian caption'),
 ('combined_evidence','train','В опубликованном разрезе ХМ3 почему интерпретация не сводилась к одному фото?',
  'Авторы совместили описание керна, ихнофациальные признаки и интерпретацию ГИС. Это исследование конкретной толщи, а не универсальное правило по снимку.',
  'Printed pages 171–172/PDF pages 2–3, Methods; pages 177–178, KhM3 discussion'),
 ('bio_reworking','holdout','В опубликованном разрезе ХМ3 могла ли первичная слоистость почти исчезнуть при биотурбации?',
  'Да. Авторы описывают участки, где биотурбация почти полностью переработала первичную слоистость. Это наблюдение относится к изученному разрезу.',
  'Printed page 177/PDF page 8, paragraph before Figure 5'),
]

PANEL_NOTES={
 'A':'Биотурбированный глинистый материал', 'B':'Волнисто-линзовидные прослои',
 'C':'Алевритовые линзы', 'D':'Взмучивание; спорная ихнометка',
 'E':'Пологоволнистое переслаивание', 'F':'Косоволнистое переслаивание',
 'G':'Перемешанный глинисто-алеврито-песчаный материал', 'H':'Косоволнистый песчаник',
 'I':'Косослоистый песчаник', 'J':'Песчаник; сидеритовые стяжения',
 'K':'Заполненные трещины усыхания', 'L':'Волнисто-линзовидное переслаивание',
}

def main():
 originals={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.jsonl','holdout.jsonl']}
 pdf=ROOT/SOURCE['pdf_file']
 if hashlib.sha256(pdf.read_bytes()).hexdigest()!=PDF_SHA: raise ValueError('Source PDF hash differs')
 records=[]
 for suffix,split,q,a,locator in ROWS:
  records.append({'conversations':[{'from':'human','value':q},{'from':'gpt','value':a}],
   'metadata':{
    'record_id':'russia2025_yamal_case_'+suffix, 'record_type':'text_only_knowledge',
    'topic':'published_russian_case_study', 'source_group':'russia_yamal_case_study_2025',
    'case_study_id':SOURCE_ID, 'split':split,'training_approved':split=='train',
    'geology_approved':False, 'target_language':'ru',
    'review_status':'user_requested_source_checked_case_study_not_geologist_reviewed',
    'approval_scope':'noncommercial_research_only',
    'scope':'Published case study only: author observations/interpretation of KhM3, northern Yamal; never a universal image/facies label.',
    'source_reuse':'Original short Russian paraphrase, not copied caption or image training pair.',
    'sources':[{'source_id':SOURCE_ID,'url':PDF_URL,'locator':locator,
                'license':'CC BY-NC 4.0 per journal editorial policy','license_basis_url':POLICY_URL}],
   }})
 derived_words=sum(len((r['conversations'][0]['value']+' '+r['conversations'][1]['value']).split()) for r in records)
 derived_words+=sum(len(v.split()) for v in PANEL_NOTES.values())
 if derived_words>200: raise ValueError(f'Short paraphrase budget exceeded: {derived_words}')
 for name,rows in [('case_study_russia',records),('case_study_russia_train',[r for r in records if r['metadata']['split']=='train']),
                   ('case_study_russia_holdout',[r for r in records if r['metadata']['split']=='holdout'])]:
  (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 (ROOT/'case_study_russia_sources.json').write_text(json.dumps({SOURCE_ID:SOURCE},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 inventory=[]
 for panel,note in PANEL_NOTES.items():
  flags=[]
  if panel=='D': flags.append('Photo annotation Tch conflicts with Russian caption Shaubcylindrichnus; no taxon label accepted.')
  if panel=='E': flags.append('Russian caption says gently wavy; English wording hollow layering is not used as a target.')
  inventory.append({'candidate_id':f'{SOURCE_ID}_fig5_{panel}','source_id':SOURCE_ID,'url':PDF_URL,
     'figure':5,'panel':panel,'printed_page':177,'pdf_page_1based':8,
     'author_caption_paraphrase_ru':note,'source_well':'Published anonymized well 1',
     'caption_language_used':'ru','visual_page_inspected':True,
     'status':'inventory_only_requires_manual_pair_review','training_approved':False,'geology_approved':False,
     'image_path':None,'crop_box':None,'depth_top':None,'depth_bottom':None,
     'scope':'Locator of author figure panel only, not accepted image ground truth.',
     'review_flags':flags})
 (ROOT/'case_study_russia_photo_inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory),encoding='utf-8')
 for name,before in originals.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=before: raise ValueError('Earlier block changed')
 summary={'train_rows':3,'holdout_rows':1,'photo_candidates':12,'image_training_pairs':0,
          'question_answer_and_inventory_words':derived_words,'license':'CC BY-NC 4.0 per journal policy',
          'unchanged_original_block_sha256':originals}
 (ROOT/'case_study_russia_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
