import hashlib
import json
from pathlib import Path

folder = Path(__file__).parent
source = folder / 'training_positive_extension.json'
dest = folder / 'training_positive_extension_v2.json'
if dest.exists():
    raise SystemExit('Refusing to overwrite')
data = json.loads(source.read_text(encoding='utf-8'))
before = json.loads(source.read_text(encoding='utf-8'))
old = 'В исследованном разрезе Panther Tongue береговые отложения связаны с региональной геологической обстановкой.'
new = 'В региональной последовательности дельтовый Panther Tongue перекрыт Mancos Shale и затем береговыми отложениями Storrs и Spring Canyon; интерпретацию сопоставляют с геологическим контекстом.'
for c in data['cards']:
    if c['id'] in {f'facies_positive_shore_0{i}' for i in range(1, 4)}:
        assert old in c['support_text']
        c['support_text'] = c['support_text'].replace(old, new)
    if c['id'] == 'facies_positive_tide_03':
        old_tail = 'независимые свидетельства изменения направления или режима течения'
        assert old_tail in c['answer']
        c['answer'] = c['answer'].replace(old_tail, 'независимые признаки приливного влияния')
changed = {}
for a, b in zip(before['cards'], data['cards']):
    assert a['id'] == b['id']
    fields = [k for k in a if a[k] != b[k]]
    if fields: changed[a['id']] = fields
assert changed == {'facies_positive_tide_03':['answer'], 'facies_positive_shore_01':['support_text'], 'facies_positive_shore_02':['support_text'], 'facies_positive_shore_03':['support_text']}
payload = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
dest.write_bytes(payload)
print('sha256:', hashlib.sha256(payload).hexdigest())
print('changed:', changed)
print('max answer:', max(len(c['answer'].split()) for c in data['cards']))
print('max support:', max(len(c['support_text'].split()) for c in data['cards']))
