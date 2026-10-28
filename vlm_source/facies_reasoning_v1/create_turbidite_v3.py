import hashlib
import json
from pathlib import Path

folder = Path(__file__).parent
old = 'разжиженных турбидных потоков'
new = 'тонкозернистых относительно разбавленных турбидных течений'
jobs = [('training_reasoned_cards_v2.json', 'training_reasoned_cards_v3.json', 'facies_train_turbidite_', 5), ('training_positive_extension_v2.json', 'training_positive_extension_v3.json', 'facies_positive_turbidite_', 3)]
for src_name, dst_name, prefix, expected in jobs:
    src, dst = folder / src_name, folder / dst_name
    if dst.exists():
        raise SystemExit('Refusing to overwrite: ' + str(dst))
    data = json.loads(src.read_text(encoding='utf-8'))
    original = json.loads(src.read_text(encoding='utf-8'))
    changed = []
    for c in data['cards']:
        if c['id'].startswith(prefix):
            assert c['support_text'].count(old) == 1
            c['support_text'] = c['support_text'].replace(old, new)
            changed.append(c['id'])
    assert len(changed) == expected
    for a, b in zip(original['cards'], data['cards']):
        assert a['id'] == b['id']
        assert {k: v for k, v in a.items() if k != 'support_text'} == {k: v for k, v in b.items() if k != 'support_text'}
        if a['id'] not in changed:
            assert a == b
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    dst.write_bytes(payload)
    print(dst_name, hashlib.sha256(payload).hexdigest(), len(changed), 'max support', max(len(c['support_text'].split()) for c in data['cards']))
