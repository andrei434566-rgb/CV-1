"""Validate actual model strings; never complete missing geological fields for it."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def description_budget(request, *, batch=False, retry=False):
    default = 768 if retry else 448 if request.get('primary_reference') else 384 if batch else 512
    value = request.get('max_new_tokens', request.get('max_tokens', default))
    try:
        value = int(value) if not isinstance(value, bool) else default
    except (ValueError, TypeError, OverflowError):
        value = default
    return max(default if retry else 256, min(value, 1024))


def validate_description(result, required):
    supplied = result.pop('fields', None)
    answer = str(result.get('answer') or '').strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', answer, flags=re.I)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned).strip()
    fields = supplied if 'answer' not in result else None
    try:
        # Tolerate a prose introduction/fence around a complete object, but
        # never add a closing brace, missing value, or a guessed property.
        start = cleaned.find('{')
        if start >= 0:
            fields, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except (ValueError, TypeError):
        pass
    missing = [key for key in required if not isinstance(fields, dict) or key not in fields]
    invalid = [key for key in required if isinstance(fields, dict) and key in fields
               and (not isinstance(fields[key], str) or not fields[key].strip())]
    code = ('token_limit' if result.get('incomplete') else 'invalid_json' if not isinstance(fields, dict)
            else 'missing_fields' if missing else 'invalid_fields' if invalid else '')
    diagnostics = dict(result.get('diagnostics') or {})
    diagnostics.update(validation_error=code, missing_fields=missing, invalid_fields=invalid,
                       required_fields=list(required))
    result['diagnostics'] = diagnostics
    result['ok'] = not code
    if code:
        result['recoverable'] = True
        reason = {'token_limit': 'Ответ остановлен по лимиту токенов.',
                  'invalid_json': 'Модель не вернула законченный JSON описания.',
                  'missing_fields': 'В ответе отсутствуют поля: ' + ', '.join(missing) + '.',
                  'invalid_fields': 'Пустые или некорректные поля: ' + ', '.join(invalid) + '.'}[code]
        result['error'] = reason
    else:
        result.pop('recoverable', None)
        result.pop('error', None)
        result['fields'] = {key: fields[key].strip() for key in required}
    return result


def record_description_diagnostics(result, folder):
    """Keep bounded local evidence for failures/recovery, without photo bytes."""
    entries = result.get('items') if isinstance(result.get('items'), dict) else {None: result}
    rows = []
    for item_id, value in entries.items():
        if not isinstance(value, dict):
            continue
        diagnostics = value.get('diagnostics') or {}
        if value.get('ok', True) and not value.get('incomplete') and not diagnostics.get('recovery'):
            continue
        rows.append({'at': datetime.now(timezone.utc).isoformat(), 'request_id': result.get('id'),
                     'item_id': item_id, 'ok': value.get('ok'), 'error': value.get('error'),
                     'answer': str(value.get('answer') or '')[:6000], 'diagnostics': diagnostics,
                     'timing': value.get('timing'), 'model_info': value.get('model_info')})
    if not rows:
        return
    try:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        path = folder/'vlm_diagnostics.jsonl'
        if path.exists() and path.stat().st_size > 2*1024*1024:
            path.replace(folder/'vlm_diagnostics.previous.jsonl')
        with path.open('a', encoding='utf8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    except OSError:
        pass  # A read-only disk must not discard a model reply.
