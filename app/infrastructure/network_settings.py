"""Offline configuration: validation must never contact a network share."""
import ntpath
import re


def normalize_hub_path(value):
    value = str(value or '').strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
    if not value:
        return ''
    value = value.replace('/', '\\')
    if any(ord(char) < 32 for char in value) or any(char in value for char in '*?<>|"'):
        raise ValueError('В пути есть недопустимые символы.')
    if value.startswith('\\\\'):
        parts = value[2:].split('\\')
        if len(parts) < 2 or not parts[0] or not parts[1] or parts[0] in ('.', '?') or ':' in value:
            raise ValueError('Укажите сервер и общую папку, например \\\\SERVER\\CV1.')
    elif not re.match(r'^[A-Za-z]:\\', value) or ':' in value[2:]:
        raise ValueError('Укажите полный сетевой путь или папку на подключённом диске, например Z:\\CV1.')
    return ntpath.normpath(value)


def network_enabled(settings):
    return bool(str(settings.value('network/hub', '') or '').strip()) and str(
        settings.value('network/enabled', 'true')).lower() != 'false'
