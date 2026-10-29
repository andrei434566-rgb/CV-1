"""Constrain rock names without masking the vocabulary for free observations.

The full JSON response is validated after generation. Only the lithology value
is constrained here, leaving every other field's logits untouched.
"""
import re
from functools import lru_cache


def lithology_logits_processor(tokenizer, labels, prompt_length):
    import torch

    labels = tuple(labels)

    @lru_cache(maxsize=256)
    def next_tokens(tail):
        if re.fullmatch(r'\s*:?\s*', tail):
            separator = '"' if ':' in tail else ': "'
            continuations = [separator + label + '"' for label in labels]
        else:
            match = re.fullmatch(r'\s*:\s*"([^"\\]*)', tail)
            if match is None:
                return None
            prefix = match.group(1)
            continuations = [label[len(prefix):] + '"' for label in labels if label.startswith(prefix)]
        if not continuations:
            raise ValueError("Модель начала название породы вне справочника; повторите описание.")
        return tuple({tokenizer.encode(text, add_special_tokens=False)[0] for text in continuations})

    class RockVocabularyProcessor:
        def __call__(self, input_ids, scores):
            for row, ids in enumerate(input_ids[:, prompt_length:].tolist()):
                text = tokenizer.decode(ids, skip_special_tokens=True)
                if '"lithology"' not in text:
                    continue
                allowed = next_tokens(text.rsplit('"lithology"', 1)[1])
                if allowed is not None:
                    original = scores[row, list(allowed)].clone()
                    scores[row].fill_(-torch.inf)
                    scores[row, list(allowed)] = original
            return scores

    return RockVocabularyProcessor()
