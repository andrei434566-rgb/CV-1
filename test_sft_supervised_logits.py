"""CPU-only parity checks for memory-saving supervised causal LM loss."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from scripts.train_qwen_source_pilot import sft_supervised_token_loss


class TinySelectedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.raw = torch.nn.Parameter(logits.clone())
        self.positions = None

    def forward(self, input_ids, logits_to_keep, use_cache):
        self.positions = logits_to_keep.detach().tolist()
        return SimpleNamespace(logits=self.raw.index_select(1, logits_to_keep))


class SupervisedLogitsTests(unittest.TestCase):
    def _compare(self, divisor):
        torch.manual_seed(1)
        initial = torch.randn(2, 7, 13)
        # Unequal target masks exercise union of positions, including EOS.
        labels = torch.tensor([[-100, -100, -100, 2, 5, 3, 4],
                               [-100, -100, 8, -100, 6, 1, -100]])
        tiny = TinySelectedModel(initial)
        loss, _ = sft_supervised_token_loss(tiny, {"input_ids": torch.zeros_like(labels), "labels": labels},
                                             num_items_in_batch=divisor)
        loss.backward()

        full = initial.clone().requires_grad_()
        targets = F.pad(labels, (0, 1), value=-100)[:, 1:].contiguous()
        reference = F.cross_entropy(full.float().reshape(-1, 13), targets.reshape(-1), ignore_index=-100,
                                    reduction="sum" if divisor is not None else "mean")
        if divisor is not None:
            reference = reference / divisor
        reference.backward()
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(tiny.raw.grad, full.grad, rtol=1e-6, atol=1e-7))
        self.assertEqual(tiny.positions, [1, 2, 3, 4, 5])

    def test_loss_and_gradients_match_hf_token_mean(self):
        self._compare(None)

    def test_loss_and_gradients_match_accumulated_token_normalization(self):
        self._compare(torch.tensor(27))

    def test_unmasked_first_token_rejected(self):
        tiny = TinySelectedModel(torch.randn(1, 3, 5))
        with self.assertRaises(ValueError):
            sft_supervised_token_loss(tiny, {"input_ids": torch.tensor([[1, 2, 3]]),
                                              "labels": torch.tensor([[1, 2, 3]])})


if __name__ == "__main__":
    unittest.main()
