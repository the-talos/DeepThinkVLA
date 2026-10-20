import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from experiments import device as dev  # noqa: E402

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@contextmanager
def fake_host(xpu=False, cuda=False):
    with mock.patch.object(dev, "is_available", side_effect=lambda t: {"xpu": xpu, "cuda": cuda, "cpu": True}[t]):
        yield


@pytest.mark.parametrize(
    "xpu,cuda,expected",
    [(False, False, "cpu"), (False, True, "cuda:0"), (True, False, "xpu:0"), (True, True, "xpu:0")],
)
def test_default_device_follows_the_host(xpu, cuda, expected):
    with fake_host(xpu=xpu, cuda=cuda):
        assert str(dev.resolve_device(None)) == expected


def test_explicit_device_wins_over_detection():
    with fake_host(xpu=True, cuda=True):
        assert str(dev.resolve_device("cpu")) == "cpu"
        assert str(dev.resolve_device("cuda:1")) == "cuda:1"
        assert str(dev.resolve_device(torch.device("xpu:1"))) == "xpu:1"
    with fake_host():
        assert str(dev.resolve_device("xpu")) == "xpu:0"
        assert str(dev.resolve_device("cuda")) == "cuda:0"


def test_manual_seed_all_uses_the_detected_backend():
    with fake_host(xpu=True), mock.patch.object(torch.xpu, "manual_seed_all", create=True) as seed:
        dev.manual_seed_all(7)
        seed.assert_called_once_with(7)
    with fake_host(), mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
        dev.manual_seed_all(7)
        cuda_seed.assert_not_called()


def _load_with_mocked_checkpoint(device):
    utils = pytest.importorskip("experiments.deepthinkvla_utils")
    model = mock.MagicMock()
    model.eval.return_value = model
    cfg = types.SimpleNamespace(pretrained_checkpoint="ckpt", compute_dtype="bfloat16", device=device)
    with mock.patch.object(utils.DeepThinkVLA, "from_pretrained", return_value=model) as loader, mock.patch.object(
        utils, "_get_unomrmalize_action", return_value=None
    ):
        utils.get_vla(cfg)
    assert loader.call_args.kwargs["attn_implementation"] == "sdpa"
    return model.to.call_args.args[0]


def test_get_vla_places_the_model_on_the_resolved_device():
    with fake_host(xpu=True):
        assert str(_load_with_mocked_checkpoint(None)) == "xpu:0"
    with fake_host(cuda=True):
        assert str(_load_with_mocked_checkpoint(None)) == "cuda:0"
    with fake_host():
        assert str(_load_with_mocked_checkpoint(None)) == "cpu"


def test_get_vla_honours_an_explicit_device():
    with fake_host(xpu=True, cuda=True):
        assert str(_load_with_mocked_checkpoint("cpu")) == "cpu"


@pytest.mark.skipif(not HAS_XPU, reason="needs an Intel XPU")
def test_tiny_model_cpu_and_xpu_agree():
    modeling = pytest.importorskip("sft.modeling_deepthinkvla")
    from transformers import PaliGemmaConfig

    config = PaliGemmaConfig(
        text_config=dict(model_type="gemma", vocab_size=400, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                         num_image_tokens=4),
        vision_config=dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, image_size=28, patch_size=14, projection_dim=64, num_image_tokens=4),
        image_token_index=399, projection_dim=64, hidden_size=64, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    config.action_token_begin_idx, config.action_token_end_idx = 100, 195
    config.ignore_index, config.eos_token_id = -100, 1
    config.action_end_token_index, config.action_start_token_index = 398, 397
    config.think_start_token_index, config.think_end_token_index = 395, 396
    torch.manual_seed(0)
    model = modeling.DeepThinkVLA(config).to(torch.bfloat16).eval()
    model.prompt_end_token_id = [11, 12]
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    ids = torch.tensor([[0, 2] + [399] * 4 + [5, 6, 11, 12, 395, 7, 8]])
    pixels = torch.randn(1, 3, 28, 28).to(torch.bfloat16)
    mask = (ids != 0).long()

    def run(m, device):
        with torch.inference_mode():
            logits, start = m.prompt_cot_predict_action(
                input_cot_ids=ids.to(device), pixel_values=pixels.to(device), attention_mask=mask.to(device)
            )
        return logits.float().cpu(), int(start)

    cpu_logits, start = run(model, "cpu")
    xpu_logits, xpu_start = run(model.to("xpu"), "xpu")
    assert start == xpu_start
    assert cpu_logits.std() > 0
    torch.testing.assert_close(xpu_logits, cpu_logits, atol=0.2, rtol=0.05)
