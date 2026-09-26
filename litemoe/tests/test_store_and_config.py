"""make_store dispatch + directory support; config loading; cache factory."""
from __future__ import annotations

import pytest

from litemoe.cache.lru import LRUCache, PerLayerLRUCache
from litemoe.config import LitemoeConfig, ModelConfig, parse_dtype
from litemoe.interface import make_cache
from litemoe.model.loader import ModelMeta
from litemoe.store import make_store
from tests.conftest import FakeStore


def test_make_store_directory(tmp_path):
    """A directory with a .safetensors resolves to the safetensors backend.

    We only verify *dispatch* (dir -> file -> SafetensorsExpertStore), not a
    real load: an empty file fails inside the loader, which is fine — the point
    is it must NOT be rejected as an "unknown checkpoint extension".
    """
    (tmp_path / "model.safetensors").write_bytes(b"")
    try:
        make_store(str(tmp_path))
    except ValueError as e:
        if "unknown checkpoint extension" in str(e):
            pytest.fail(f"directory path was not resolved to a file: {e}")
        raise  # any other error means dispatch reached the backend
    except Exception:
        pass  # loader error on empty file == dispatch succeeded


def test_make_store_missing_dir(tmp_path):
    """An existing EMPTY directory -> FileNotFoundError (no *.safetensors)."""
    d = tmp_path / "empty-dir"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        make_store(str(d))


def test_model_config_kind():
    assert ModelConfig(path="/x/y.gguf").kind == "gguf"
    assert ModelConfig(path="/x/y.safetensors").kind == "safetensors"
    assert ModelConfig(path="/x/y.st").kind == "safetensors"


def test_config_from_dict_roundtrip(tmp_path):
    d = {
        "model": {"path": "/x/m.safetensors", "tokenizer": None},
        "compute": {"dtype": "bf16", "device": "cpu"},
        "cache": {"strategy": "per_layer_lru", "max_experts": 4,
                  "predict_window": 8, "unknown_key": 1},
        "metrics": {"log_activations": False,
                    "activations_file": "/tmp/a.jsonl"},
        "prompt": "hi", "max_new_tokens": 12, "seed": 7,
    }
    cfg = LitemoeConfig.from_dict(d)
    assert cfg.model.path == "/x/m.safetensors"
    assert cfg.compute.dtype == "bf16"
    assert cfg.cache.strategy == "per_layer_lru"
    assert cfg.cache.max_experts == 4
    assert cfg.metrics.log_activations is False
    assert cfg.max_new_tokens == 12 and cfg.seed == 7
    # unknown keys are dropped, not fatal
    p = tmp_path / "c.json"
    cfg.to_file(str(p))
    cfg2 = LitemoeConfig.from_file(str(p))
    assert cfg2.to_dict() == cfg.to_dict()


def test_parse_dtype():
    import torch
    assert parse_dtype("fp16") is torch.float16
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("fp32") is torch.float32
    with pytest.raises(ValueError):
        parse_dtype("fp8")


def test_make_cache_strategies(fake_store):
    lru = make_cache(fake_store, strategy="lru", max_experts=4, device="cpu")
    assert isinstance(lru, LRUCache) and lru.max_experts == 4

    pde = make_cache(fake_store, strategy="pde", max_experts=4, device="cpu")
    # PDECache is a P0 placeholder: must exist and expose the ExpertCache surface
    assert hasattr(pde, "get") and hasattr(pde, "stats")

    per = make_cache(fake_store, strategy="per_layer_lru", max_experts=2,
                     device="cpu")
    assert isinstance(per, PerLayerLRUCache)

    with pytest.raises(ValueError, match="unknown cache strategy"):
        make_cache(fake_store, strategy="fifo")
