"""Tests for config loading."""

import pytest
from leander_tts.config import DictConfig, load_config


class TestDictConfig:
    def test_attribute_access(self):
        cfg = DictConfig({"a": 1, "b": "hello"})
        assert cfg.a == 1
        assert cfg.b == "hello"

    def test_nested_access(self):
        cfg = DictConfig({"model": {"name": "test", "d_model": 64}})
        assert cfg.model.name == "test"
        assert cfg.model.d_model == 64

    def test_list_access(self):
        cfg = DictConfig({"items": [1, 2, 3]})
        assert cfg.items == [1, 2, 3]

    def test_get_with_default(self):
        cfg = DictConfig({"a": 1})
        assert cfg.get("a") == 1
        assert cfg.get("b", 42) == 42

    def test_contains(self):
        cfg = DictConfig({"a": 1})
        assert "a" in cfg
        assert "b" not in cfg


class TestLoadConfig:
    def test_load_model_config(self):
        cfg = load_config("configs/model/leander_m.yaml")
        assert cfg.model.name == "leander-m"
        assert cfg.model.lm.d_model == 1536
        assert cfg.model.codec.codebook_size == 4096

    def test_load_training_config(self):
        cfg = load_config("configs/training/phase3_lm.yaml")
        assert cfg.training.phase == 3
        assert cfg.training.bf16 is True

    def test_model_variants(self):
        """All model configs should load and have consistent structure."""
        for variant in ["leander_s", "leander_m", "leander_l"]:
            cfg = load_config(f"configs/model/{variant}.yaml")
            mc = cfg.model
            assert hasattr(mc, "lm")
            assert hasattr(mc, "codec")
            assert hasattr(mc, "speaker_encoder")
            assert hasattr(mc, "detokenizer")
            assert mc.codec.codebook_size == 4096
            assert mc.codec.n_levels == 3
