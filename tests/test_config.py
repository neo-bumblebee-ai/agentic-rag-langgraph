"""Tests for configuration module."""

from src.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.ollama_model == "llama3.2"
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200
    assert cfg.top_k_retrieval == 10
    assert cfg.top_k_final == 5
    assert cfg.max_iterations == 3


def test_config_custom():
    cfg = Config(ollama_model="mistral", chunk_size=500, top_k_final=3)
    assert cfg.ollama_model == "mistral"
    assert cfg.chunk_size == 500
    assert cfg.top_k_final == 3
