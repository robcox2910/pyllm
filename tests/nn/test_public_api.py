"""Tests for public api."""


def test_public_api_exports():
    """Public api exports."""
    from pyllm import nn

    expected = [
        "Module",
        "Linear",
        "Embedding",
        "LayerNorm",
        "Dropout",
        "Head",
        "MultiHeadAttention",
        "FeedForward",
        "TransformerBlock",
        "softmax",
        "cross_entropy",
        "gelu",
        "embedding",
        "concat",
    ]
    for name in expected:
        assert hasattr(nn, name), f"pyllm.nn is missing {name}"
