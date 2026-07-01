def test_public_api_exports():
    import pyllm.nn as nn

    expected = [
        "Module", "Linear", "Embedding", "LayerNorm", "Dropout",
        "Head", "MultiHeadAttention", "FeedForward", "TransformerBlock",
        "softmax", "cross_entropy", "gelu", "embedding", "concat",
    ]
    for name in expected:
        assert hasattr(nn, name), f"pyllm.nn is missing {name}"
