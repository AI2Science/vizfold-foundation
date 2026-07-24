"""
ESMFold backend: inference and trace export via HuggingFace Transformers.
"""
# Lazy import so schema/ can be used without requiring torch/fair-esm
def __getattr__(name):
    if name == "ESMFoldRunner":
        from vizfold.backends.esmfold.inference import ESMFoldRunner
        return ESMFoldRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ESMFoldRunner"]
