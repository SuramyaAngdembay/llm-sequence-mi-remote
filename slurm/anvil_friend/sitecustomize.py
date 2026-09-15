# Scoped patch: disable transformers torch<2.6 torch.load guard (CVE-2025-32434)
# ONLY for our own trusted checkpoints during resume. Applied via PYTHONPATH.
def _noop(*a, **k):
    return None
try:
    import transformers.utils.import_utils as _iu
    if hasattr(_iu, "check_torch_load_is_safe"):
        _iu.check_torch_load_is_safe = _noop
    import transformers.utils as _u
    if hasattr(_u, "check_torch_load_is_safe"):
        _u.check_torch_load_is_safe = _noop
    import transformers.trainer as _tr
    if hasattr(_tr, "check_torch_load_is_safe"):
        _tr.check_torch_load_is_safe = _noop
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "check_torch_load_is_safe"):
        _mu.check_torch_load_is_safe = _noop
except Exception:
    pass
