"""Temporarily replace a model activation with an SAE reconstruction."""
from contextlib import contextmanager


from sae_pipeline.hooks.attention import head_count, output_projection
from sae_pipeline.hooks.extractor import _resolve_block, _resolve_module


@contextmanager
def patch_sae(model, sae, spec, site, *, input_scale=1.0, input_center=None):
    if spec.kind in {"resid_pre", "resid_post"}:
        module = _resolve_block(model, spec.layer)
    else:
        module = _resolve_module(model, site.module_path)
    pre = spec.kind in {"resid_pre", "attn_out_prelinear", "attn_head"}
    heads = None
    if spec.kind in {"attn_out_prelinear", "attn_head"}:
        heads = (site.extra or {}).get("num_heads") or head_count(module)
        module = output_projection(module)

    def reconstruct(hidden):
        original = hidden
        if spec.kind == "attn_head":
            if not heads or spec.head_idx is None or spec.head_idx >= heads or hidden.shape[-1] % heads:
                raise ValueError("Cannot resolve attention head for SAE patch")
            width = hidden.shape[-1] // heads
            start = spec.head_idx * width
            hidden = hidden[..., start:start + width]
        dtype = next(sae.parameters()).dtype
        normalized = hidden.to(dtype=dtype) * input_scale
        if input_center is not None:
            normalized = normalized - input_center
        restored = sae.decode(sae.encode(normalized))
        if input_center is not None:
            restored = restored + input_center
        restored = (restored / input_scale).to(dtype=hidden.dtype)
        if spec.kind == "attn_head":
            result = original.clone()
            result[..., start:start + width] = restored
            return result
        return restored

    def before(_module, args, kwargs):
        if args:
            return (reconstruct(args[0]), *args[1:]), kwargs
        key = "hidden_states" if "hidden_states" in kwargs else "input"
        return args, {**kwargs, key: reconstruct(kwargs[key])}

    def after(_module, _args, output):
        if isinstance(output, tuple):
            return (reconstruct(output[0]), *output[1:])
        return reconstruct(output)

    handle = module.register_forward_pre_hook(before, with_kwargs=True) if pre else module.register_forward_hook(after)
    try:
        yield
    finally:
        handle.remove()
