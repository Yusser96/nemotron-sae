"""Test topology classifier on a synthetic module tree.

We mock a Nemotron-shaped model: model.layers[L].{block_sparse_moe.experts[E], mamba, self_attn}.
Real model has a custom trust_remote_code path; this test just confirms the regex
classifier picks up modules that match the expected naming conventions.
"""

import torch.nn as nn

from sae_pipeline.model.topology import enumerate_hooks


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)


class _MoE(nn.Module):
    def __init__(self, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(n_experts)])
        self.shared_expert = _Expert()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = nn.Linear(8, 8)
        self.self_attn = nn.Linear(8, 8)
        self.block_sparse_moe = _MoE()


class _Model(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])


def test_classifier_picks_up_experts_and_residuals():
    m = _Model(n_layers=3)

    # Wrap with .model.layers... shape so the regex finds layers.
    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = m

    full = _Wrap()
    sites = enumerate_hooks(full)

    # 3 layers × 4 experts = 12 expert sites, plus shared experts.
    by_kind: dict[str, int] = {}
    for s in sites:
        by_kind.setdefault(s.component_kind, 0)
        by_kind[s.component_kind] += 1

    assert by_kind.get("expert", 0) == 12
    assert by_kind.get("mamba_out", 0) == 3
    assert by_kind.get("attn_out_prelinear", 0) == 3
    assert by_kind.get("moe_out", 0) == 3
    assert by_kind.get("resid_pre", 0) == 3
    assert by_kind.get("resid_post", 0) == 3
