# Code audit fix ledger

The audit findings are addressed in focused commits, each including relevant
regression coverage. Unsupported execution paths now reject incompatible
requests explicitly rather than producing misleading results.

| Finding | Resolution | Commit | Regression coverage |
| --- | --- | --- | --- |
| Streaming weights and token cursor were saved at different boundaries; sites could resume from different forwards | Publish uniquely named pairs for every site with one atomic cursor marker; select committed pairs; protect previous pairs until publication; trim uncommitted log rows | `9ac2275` | Mid-forward checkpoint interval, failure between site saves and before cursor publication, retention of one pair, exact resumed/uninterrupted weight equality |
| Local sweep worker could not be pickled | Threads supervise subprocesses | `18057a7` | Concurrent local execution |
| Cache/train/evaluation jobs raced and GPU assignments overlapped | Phase barriers, failure propagation and exclusive GPU leases | `18057a7` | Dependency ordering, failure blocks training, exclusive leases |
| Attention prelinear hook returned projected output | Hook projection input; expose projected output as `attn_out`; keep existing example checkpoint slugs with corrected component | `9de1ef5` | Different pre/post dimensions and BF16 HF/worker parity at lengths 8, 128 and 2048 |
| Hybrid mixers and root-level layer paths were misclassified | Classify mixer types and recognize root layer paths | `9de1ef5` | Synthetic Mamba, attention and MoE topology |
| Cache reuse ignored extraction identity | Fingerprint hook path, model settings and backend; compare dtype and shuffle seed | `fb19941` | Hook, backend, dtype and quantization changes reject reuse |
| Residual reset candidate count could exceed available rows; one-row quantile could be invalid | Bound reset count and inferred frequency | `db75eb4` | Single candidate and small pool |
| Centering metadata could silently be stale | Validate cache signature, scale and sample count | `5492305` | Same dimension with changed scale, sample or recipe |
| Disabled early stopping adapted LR; cached objective mode silently used stability | Gate plateau adaptation; reject unsupported cached objective mode | `55e11eb` | Repeated plateau with stopping disabled; rejection before I/O |
| Streaming canonical and best weights lacked normalization metadata | Persist normalization and validate resumed scale | `5920e61` | Site and best metadata, standalone resume, changed scale rejection |
| Square raw decoders were not transposed | Identify public layout from tensor keys | `4da434e` | Square raw and canonical functional reconstruction |
| Streaming token cap overshot within a forward | Stop at the exact minibatch cap and require batch alignment | `9ac2275` | Cap within forward; invalid fractional batch rejected |
| Explicit site sweep emitted `None` and lost custom slugs/hook paths | Add exact `--site` selection consistently and materialize validation jobs | `a91aa5e` | Explicit sweep and indexed/custom-site CLI selection |
| Keyword residual inputs were missed | Read `hidden_states` from keyword inputs | `9de1ef5` | Keyword-only residual capture |
| Shared experts, attention heads and expert indexes were inconsistent | Parse indexed components, discover and slice heads, preserve indexed cache validation; reject dense routed-expert caching | `9de1ef5`, `a91aa5e`, `fccdf8c` | Indexed parsing, head slices, shared-expert containers do not shadow children |
| Activation comparison could pass with missing sites or fail identical zeros | Require equal site keys by default; define two zero rows as identical | `edddb8c` | Missing-site opt-in and zero/nonzero cases |
| Plot regeneration looked outside evaluation partition directories | Share artifact path construction with evaluation | `811d640` | Partition, language, output ID and tag |
| Streaming ignored seed and L0 target warmup | Reuse cached seed and target scheduling helpers | `55e11eb` | Repeatable RNG and interpolated target |
| Configuration fields were unused | Honor centering directory and explicit reset frequency; reject deprecated dead-frequency setting | `5492305`, `db75eb4` | Preprocessing and reset configuration checks |
| float16 manifest disagreed with shard dtype | Use configured storage dtype | `fb19941` | Actual safetensors dtype |
| Feature-support budget could not exceed reconstruction budget | Accumulate independent windows | `50cdd1e` | Feature first fires after reconstruction window; counts and arrays verified |

## Compatibility and validation limits

- Existing incorrectly labelled attention checkpoints keep their historical
  slugs in the examples but select `attn_out` for projected output. Corrected
  prelinear hooks require newly extracted activations.
- Legacy cache manifests and cached preprocessing metadata lack the provenance
  required for safe reuse. Use a fresh cache run and metadata directory.
- Legacy streaming cursor files cannot establish a consistent snapshot and are
  rejected for exact resume. Use their weights and normalization as a warm start
  in a new run.
- Routed-expert cache materialization and cached objective early stopping are
  explicitly unsupported. Diagnostic expert extraction and streaming objective
  stopping remain available.
- vLLM prelinear capture requires tensor parallelism of one. Head and indexed
  expert extraction in vLLM fail explicitly rather than collecting a different
  site.
- Fill-backed runs retain the queue's documented eviction/skip behavior. Exact
  numerical replay requires the same input fills to remain available.
- Validation here uses CPU unit/integration tests and a synthetic worker harness.
  Real Nemotron/vLLM GPU execution and tensor-parallel gathering were not run.
  The local environment has CPU PyTorch 2.14.1, outside the pinned `<2.9` range.

## Final local verification

- Full CPU suite: **251 passed, 2 CUDA tests deselected**.
- All 54 source Python modules parsed successfully.
- Git whitespace checks passed.
- The user's pre-existing `.gitignore` edit was left unstaged and unchanged.

## Follow-up: explicit local warm-start paths (12 October 2026)

The final all-site job compared and revalidated its existing baselines but
failed before training because the local directory resolver could not find
an explicit nested raw export such as `L2_resid_post/sae_weights.safetensors`.
Directory resolution also parsed arbitrary export names as numbered steps.
Explicit relative filenames now use the direct-file resolver, which supports
raw exports and applies the existing companion-state requirements for resume.
Automatic latest-checkpoint discovery remains limited to the source directory.

Checkpoint, streaming CLI and streaming trainer regressions: **54 passed**,
including nested warm starts, arbitrary raw-export names, resume companions
and explicit relative-path validation. The failed job and its 14 preserved
baseline exports remain unchanged. GPU execution of this follow-up is pending.
