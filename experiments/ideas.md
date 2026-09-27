# Adapter improvement ideas

## Multi-layer DA3 features

**Motivation:** `extract_da3_tokens` currently uses only the final ViT-L backbone layer (`da3_source_layer_index: 3`). This is the most semantic layer. Earlier layers retain more spatial/geometric structure, which may be more useful for predicting z_star — a geometric token. Giving the Q-Former access to multiple layers lets it attend to both low-level geometry and high-level semantics.

**Approach:** concatenate tokens from multiple layers along the token dimension (not the feature dimension). All ViT layers produce the same number of tokens per frame (fixed by patch size and image resolution), and the Q-Former's cross-attention already handles variable-length source sequences, so no architectural change is needed.

**Code change** — `extract_da3_tokens` in `experiments/overfit_8frames/multi_scene_train.py`:

```python
# layer_idx: int | list[int]
indices = [layer_idx] if isinstance(layer_idx, int) else layer_idx
parts = []
for i in indices:
    raw = backbone_out[i][0].float()                           # (N_frames, H*W, 2048)
    flat = raw.reshape(raw.shape[0], -1, raw.shape[-1])
    per_frame = select_tokens(flat, max_tokens)                # (N_frames, max_tokens, 2048)
    parts.append(per_frame.reshape(1, -1, per_frame.shape[-1]))
return torch.cat(parts, dim=1).cpu()                           # (1, N_layers × N_frames × max_tokens, 2048)
```

Cache script `--da3-layer` arg changes from `type=int` to `type=int, nargs="+"`.

**What stays the same:** `source_dim=2048`, adapter weights, all other config. No architecture change, no adapter retraining from scratch — only re-caching.

**Cost:** Q-Former cross-attention scales as O(Q × S). With `layers=[1,2,3]` and `max_tokens=2048`, S grows from 16384 to 49152 (3×). If memory is tight, halve `max_tokens` per layer to keep total S unchanged.

**Suggested starting point:** `layers=[2, 3]` — adds one earlier layer without tripling cache size.
