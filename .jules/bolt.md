# Bolt's Journal - Critical Learnings Only

## 2026-11-03 - Deduplicate Historical Context Tokenizer Encoding in Monte Carlo Paths
**Learning:** In Monte Carlo forecasting where `sample_count > 1`, the historical candlestick conditioning series (`x_norm`) is 100% identical across all sample paths. Passing a tensor repeated across `sample_count` into `KronosTokenizer.encode()` runs the multi-layer neural network transformer encoder redundantly `sample_count` times on CPU, taking up to 1,380ms for 20 samples.
**Action:** Always encode historical context once with `batch_size=1` (`~99ms`), then broadcast/repeat the resulting discrete integer token IDs `[x_token[0].repeat(sample_count, 1), x_token[1].repeat(sample_count, 1)]`. This is bit-for-bit identical and achieves a 4x–14x speedup on tokenizer encoding.
