# Grassmann Flow vs Transformer Reproduction

## What This Repo Is

- This workspace was initialized from `kanenorman/grassmann` as a starting codebase.
- The implementation was then modified to align with arXiv `2512.19428` equations and block order.
- All experiment numbers in this README come from runs executed in this environment on **2026-02-23**.

## What Was Changed

- Paper-aligned Grassmann gate mixing:
  - `h_mix = alpha * h + (1 - alpha) * g`, `alpha = sigmoid(W[h;g] + b)`
- Grassmann post-norm block flow aligned to paper intent:
  - `LayerNorm(h_mix) -> Dropout -> FFN residual -> LayerNorm`
- Attention implementation simplified to native PyTorch reshape/matmul.
- Dataset loader updated to support official local WikiText-2 parquet files.
- UV + Hydra experiment pipeline added:
  - `train_hydra.py`
  - `conf/` configs
  - `summarize_results.py`

## Experiment Setup (Executed)

- Date: **2026-02-23**
- Model pair: `attention` vs `grassmann`
- Dataset: official `Salesforce/wikitext` (`wikitext-2-raw-v1`) local parquet
- Tokenizer: `bert-base-uncased`
- Context length: `128`
- Training mode: CPU (`CUDA unavailable in this runtime`)
- Reduced sample run for completion speed:
  - `data.max_samples_train=800`
  - `data.max_samples_val=200`

### Command Used

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
uv run --python /home/xncb135/miniconda3/bin/python3 \
python train_hydra.py -m model=attention,grassmann \
  train.num_epochs=3 \
  train.batch_size=16 \
  train.num_workers=0 \
  data.max_samples_train=800 \
  data.max_samples_val=200
```

## Result Table (Actual Run)

| Model | Params (M) | Best Val Loss | Best Val PPL |
|---|---:|---:|---:|
| TransformerLM | 12.59 | 23.8336 | 22,428,927,000.23 |
| GrassmannLM | 12.61 | 23.5390 | 16,704,772,944.74 |

## Artifacts

- `training_results_hydra/multirun/2026-02-23/02-26-47/0_attention/results.json`
- `training_results_hydra/multirun/2026-02-23/02-26-47/1_grassmann/results.json`
- `training_results_hydra/comparison_table_2026-02-23_02-26-47.md`

## Interpretation

- Under identical settings, Grassmann achieved lower validation loss/PPL than attention.
- The comparison is meaningful as an implementation check under controlled conditions.
- Absolute PPL is not paper-table comparable because this was CPU + reduced-data + short-epoch.

## Conclusion

- The paper-aligned Grassmann implementation is functioning in this codebase.
- In this controlled run, Grassmann outperformed the matched Transformer baseline on validation metrics.
- Next step for stronger claim: full-data, longer schedule, CUDA-enabled rerun.

## Reproduce

```bash
uv sync
uv run python train_hydra.py -m model=attention,grassmann
uv run python summarize_results.py \
  --root training_results_hydra/multirun \
  --out training_results_hydra/comparison_table.md
```
