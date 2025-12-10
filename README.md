# ESE 3060 Final Project Part 2

## Quick Start — Best Current Run
Full-length run with the best-performing config: elementwise SDPA-output gating at higher LR.
```bash
cd /workspace/ese-3060-project
source .venv/bin/activate    # or your env

# Data: expects FineWeb bins at /workspace/fineweb10B; symlink if needed
mkdir -p data
ln -s /workspace/fineweb10B data/fineweb10B   # adjust if your data lives elsewhere

export ATTNGATE=elementwise
export GATEPOS=sdpa
export GATEACT=sigmoid
export LR=0.00468
export NUM_ITERATIONS=5100
export WARMDOWN_ITERS=1450
# optional: SEED, VAL_EVERY, WARMUP_ITERS, etc.

torchrun --standalone --nproc_per_node=8 train_gpt.py
```
Logs land in `logs/<run_id>.txt`, summary row in `experiments/results.csv`.

## Repo Layout
- `train_gpt.py` — NanoGPT-style transformer with SDPA-output gating options and logging.
- `cached_fineweb10B.py` — download GPT-2 tokenized FineWeb10B shards from HF into `/workspace/fineweb10B`.
- `scripts/run_baseline.sh` — baseline launcher; `scripts/split_results.py` — splits results into stage files.
- `notebooks/` — launchers/EDA for stages 1, 2, 2.5, 3, and final figures.
- `experiments/` — aggregated results CSVs and parsed curves.
- `logs/` — per-run logs (code + metrics).

## Data Prep
```bash
cd /workspace/ese-3060-project
python cached_fineweb10B.py 9        # first 9 chunks (~900M tokens) for quick starts; omit arg for full set
mkdir -p data
ln -s /workspace/fineweb10B data/fineweb10B
```
If your data lives elsewhere, set `input_bin`/`input_val_bin` in `train_gpt.py` or adjust the symlink.

## How to Launch Other Configs
Use env vars to override defaults (no code edits needed):
- `ATTNGATE` = `none|headwise|elementwise|const`
- `GATEPOS` = `sdpa|value`
- `GATEACT` = `sigmoid|ns_sigmoid`
- `LR`, `SEED`, `NUM_ITERATIONS`, `WARMUP_ITERS`, `WARMDOWN_ITERS`, `VAL_EVERY`

Example baseline (no gate) short run:
```bash
ATTNGATE=none NUM_ITERATIONS=1500 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiment Summary (from `ESE_3060_Final_Project (1).pdf`)
All runs use the provided data pipeline; no data-order changes; torch.compile enabled.

### Stage 1 (1500 iters @ LR=0.0036, 2 seeds)
- Elementwise/sigmoid SDPA gating is best: ~0.5% lower val loss than baseline.
- Ordering matches paper: elementwise > headwise; sigmoid > ns-sigmoid; const slightly worse than baseline.

### Stage 2 (800 iters, LR sweep {0.0036, 0.00396, 0.00432, 0.00468}, 2 seeds)
- Higher LR helps both baseline and gating.
- Gating adds ~6% step-time overhead; early-loss advantage small at 800 steps.

### Stage 2.5 (1500 iters @ LR=0.00468, 2 seeds)
- Elementwise gate: final val loss ≈ 3.5099 vs baseline 3.5298 (~0.6% better) at same steps.
- Wall-clock advantage small; step time still ~6% slower.

### Stage 3 (5100 iters full runs, 3 seeds each @ LR=0.00468)
- Elementwise gate: mean final val loss ≈ 3.2723 vs tuned baseline 3.2927 (~0.6% better).
- Cost: ~6% longer wall-clock; gains are accuracy at fixed steps, not raw speed.
- One reference run of original baseline @ LR=0.0036: ~3.2938.

## Files to Check
- Results: `experiments/results.csv` and stage splits (`results_stage1.csv`, `results_stage2.csv`, `results_stage2_5.csv`, etc.).
- Curves: `experiments/log_curves_*.csv`.
- Logs: `logs/<run_id>.txt` (includes git hash, args, nvidia-smi, val/train traces).

## Constraints & Notes
- Do not change data ordering or add extra torch.compile flags beyond the baseline settings.
- Muon optimizer expects only 2D params in `transformer.h`; gating layers are bias-free/2D to stay Muon-safe.
- torch.compile adds ~5–10 minutes on first run (CPU-side graph build) while GPU memory is already allocated.

