# Baseline Run Setup (Stage 0)

Step-by-step to reproduce the ungated baseline on a fresh GPU VM.

## 1) Provision the VM
- OS: Ubuntu 22.04 or similar with CUDA-capable NVIDIA GPUs (≥8 if you want the default global batch of 512).  
- Example hardware: 8× A10G or 8× A100 40GB.  
- Attach fast local SSD (dataset is multi-GB).

## 2) System prep
```bash
sudo apt-get update
sudo apt-get install -y build-essential git python3.10 python3.10-venv python3-pip
# Install an NVIDIA driver (525+). If image includes CUDA toolkit/driver, keep it.
```

## 3) Clone repo and create env
```bash
git clone <your-fork-or-repo-url> ese-3060-project
cd ese-3060-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Ensure the PyTorch wheel matches your CUDA version (the requirements pin should pull a CUDA build).

## 4) Prepare data
- Place FineWeb train/val shards here:
  - `data/fineweb10B/fineweb_train_*.bin`
  - `data/fineweb10B/fineweb_val_*.bin`
- The loader expects the `.bin` header magic `20240520` and enough tokens for your batch/seq settings.

## 5) Optional: set baseline env vars
```bash
set -a
source .example.baseline.env   # keeps gating off, sets lr/seed defaults
set +a
```
You can override GPU count or run count when launching (see next step).

## 6) Run baseline (Stage 0)
Use the helper script; it runs the ungated model `COUNT` times with incrementing seeds.
```bash
# Default: 3 runs, seeds 1337-1339, 8 GPUs
scripts/run_baseline.sh

# Adjust GPU count or number of runs:
NPROC=4 COUNT=5 BASE_SEED=2000 scripts/run_baseline.sh
```
Each run writes `logs/<run_id>.txt`, checkpoints (if enabled), and appends a row to `experiments/results.csv`.

## 7) Monitor
- Tail the master log: `tail -f logs/<run_id>.txt` (rank 0 only writes logs).
- `nvidia-smi` to verify utilization and memory.

## 8) Cleanup / resume
- Logs remain in `logs/`; results table in `experiments/results.csv`.
- Checkpoint saving is off by default (`save_every=0`); set via CLI edit if you need recovery.
