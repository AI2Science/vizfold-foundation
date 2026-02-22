#!/bin/bash
#SBATCH -J boltz_trace
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --tmp=100G
#SBATCH -o outputs/boltz_runs/slurm-%j.out
#SBATCH -e outputs/boltz_runs/slurm-%j.err

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/boltz_runs

bash scripts/boltz/run_boltz_trace.sh