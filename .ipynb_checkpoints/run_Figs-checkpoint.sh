#!/bin/sh

#SBATCH --job-name=run_figs
#SBATCH --output=HPT_LOG/run_figs.out
#SBATCH --error=HPT_LOG/run_figs.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:30:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate
conda activate /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/envs/prc_env

python Figs.py
