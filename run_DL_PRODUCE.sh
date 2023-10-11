#!/bin/sh

#SBATCH --job-name=DL_PRODUCE
#SBATCH --output=HPT_LOG/DL_PRODUCE.out
#SBATCH --error=HPT_LOG/DL_PRODUCE.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

module load TensorFlow matplotlib xarray

python DL_PRODUCE.py --lr 0.001 --bs 16 --lr_factor 0.5 --filters 64 --date_start "2020-07-01T13" --date_end "2023-04-26T23"
