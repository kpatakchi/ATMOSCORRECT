#!/bin/sh

#SBATCH --job-name=run_PRODUCE
#SBATCH --output=LOGS/run_PRODUCE.out
#SBATCH --error=LOGS/run_PRODUCE.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf


module load TensorFlow matplotlib xarray

# for HPT_v1
python DL_PRODUCE.py --lr 0.001 --bs 16 --lr_factor 0.5 --filters 64 --date_start "2020-07-01T13" --date_end "2023-04-26T23" --mask_type "no_na" --HPT_path "HPT_v1"
#"mse_64_0.001_1e-14_0.5_2_16_8_0.1_64"

# for HPT_v2
#python DL_PRODUCE.py --lr 0.0001 --bs 8 --lr_factor 0.5 --filters 64 --date_start "2020-07-01T13" --date_end "2023-04-26T23" --mask_type "no_na_intensity" --HPT_path "HPT_intensity"

