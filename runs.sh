#!/bin/sh

#SBATCH --job-name=runs
#SBATCH --output=LOGS/runs.out
#SBATCH --error=LOGS/runs.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:30:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate
conda activate /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/envs/prc_env

#######################################
#python Figs.py
#######################################

#######################################
python STATS.py --HPT_path HPT_intensity --train_data_name train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-03-26T23_no_na_intensity.npz --training_unique_name mse_64_0.0001_1e-14_0.5_2_8_8_0.1_64 --produce_unique_name produce_for_train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-04-26T23_no_na_intensity.npz --delete False
#######################################

exit