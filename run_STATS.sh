#!/bin/sh

#SBATCH --job-name=STATS
#SBATCH --output=LOGS/STATS.out
#SBATCH --error=LOGS/STATS.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate
conda activate /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/envs/prc_env

pip install xskillscore

#rm $STATS/* #WARNING!!! be careful it will remove all files in STATS

# for HPT_v1
python STATS.py --HPT_path HPT_v1 --train_data_name "train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-03-26T23_no_na.npz" --produce_data_name "produce_for_train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-04-26T23_no_na.npz" --training_unique_name "mse_64_0.001_1e-14_0.5_2_16_8_0.1_64"


# for HPT_v2
#python STATS.py --HPT_path HPT_v1 --train_data_name "train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-03-26T23_no_na.npz" --produce_data_name "produce_for_train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-04-26T23_no_na.npz" --training_unique_name "mse_64_0.001_1e-14_0.5_2_16_8_0.1_64"

exit