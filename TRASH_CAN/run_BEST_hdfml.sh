# remove existing directories:
#rm -r /p/project/deepacf/kiste/patakchiyousefi1/CODES/CODES-MS2/PLAYGROUND/ATMOSCORRECT/HPT_LOG/*
#rm -r /p/scratch/deepacf/kiste/patakchiyousefi1/BEST_DL/*

# Set the maximum number of jobs running at a time
MAX_JOBS=8

LR_combo=(0.001)
BS_combo=(16)
LR_factor_combo=(0.5)
Filters_combo=(16)

check_job_limit() {
  # Get the number of currently running and pending jobs with names starting with "BEST_DL_TRAIN_LR_"
  local running_jobs=$(squeue -u patakchiyousefi1 -t R -o "%A %j" | grep "BEST_DL_TRAIN_LR_" | wc -l)
  local pending_jobs=$(squeue -u patakchiyousefi1 -t PD -o "%A %j" | grep "BEST_DL_TRAIN_LR_" | wc -l)
  local total_jobs=$((running_jobs + pending_jobs))
  
  if [ "$total_jobs" -ge "$MAX_JOBS" ]; then
    echo "Maximum number of jobs reached. Waiting..."
    sleep 600  # Sleep for 60 seconds and check again
    check_job_limit
  fi
}

for LR in "${LR_combo[@]}"; do
  for BS in "${BS_combo[@]}"; do
    for LR_factor in "${LR_factor_combo[@]}"; do
      for Filters in "${Filters_combo[@]}"; do
        check_job_limit
        squeue -u patakchiyousefi1
        sbatch <<EOF
#!/bin/sh
#SBATCH --job-name=BEST_DL_TRAIN_LR_${LR}_BS_${BS}_LRF_${LR_factor}_F_${Filters}
#SBATCH --output=HPT_LOG/BEST_DL_TRAIN_LR_${LR}_BS_${BS}_LRF_${LR_factor}_F_${Filters}.out
#SBATCH --error=HPT_LOG/BEST_DL_TRAIN_LR_${LR}_BS_${BS}_LRF_${LR_factor}_F_${Filters}.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=3:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf
#SBATCH --gres=gpu:4

module load TensorFlow matplotlib xarray
#source /p/project/deepacf/kiste/patakchiyousefi1/SC_VENV/bin/activate prc_env

python DL_TRAIN-BEST.py --lr $LR --bs $BS --lr_factor $LR_factor --filters $Filters

EOF
    
      done
    done
  done
done

