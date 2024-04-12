#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 553bf59a5a9e74c4d06ceefecdd75131d0c916f9"
cd $TMPDIR/code
git checkout 553bf59a5a9e74c4d06ceefecdd75131d0c916f9

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training" 
python -m train.py --data_dir /path_to_data/ --log_interval 5000 --save_dir 'model_multicoil' --save_interval 5000 --image_size 384 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --lr 1e-4 --batch_size 1 --lr_anneal_steps 15000 --undersampling_rate 2 --data_type 'multicoil'


if [[ $? != 0 ]]; then
    exit -1
fi
