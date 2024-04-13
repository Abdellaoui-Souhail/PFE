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

echo "Checking out the correct version of the code commit_id fa9ab2fac3a26b62a404775dc43f605e7dfed322"
cd $TMPDIR/code
git checkout fa9ab2fac3a26b62a404775dc43f605e7dfed322

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training" 
python -m project.train --data_dir /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/PFE/ --log_interval 100 --save_dir /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/PFE/model_multicoil --save_interval 200 --image_size 512 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 200 --lr 1e-4 --batch_size 2 --lr_anneal_steps 5000 --undersampling_rate 4 --data_type 'multicoil'

if [[ $? != 0 ]]; then
    exit -1
fi
