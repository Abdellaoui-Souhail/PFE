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

echo "Checking out the correct version of the code commit_id 97df459e3d56638f68bec758a35307b5733797a7"
cd $TMPDIR/code
git checkout 97df459e3d56638f68bec758a35307b5733797a7

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training" 
python -m project.train --data_dir /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/PFE/ --log_interval 100 --save_dir /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/model_multicoil_total_aussi --save_interval 200 --image_size 512 --num_channels 32 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 5000 --lr 1e-4 --batch_size 1 --lr_anneal_steps 15000 --undersampling_rate 4 --data_type 'multicoil'

if [[ $? != 0 ]]; then
    exit -1
fi
