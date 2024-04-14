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

echo "Checking out the correct version of the code commit_id cc064745dc08f6e84e5fa05e2d08a57dbe17dc6b"
cd $TMPDIR/code
git checkout cc064745dc08f6e84e5fa05e2d08a57dbe17dc6b

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Sampling" 
python -m project.sample --model_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/model_multicoil_1/ema_0.9999_005000.pt --data_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/ --image_size 512 --num_channels 32 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --save_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/model_multicoil_1/ --num_samples 1 --batch_size 1 --data_type 'multicoil' --R 4 --contrast ''
if [[ $? != 0 ]]; then
    exit -1
fi
