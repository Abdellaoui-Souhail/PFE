#!/usr/bin/python

import os
import sys
import subprocess

def makejob(commit_id, nruns):
    return f"""#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Sampling" 
python -m project.sample --model_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/model_multicoil_1/ema_0.9999_005000.pt --data_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/ --image_size 512 --num_channels 32 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 100 --save_path /usr/users/cei2023_2024_inserm_nir_irm/abdellaoui_sou/model_multicoil_1/ --num_samples 1 --batch_size 1 --data_type 'multicoil' --R 4 --contrast ''
if [[ $? != 0 ]]; then
    exit -1
fi
"""
#change echo training 
def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")

# Ensure all the modified files have been staged and commited
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

nruns = 1 if len(sys.argv) == 1 else int(sys.argv[1])

# Launch the batch jobs
submit_job(makejob(commit_id, nruns))


