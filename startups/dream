#!/bin/bash -l
#SBATCH --array=0-999
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=def-ekarimi
# Standard output and error:
#SBATCH -o ./dreamjobsout/out.%j
#SBATCH -e ./dreamjobserr/err.%j
#SBATCH -D ./
#SBATCH -J dream_output
# memory
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

module load python/3.9
module load scipy-stack
source env/bin/activate

srun python dream.py $SLURM_ARRAY_TASK_ID