#!/bin/bash
#
#================ SLURM OPTIONS =================#
#
#SBATCH --job-name=my_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Optional GPU request
##SBATCH --gres=gpu:1
#
# Optional email notifications
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@example.com
#
#================================================#

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node List:     $SLURM_NODELIST"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "Working Dir:   $(pwd)"
echo "Start Time:    $(date)"
echo "========================================"

#
# Load environment/modules
#
# module purge

# Example modules
# module load gcc
# module load cuda
# module load python

#
# Activate virtual environment if needed
#
# source ~/venvs/myenv/bin/activate

#
# Change to submission directory
#
cd $SLURM_SUBMIT_DIR

#
# Run commands
#

echo "Running hostname..."
hostname

echo "Running Python script..."
python compare.py --extract
python compare.py --cluster --no-wavelets

#
# Example MPI job
#
# srun ./my_mpi_program

#
# Example GPU check
#
# nvidia-smi

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
