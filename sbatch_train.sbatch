#!/bin/bash

#SBATCH -J objint           # Job name
#SBATCH -o objint.o%j       # Name of stdout output file
#SBATCH -e objint.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-dev     # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00         # Run time (hh:mm:ss)
#SBATCH -A CCR25018         # Project/Allocation name (req'd if you have more than 1)

module purge 
module load gcc/11.2.0   
module load cuda/11.4
export CC=$(which gcc)
export CXX=$(which g++)

export CUDA_HOME=/opt/apps/cuda/11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CONDA_BASE="$SCRATCH/anaconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/10785/nidhinthomas/conda_envs/objint

echo "--- Environment Verification ---"
echo "Which nvcc: $(which nvcc)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Python executable: $(which python)"
echo "------------------------------"

#python scripts/train.py -d data/example --log-unique

python scripts/test.py -p logs/_data_example_train_20250715_192955 -depth 16
