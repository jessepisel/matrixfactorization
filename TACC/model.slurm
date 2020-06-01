#!/bin/bash
#----------------------------------------------------
# RecSys Slurm job script
#----------------------------------------------------

#SBATCH -p gtx            # Queue (partition) name
#SBATCH -J recsys         # Job name
#SBATCH -o recsys.o%j     # Name of stdout output file
#SBATCH -e recsys.e%j     # Name of stderr error file
#SBATCH -N 2              # Total # of nodes (must be 1 for serial)
#SBATCH -n 2              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:00:00       # Run time (hh:mm:ss)
#SBATCH --mail-user=jpisel@tacc.utexas.edu
#SBATCH --mail-type=all   # Send email at begin and end of job
#SBATCH -A FRI-RL         # Allocation name (req'd if you have more than 1)
module load cuda/10.1
python3 recsys.py
