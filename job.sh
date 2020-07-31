#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem=300gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ucgvm@student.kit.edu

source /pfs/data5/home/kit/stud/ucgvm/testenv/bin/activate

python3 -u /pfs/work7/workspace/scratch/ucgvm-input-0/neural_citation/ncn/data.py