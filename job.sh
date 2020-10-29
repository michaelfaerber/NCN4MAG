#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=600gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ucgvm@student.kit.edu
#SBATCH --mail-type=FAIL

source /pfs/data5/home/kit/stud/ucgvm/testenv/bin/activate

python3 -u /pfs/work7/workspace/scratch/ucgvm-input-0/NCN4MAG/ncn/pandas_all.py