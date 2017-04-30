#!/bin/bash -l

module load cuda/8.0
module load cudnn/5.1
#module load python/2.7.13
#module load tensorflow/r1.0_python-2.7.13 
module load python/3.6.0
module load tensorflow/r1.0_python-3.6.0
module load gcc

#python3 pset5.py
python3 main.py --dataset catface --dataset2 catblur --is_crop --is_train --epoch 600

#module unload tensorflow
#module unload python
#module unload cudnn
#module unload cuda
