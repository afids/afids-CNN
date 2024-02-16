#!/bin/bash
set -eo pipefail 
module load cuda cudnn 
source /scratch/ataha24/0_dev/0.afids-CNN/tensorflow/bin/activate
auto_afids_cnn_train_bids --frequency 300 --afids_dir /scratch/ataha24/afids-data/data/autoafids/train/ /scratch/ataha24/afids-data/data/autoafids/train/ /scratch/ataha24/afids-data/data/autoafids/train/training_20240209 --validation-afids-dir /scratch/ataha24/afids-data/data/autoafids/validation --validation-bids-dir /scratch/ataha24/afids-data/data/autoafids/validation participant --epochs 200 --steps-per-epoch 100 --validation-steps 20 --do_early_stopping --cores 1 --use-singularity --config containers='{c3d: "/project/6050199/akhanf/singularity/bids-apps/itksnap_v3.8.2.sif"}'
