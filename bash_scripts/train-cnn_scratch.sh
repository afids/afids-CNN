#!/bin/bash
set -eo pipefail 
source /project/ctb-akhanf/ataha24/venv_archives/virtualenvs/afids-cnn-2o1BhLKg-py3.9/bin/activate
auto_afids_cnn_train_bids --frequency 300 --afids_dir /scratch/ataha24/afids-data/data/autoafids/train/ /scratch/ataha24/afids-data/data/autoafids/train/ /scratch/ataha24/afids-data/data/autoafids/train/training_20240205 --validation-afids-dir /scratch/ataha24/afids-data/data/autoafids/validation --validation-bids-dir /scratch/ataha24/afids-data/data/autoafids/validation participant --epochs 200 --steps-per-epoch 100 --validation-steps 100 --do_early_stopping --cores 2 --use-singularity --config containers='{c3d: "/project/6050199/akhanf/singularity/bids-apps/itksnap_v3.8.2.sif"}'
