#!/bin/bash

# Exit on error
set -e
set -o pipefail


storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python


# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
eval_use_gpu=1
out_dir=librimix # Controls the directory name associated to the evaluation results inside the experiment directory
sample_rate=8000
mode=min
n_src=2
task=sep_noisy  # one of 'enh_single', 'enh_both', 'sep_clean', 'sep_noisy'
eval_mode=

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode

if [ -z "$eval_mode" ]; then
  eval_mode=$mode
fi

train_dir=data/$suffix/train-100
valid_dir=data/$suffix/dev
test_dir=data/wav${sr_string}k/$eval_mode/test

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating Librimix dataset"
	if [ -z "$storage_dir" ]; then
		echo "Need to fill in the storage_dir variable in run.sh to run stage 0. Exiting"
		exit 1
	fi
  . local/generate_librimix.sh --storage_dir $storage_dir --n_src $n_src
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating csv files including wav path and duration"
  . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
  
fi

# # Generate a random ID for the run if no tag is specified
# uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
# if [[ -z ${tag} ]]; then
# 	tag=${uuid}
# fi

# expdir=exp/dprnn_tse_${tag}
# mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
# echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 2 ]]; then
  echo "Stage 2: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py

fi

if [[ $stage -le 3 ]]; then
	echo "Stage 3 : Evaluation"
    CUDA_VISIBLE_DEVICES=$id $python_path eval.py --test_dir data/wav8k/min/test \
                         --task $task \
                         --model_path exp/tmp/best_model.pth \
                         --out_dir exp/tmp/out_best \
                         --exp_dir exp/tmp \
                         --use_gpu=$eval_use_gpu

