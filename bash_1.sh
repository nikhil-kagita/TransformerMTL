#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

eval "$(conda shell.bash hook)"
conda activate pytorch2
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main_image_1.py \
    --batch_size 10 --cls_token \
    --finetune ./mae_pretrain_vit_b.pth \
    --dist_eval --data_path ./data/ \
    --output_dir ./output8/  \
    --drop_path 0.0  --blr 0.1 \
    --dataset nyu_v2 --ffn_adapt --global_pool --epochs 400
