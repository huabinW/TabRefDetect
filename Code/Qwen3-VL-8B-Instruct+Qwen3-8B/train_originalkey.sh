#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=1605632
export OMP_NUM_THREADS=$(nproc)

# 使用之前微调过的模型路径
MODEL_PATH="./Qwen3-VL-8B-Instruct"
TRAIN_DATA="./qwen3vl_originalkey_fold0.jsonl"
OUTPUT_DIR="./qwen3-sft-original_fold0"
VAL_DATA="./qwen3vl_originalkey_val_fold0.jsonl"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 执行 Swift SFT 微调（使用 Swift 支持的参数）
swift sft \
    --model $MODEL_PATH \
    --dataset $TRAIN_DATA \
    --tuner_type lora \
    --lorap_lr_ratio 10 \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --save_steps 20 \
    --eval_steps 20 \
    --save_total_limit 8 \
    --logging_steps 10 \
    --seed 42 \
    --learning_rate 1e-5 \
    --init_weights true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-08 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --warmup_steps 0 \
    --gradient_checkpointing true \
    --report_to none \
    --val_dataset $VAL_DATA
