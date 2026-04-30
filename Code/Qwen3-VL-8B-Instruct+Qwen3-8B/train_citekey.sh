#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export MAX_PIXELS=1605632
export OMP_NUM_THREADS=$(nproc)

MODEL_PATH="./Qwen3-VL-8B-Instruct"
TRAIN_DATA="./qwen3vl_citekey_training_fold0.jsonl"
OUTPUT_DIR="./qwen3-citekey_fold0"
VAL_DATA="./qwen3vl_citekey_val_fold0.jsonl"

mkdir -p $OUTPUT_DIR

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
    --val_dataset $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --save_steps 50 \
    --eval_steps 25 \
    --save_total_limit 4 \
    --logging_steps 20 \
    --seed 42 \
    --learning_rate 1e-5 \
    --init_weights true \
    --lora_rank 16 \
    --lora_alpha 64 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-08 \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 64 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --warmup_steps 0 \
    --gradient_checkpointing true \
    --report_to none \
    --lora_dropout 0.05