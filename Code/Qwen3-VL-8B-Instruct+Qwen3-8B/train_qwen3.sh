#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=1605632
export OMP_NUM_THREADS=$(nproc)

MODEL_PATH="./Qwen3-8B"
TRAIN_DATA="./sft_semantic_match_qwen3_fold0.jsonl"
OUTPUT_DIR="./qwen3-8b_semantic_fold0"
VAL_DATA="./sft_semantic_val_qwen3_fold0.jsonl"

mkdir -p $OUTPUT_DIR

swift sft \
    --model $MODEL_PATH \
    --dataset $TRAIN_DATA \
    --train_type lora \
    --lorap_lr_ratio 10 \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --val_dataset $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 8 \
    --save_steps 4 \
    --eval_steps 4 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --seed 42 \
    --learning_rate 4e-5 \
    --init_weights true \
    --lora_rank 16 \
    --lora_alpha 64 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-08 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 64 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --warmup_steps 0 \
    --gradient_checkpointing true \
    --report_to none \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false 
