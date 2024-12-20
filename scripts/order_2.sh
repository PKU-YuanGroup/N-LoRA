#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/outputs/1-dbpedia \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4

sleep 5

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/outputs/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/outputs/2-amazon \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4

sleep 5

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/outputs/3-agnews \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4

sleep 5

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/outputs/3-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/outputs/4-yahoo \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4 
