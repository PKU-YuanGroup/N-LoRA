#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_4.sh> logs_and_outputs/order_4/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/1-MNLI \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round1 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/1-MNLI/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/2-CB \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round2 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/2-CB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/3-WiC \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round3 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/3-WiC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/4-COPA \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round4 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/4-COPA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/5-QQP \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round5 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/5-QQP/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/6-BoolQA \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round6 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/6-BoolQA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/7-RTE \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round7 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/7-RTE/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/8-IMDB \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round8 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/8-IMDB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/9-yelp \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round9 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/9-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/10-amazon \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round10 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/10-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/11-SST-2 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round11 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/11-SST-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/12-dbpedia \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round12 \
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
   --lamda_1 1.2

sleep 5

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/12-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/13-agnews \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round13 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/13-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/MultiRC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/14-MultiRC \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round14 \
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

CUDA_VISIBLE_DEVICES=3 deepspeed --master_port $port src/run_N_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/outputs/14-MultiRC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/outputs/15-yahoo \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 512 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-04 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order4_round15 \
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
   --lamda_1 1.2