../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 1 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_1 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 2 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_2 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 3 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_3 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 4 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_4 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 5 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_5 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 6 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_6 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 7 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_7 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 8 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_8 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 9 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_9 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 10 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_10 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 11 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_11 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 12 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_12 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 13 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_13 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 14 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_14 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 15 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_15 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 16 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_16 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 17 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_17 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 18 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_18 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda
    
../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 19 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_19 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 20 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/MIXOUT/RTE/mix_clean_20 \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda