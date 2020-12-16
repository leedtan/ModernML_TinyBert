../../revisit-bert-finetuning/bert/bin/python run_glue_datasets.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data --max_seq_length 64 \
    --per_gpu_eval_batch_size 8 --weight_decay 0 --seed 1 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir tests/FULLTESTS/decay_50_to_20 \
    --reinit_pooler --normalize --mixout_layers 12 --mixout .2 \
    --trials 20 --mixout_decay 0.5

# ../../revisit-bert-finetuning/bert/bin/python run_glue_datasets.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir ../../revisit-bert-finetuning/glue_data --max_seq_length 64 \
#     --per_gpu_eval_batch_size 8 --weight_decay 0 --seed 1 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 8 \
#     --gradient_accumulation_steps 4 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir tests/FULLTESTS/decay_90_to_20 \
#     --reinit_pooler --normalize --mixout_layers 12 --mixout .2 \
#     --trials 20 --mixout_decay 0.9

# ../../revisit-bert-finetuning/bert/bin/python run_glue_datasets.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir ../../revisit-bert-finetuning/glue_data --max_seq_length 64 \
#     --per_gpu_eval_batch_size 8 --weight_decay 0 --seed 1 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 8 \
#     --gradient_accumulation_steps 4 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir tests/FULLTESTS/classic --all_datasets \
#     --reinit_pooler --normalize --mixout_layers 12 --mixout .3 \
#     --trials 10
    
# ../../revisit-bert-finetuning/bert/bin/python run_glue_datasets.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir ../../revisit-bert-finetuning/glue_data --max_seq_length 64 \
#     --per_gpu_eval_batch_size 8 --weight_decay 0 --seed 1 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 8 \
#     --gradient_accumulation_steps 4 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir tests/FULLTESTS/classic --all_datasets \
#     --reinit_pooler --normalize --mixout_layers 12 --mixout .3 \
#     --trials 10