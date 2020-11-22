# here i've set max_steps = 1 for testing purposes, it definitely should not be 1
# ../../revisit-bert-finetuning/bert/bin/python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-base-uncased --task_name RTE \
#     --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 64 \
#     --per_gpu_eval_batch_size 16 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 4 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw \
#     --cache_dir  /home/ubuntu/hf-transformers-cache \
#     --num_train_epochs 1.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/RYAN --mixout 0.1 --max_steps 1

# ../../revisit-bert-finetuning/bert/bin/python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 2 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 2 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir tests/MIXOUT/RTE/MIX2 --mixout 0.2
    
# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 3 \
#     --per_gpu_eval_batch_size 1 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 1 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/SEED0 --mixout 0.1 --no_cuda
    
# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/reverseds1e2 --mixout 0.1 --no_cuda
    
# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/fan_out_xdetach1e22 --mixout 0.1 --no_cuda
    
# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/mixout4 --mixout 0.2 --no_cuda

../../revisit-bert-finetuning/bert/bin/python run_glue_experiment.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir ../../revisit-bert-finetuning/glue_data/RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 1 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/mixout2l2reg3e3 --mixout 0.2

# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/fan_out_no_detach1e2 --mixout 0.1 --no_cuda

# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/fan_out_raw_1e2 --mixout 0.1 --no_cuda

# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-base-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 16 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/fan_out_no_detach1e2 --mixout 0.1 --no_cuda

# python run_glue_experiment.py \
#     --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
#     --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
#     --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
#     --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
#     --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
#     --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
#     --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
#     --output_dir bert_output/MIXOUT/RTE/L5en1 --mixout 0.1 --no_cuda