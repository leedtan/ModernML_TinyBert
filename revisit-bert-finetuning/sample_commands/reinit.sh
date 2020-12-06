# using bias correction
python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir /persist/data/glue_data/RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 --fp16 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/ubuntu/hf-transformers-cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/REINIT5/RTE/SEED0 \
    --reinit_pooler --reinit_layers 6

    
python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/mixout_less_scaling \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda
    
python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/layer_mixout \
    --reinit_pooler --reinit_layers 4 --mixout_layers 4 --mixout .3 --no_cuda --layer_mixout

    
python run_glue_experiment.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
    --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/mixout_less_scaling --mixout 0.2 --no_cuda
