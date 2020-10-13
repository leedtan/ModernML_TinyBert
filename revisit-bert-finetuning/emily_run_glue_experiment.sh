# here i've set max_steps = 1 for testing purposes, it definitely should not be 1
python run_glue_experiment.py \
    --model_type bert --model_name_or_path bert-base-uncased --task_name RTE \
    --do_train --data_dir ~/Documents/ModernML_TinyBert/RTE --max_seq_length 64 \
    --per_gpu_eval_batch_size 16 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw \
    --cache_dir  ~/Documents/ModernML_TinyBert/cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/ --mixout 0.1 --max_steps 1


    
python run_glue_experiment.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 16 \
    --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir C:\\workspace\\ModernML_TinyBert\\glue_data\\cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/SEED0 --mixout 0.1 --no_cuda