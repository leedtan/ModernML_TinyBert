python run_glue_datasets.py \
    --model_type bert --model_name_or_path bert-base-uncased --task_name RTE \
    --do_train --data_dir glue_data --max_seq_length 128 \
    --per_gpu_eval_batch_size 8 --weight_decay 0 --seed 0 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 2 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir glue_data\\cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/MIXOUT/RTE/SEED0 --mixout 0.1 --no_cuda --all_datasets
