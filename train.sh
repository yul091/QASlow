MODEL_PATH=t5-small # 'facebook/bart-base', 't5-small'
OUTPUT_DIR=results/t5 # 'results/bart', 'results/t5'
EPOCHS=50

CUDA_VISIBLE_DEVICES=0 python train.py \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train --do_eval --do_predict \
--num_train_epochs $EPOCHS \
--per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
--gradient_accumulation_steps 16 \
--overwrite_cache \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 
