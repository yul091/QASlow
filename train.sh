##########################################################################################
#                                   Seq2Seq Training
##########################################################################################

# MODEL_PATH=t5-small # 'facebook/bart-base', 't5-small'
# OUTPUT_DIR=results/t5 # 'results/bart', 'results/t5'
# DATASET=blended_skill_talk # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
# EPOCHS=50

# CUDA_VISIBLE_DEVICES=0 python train_seq2seq.py \
    # --model_name_or_path $MODEL_PATH \
    # --dataset $DATASET \
    # --output_dir $OUTPUT_DIR \
    # --do_train --do_eval --do_predict \
    # --num_train_epochs $EPOCHS \
    # --per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
    # --gradient_accumulation_steps 16 \
    # --overwrite_cache \
    # --max_source_length 256 --max_target_length 256 --val_max_target_length 256 


##########################################################################################
#                            Causal Language Modeling Training
##########################################################################################

MODEL_PATH=gpt2 # microsoft/DialoGPT-small, gpt2
OUTPUT_DIR=results/personagpt # results/dialogpt, results/personagpt
DATASET=AlekseyKorshuk/persona-chat # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
EPOCHS=50

CUDA_VISIBLE_DEVICES=0 python train_clm.py \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --do_train --do_eval --do_predict \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 16 \
    --overwrite_cache \
    --pad_to_max_length \
    --max_length 128