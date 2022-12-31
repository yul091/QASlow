ATT_METHOD=bae # word, structure, pwws, scpn, viper, bae
MAX_PER=5
MODEL_PATH=results/bart # results/bart, results/t5, results/dialogpt, results/personagpt, gpt2
DATASET=AlekseyKorshuk/persona-chat # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
NUM_SAMPLES=5
MAX_LENGTH=256


CUDA_VISIBLE_DEVICES=0 python attack.py \
--attack_strategy $ATT_METHOD \
--max_per $MAX_PER \
--model_name_or_path $MODEL_PATH \
--dataset $DATASET \
--max_num_samples $NUM_SAMPLES \
--max_len $MAX_LENGTH \
--out_dir results/logging
