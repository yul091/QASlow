ATT_METHOD=structure # word, structure, pwws, scpn, viper
MAX_PER=5
MODEL_PATH=results/dialogpt # results/bart, results/t5, results/dialogpt
DATASET=blended_skill_talk # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
NUM_SAMPLES=3
MAX_LENGTH=256


CUDA_VISIBLE_DEVICES=1 python attack.py \
--attack_strategy $ATT_METHOD \
--max_per $MAX_PER \
--model_name_or_path $MODEL_PATH \
--dataset $DATASET \
--max_num_samples $NUM_SAMPLES \
--max_len $MAX_LENGTH \
--out_dir results/logging
