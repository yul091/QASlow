# ATT_METHOD=pwws # word, structure, pwws, scpn, viper, bae, fd, hotflip, textbugger
MAX_PER=5
MODEL_PATH=results/t5 # bart, t5, dialogpt, personagpt, gpt2 (/nfs/intern_data/yufli/)
DATASET=blended_skill_talk # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
FITNESS=length # performance, length
NUM_SAMPLES=30
MAX_LENGTH=128
SELECT_BEAMS=1

# for ATT_METHOD in structure pwws scpn viper bae fd hotflip textbugger
for ATT_METHOD in scpn viper bae fd hotflip textbugger
do
    CUDA_VISIBLE_DEVICES=0 python attack.py \
        --attack_strategy $ATT_METHOD \
        --max_per $MAX_PER \
        --model_name_or_path $MODEL_PATH \
        --dataset $DATASET \
        --max_num_samples $NUM_SAMPLES \
        --max_len $MAX_LENGTH \
        --select_beams $SELECT_BEAMS \
        --out_dir logging \
        --fitness $FITNESS \
        --use_combined_loss
done