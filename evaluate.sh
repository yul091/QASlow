TASK=personachat # blended_skill_talk, cmu_dog, personachat, convai2
MODEL=seq2seq # ir_baseline, seq2seq, transformer/generator, transformer/ranker
DATATYPE=test # train, train:evalmode, train:ordered, train:stream, train:stream:ordered, valid, test
DATAPATH=/home/dsi/yufli/ParlAI/data
OUTDIR=log
CKPT=/home/dsi/yufli/ParlAI/results/${TASK}/${MODEL}

mkdir -p $OUTDIR/${TASK}
LOGFILE=${OUTDIR}/${TASK}/${MODEL}.log # log file to save the evaluation results


# Evaluate the model
CUDA_VISIBLE_DEVICES=0 python eval_model.py \
    --model ${MODEL} \
    --model_file ${CKPT} \
    --task ${TASK} \
    --datapath ${DATAPATH} \
    --datatype ${DATATYPE} \
    --batchsize 32 \
    --metrics 'ppl,rouge,bleu' \
    --report-filename ${LOGFILE} 