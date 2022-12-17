import sys
sys.dont_write_bytecode = True

import time
import random
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
import evaluate
from attackers.my_attacker import WordAttacker, StructureAttacker


def seq2seq_generation(
    instance, 
    tokenizer, 
    model, 
    device, 
    attacker, 
    max_source_length, 
    max_target_length, 
    bleu,
    rouge,
    meteor,
):
    num_entries = len(instance["free_messages"])
    persona_pieces = [
        f"<PS> {instance['personas'][0]}",
        f"<PS> {instance['personas'][1]}",
    ]
    if instance['context'] == "wizard_of_wikipedia":
        additional_context_pieces = [f"<CTX> {instance['additional_context']}. <SEP> "]
    else:
        additional_context_pieces = ["<SEP> "]

    previous_utterance_pieces = instance["previous_utterance"]
    ori_lens, adv_lens = [], []
    ori_bleus, adv_bleus = [], []
    ori_rouges, adv_rouges = [], []
    ori_meteors, adv_meteors = [], []
    ori_time, adv_time = [], []
    att_success = 0
    for entry_idx in range(num_entries):
        free_message = instance['free_messages'][entry_idx]
        guided_message = instance['guided_messages'][entry_idx]
        previous_utterance = ' <SEP> '.join(previous_utterance_pieces)
        original_context = ' '.join(
            persona_pieces + additional_context_pieces
        ) + previous_utterance
        print("\nDialogue history: {}".format(original_context))

        previous_utterance_pieces += [
            free_message,
            guided_message,
        ]
        # Original generation
        text = original_context + ' ' + tokenizer.eos_token + ' ' + free_message
        input_ids = tokenizer(text, max_length=max_source_length, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        t1 = time.time()
        outputs = model.generate(input_ids, max_length=max_target_length, do_sample=False)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        t2 = time.time()
        print("U--{}".format(free_message))
        print("G--{}".format(output))
        # pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        # scores = compute_metrics(output, guided_message, metric, tokenizer)
        bleu_res = bleu.compute(
            predictions=[output], 
            references=[[guided_message]],
        )
        rouge_res = rouge.compute(
            predictions=[output],
            references=[guided_message],
        )
        meteor_res = meteor.compute(
            predictions=[output],
            references=[guided_message],
        )
        pred_len = bleu_res['translation_length']
        print("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
            pred_len, t2-t1, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
        ))
        ori_lens.append(pred_len)
        ori_bleus.append(bleu_res['bleu'])
        ori_rouges.append(rouge_res['rougeL'])
        ori_meteors.append(meteor_res['meteor'])
        ori_time.append(t2-t1)

        # Attack
        success, adv_his = attacker.run_attack(text)
        new_text = adv_his[-1][0]
        # new_context = new_text.split(tokenizer.eos_token)[0].strip()
        new_free_message = new_text.split(tokenizer.eos_token)[1].strip()
        # print("Dialogue history: {}".format(new_context))
        if success:
            print("U'--{}".format(new_free_message))
        else:
            print("Attack failed!")

        input_ids = tokenizer(new_text, max_length=max_source_length, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        t1 = time.time()
        outputs = model.generate(input_ids, max_length=max_target_length, do_sample=False)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        t2 = time.time()
        print("G'--{}".format(output))
        # adv_pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        # adv_scores = compute_metrics(output, guided_message, metric, tokenizer)
        bleu_res = bleu.compute(
            predictions=[output], 
            references=[[guided_message]],
        )
        rouge_res = rouge.compute(
            predictions=[output],
            references=[guided_message],
        )
        meteor_res = meteor.compute(
            predictions=[output],
            references=[guided_message],
        )
        adv_pred_len = bleu_res['translation_length']
        print("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
            adv_pred_len, t2-t1, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
        ))
        adv_lens.append(adv_pred_len)
        adv_bleus.append(bleu_res['bleu'])
        adv_rouges.append(rouge_res['rougeL'])
        adv_meteors.append(meteor_res['meteor'])
        adv_time.append(t2-t1)
        # ASR
        att_success += (adv_pred_len > pred_len)

    return ori_lens, adv_lens, ori_bleus, adv_bleus, ori_rouges, adv_rouges, \
        ori_meteors, adv_meteors, ori_time, adv_time, att_success, num_entries



def main(
    max_num_samples=5, 
    max_per=3, 
    max_len=256, 
    model_name_or_path="results/bart", 
    dataset="blended_skill_talk",
):
    random.seed(2019)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # Load dataset
    bst_dataset = load_dataset(dataset)
    test_dataset = bst_dataset['test']
    ids = random.sample(range(len(test_dataset)), max_num_samples)
    sampled_test_dataset = test_dataset.select(ids)

    # attacker = WordAttacker(
    attacker = StructureAttacker(
        device=device,
        tokenizer=tokenizer,
        model=model,
        max_len=max_len,
        max_per=max_per,
    )

    # metric = load_metric("sacrebleu")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    Ori_lens, Adv_lens = [], []
    Ori_bleus, Adv_bleus = [], []
    Ori_rouges, Adv_rouges = [], []
    Ori_meteors, Adv_meteors = [], []
    Ori_time, Adv_time = [], []
    Att_success = 0
    Total_pairs = 0
    for i, instance in tqdm(enumerate(sampled_test_dataset)):
        ori_lens, adv_lens, ori_bleus, adv_bleus, ori_rouges, adv_rouges, \
            ori_meteors, adv_meteors, ori_time, adv_time, att_success, num_entries = \
                seq2seq_generation(
                    instance, tokenizer, model, device, attacker, max_len, max_len, bleu, rouge, meteor
                )
        Ori_lens.extend(ori_lens)
        Adv_lens.extend(adv_lens)
        Ori_bleus.extend(ori_bleus)
        Adv_bleus.extend(adv_bleus)
        Ori_rouges.extend(ori_rouges)
        Adv_rouges.extend(adv_rouges)
        Ori_meteors.extend(ori_meteors)
        Adv_meteors.extend(adv_meteors)
        Ori_time.extend(ori_time)
        Adv_time.extend(adv_time)
        Att_success += att_success
        Total_pairs += num_entries

    # Summarize eval results
    Ori_len = np.mean(Ori_lens)
    Adv_len = np.mean(Adv_lens)
    Ori_bleu = np.mean(Ori_bleus)
    Adv_bleu = np.mean(Adv_bleus)
    Ori_rouge = np.mean(Ori_rouges)
    Adv_rouge = np.mean(Adv_rouges)
    Ori_meteor = np.mean(Ori_meteors)
    Adv_meteor = np.mean(Adv_meteors)
    Ori_t = np.mean(Ori_time)
    Adv_t = np.mean(Adv_time)
    print("Original output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
        Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
    ))
    print("Perturbed output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
        Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
    ))
    print("Attack success rate: {:.2f}%".format(100*Att_success/Total_pairs))

    # # Save generation files
    # with open(f"results/ori_gen_{max_per}.txt", "w") as f:
    #     for line in ori_lens:
    #         f.write(str(line) + "\n")
    # with open(f"results/adv_gen_{max_per}.txt", "w") as f:
    #     for line in adv_lens:
    #         f.write(str(line) + "\n")


if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

    max_num_samples = 1
    max_per = 5
    max_len = 256
    main(
        max_num_samples=max_num_samples,
        max_per=max_per,
        max_len=max_len,
        # model_name_or_path="results/bart",
        model_name_or_path="results/t5",
        dataset="blended_skill_talk",
    )


