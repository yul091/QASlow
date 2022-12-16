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
from datasets import load_dataset, load_metric
from my_attack import WordAttacker, StructureAttacker, compute_metrics


def seq2seq_generation(
    instance, 
    tokenizer, 
    model, 
    device, 
    attacker, 
    max_source_length, 
    max_target_length, 
    metric,
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
    ori_time, adv_time = [], []
    att_success = 0

    for entry_idx in range(num_entries):
        free_message = instance['free_messages'][entry_idx]
        guided_message = instance['guided_messages'][entry_idx]

        previous_utterance = ' <SEP> '.join(previous_utterance_pieces)
        original_context = ' '.join(
            persona_pieces + additional_context_pieces
        ) + previous_utterance
        print("\nDialogue history: {}".format(previous_utterance))

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
        pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        eval_scores = compute_metrics(output, guided_message, metric, tokenizer)
        print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(pred_len, t2-t1, eval_scores))
        ori_lens.append(pred_len)
        ori_bleus.append(eval_scores)
        ori_time.append(t2-t1)

        # Attack
        success, adv_his = attacker.run_attack(text)
        new_text = adv_his[-1][0]
        previous_utterance = new_text.split(tokenizer.eos_token)[0].strip()
        new_free_message = new_text.split(tokenizer.eos_token)[1].strip()
        print("Dialogue history: {}".format(previous_utterance))
        if success:
            print("U'--: {}".format(new_free_message))
        else:
            print("Attack failed!")

        input_ids = tokenizer(new_text, max_length=max_source_length, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        t1 = time.time()
        outputs = model.generate(input_ids, max_length=max_target_length, do_sample=False)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        t2 = time.time()
        print("G'--{}".format(output))
        adv_pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        eval_scores = compute_metrics(output, guided_message, metric, tokenizer)
        print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(pred_len, t2-t1, eval_scores))
        adv_lens.append(adv_pred_len)
        adv_bleus.append(eval_scores)
        adv_time.append(t2-t1)

        # ASR
        att_success += (adv_pred_len > pred_len)

    return ori_lens, adv_lens, ori_bleus, adv_bleus, ori_time, adv_time, att_success, num_entries



def main(max_num_samples=5, max_per=3, max_len=256):
    random.seed(2019)
    model_name_or_path = "results/bart" # "facebook/bart-base", "results/bart"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # Load dataset
    bst_dataset = load_dataset("blended_skill_talk")
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

    metric = load_metric("sacrebleu")
    Ori_lens, Adv_lens = [], []
    Ori_bleus, Adv_bleus = [], []
    Ori_time, Adv_time = [], []
    Att_success = 0
    Total_pairs = 0

    for i, instance in tqdm(enumerate(sampled_test_dataset)):

        ori_lens, adv_lens, ori_bleus, adv_bleus, ori_time, adv_time, att_success, num_entries = \
            seq2seq_generation(instance, tokenizer, model, device, attacker, max_len, max_len, metric)

        Ori_lens.extend(ori_lens)
        Adv_lens.extend(adv_lens)
        Ori_bleus.extend(ori_bleus)
        Adv_bleus.extend(adv_bleus)
        Ori_time.extend(ori_time)
        Adv_time.extend(adv_time)
        Att_success += att_success
        Total_pairs += num_entries

        # if total_pairs >= max_num_samples:
        #     break

        # for (sentence, label) in zip(instance['free_messages'], instance['guided_messages']):
        #     input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        #     input_ids = input_ids.to(device)
        #     t1 = time.time()
            
        #     outputs = model.generate(input_ids, max_length=max_len, do_sample=False)
        #     output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        #     t2 = time.time()
        #     pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        #     eval_scores = compute_metrics(output, label, metric, tokenizer)
            
        #     ori_lens.append(pred_len)
        #     ori_bleus.append(eval_scores)
        #     ori_time.append(t2-t1)
            
        #     # Attack
        #     success, adv_his = attacker.run_attack(sentence)
        #     print('\n')
        #     print("U--{}".format(sentence))
        #     print("G--{}".format(output))
        #     print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(pred_len, t2-t1, eval_scores))

        #     if success:
        #         # print("Attack Succeed!")
        #         print("U'--{}".format(adv_his[-1][0]))
        #     else:
        #         print("Attack failed!")

        #     input_ids = tokenizer(adv_his[-1][0], return_tensors="pt").input_ids
        #     input_ids = input_ids.to(device)
        #     t1 = time.time()
        #     outputs = model.generate(input_ids, max_length=max_len, do_sample=False)
        #     output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        #     t2 = time.time()
        #     adv_pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
        #     print("G'--{}".format(output))
        #     eval_scores = compute_metrics(output, label, metric, tokenizer)
        #     print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(adv_pred_len, t2-t1, eval_scores))

        #     adv_lens.append(adv_pred_len)
        #     adv_bleus.append(eval_scores)
        #     adv_time.append(t2-t1)

        #     att_success += (adv_pred_len > pred_len)
        #     total_pairs += 1

        #     if total_pairs >= max_num_samples:
        #         break


    # Summarize eval results
    Ori_len = np.mean(Ori_lens)
    Adv_len = np.mean(Adv_lens)
    Ori_bleu = np.mean(Ori_bleus)
    Adv_bleu = np.mean(Adv_bleus)
    Ori_t = np.mean(Ori_time)
    Adv_t = np.mean(Adv_time)
    print("Original output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}".format(Ori_len, Ori_t, Ori_bleu))
    print("Adversarial output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}".format(Adv_len, Adv_t, Adv_bleu))
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
    # nltk.download('averaged_perceptron_tagger')

    max_num_samples = 1
    max_per = 5
    max_len = 256
    main(max_num_samples, max_per, max_len)


