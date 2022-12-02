import nltk
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

# nltk.download('averaged_perceptron_tagger')

def main(max_num_samples=100, max_per=3):

    random.seed(2019)
    model_name_or_path = "results/" # "facebook/bart-base", "results/"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    bst_dataset = load_dataset("blended_skill_talk")
    test_dataset = bst_dataset['test']
    ids = random.sample(range(len(test_dataset)), max_num_samples)

    sampled_test_dataset = test_dataset.select(ids)
    # print(sampled_test_dataset)

    attacker = WordAttacker(
        device=device,
        tokenizer=tokenizer,
        model=model,
        max_len=64,
        max_per=max_per,
    )

    metric = load_metric("sacrebleu")
    ori_lens, adv_lens = [], []
    ori_bleus, adv_bleus = [], []
    ori_time, adv_time = [], []
    att_success = 0
    total_pairs = 0

    for i, instance in tqdm(enumerate(sampled_test_dataset)):
        if total_pairs >= max_num_samples:
            break

        for (sentence, label) in zip(instance['free_messages'], instance['guided_messages']):

            input_ids = tokenizer(sentence, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            t1 = time.time()
            
            outputs = model.generate(input_ids, max_length=64, do_sample=False)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            t2 = time.time()
            pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
            eval_scores = compute_metrics(output, label, metric, tokenizer)
            
            ori_lens.append(pred_len)
            ori_bleus.append(eval_scores)
            ori_time.append(t2-t1)
            
            # Attack
            success, adv_his = attacker.run_attack(sentence)
            print('\n')
            print("U--{}".format(sentence))
            print("G--{}".format(output))
            print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(pred_len, t2-t1, eval_scores))

            if success:
                # print("Attack Succeed!")
                print("U'--{}".format(adv_his[-1][0]))
            else:
                print("Attack failed!")

            input_ids = tokenizer(adv_his[-1][0], return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            t1 = time.time()
            outputs = model.generate(input_ids, max_length=64, do_sample=False)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            t2 = time.time()
            adv_pred_len = np.count_nonzero(outputs[0].cpu() != tokenizer.pad_token_id)
            print("G'--{}".format(output))
            eval_scores = compute_metrics(output, label, metric, tokenizer)
            print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(adv_pred_len, t2-t1, eval_scores))

            adv_lens.append(adv_pred_len)
            adv_bleus.append(eval_scores)
            adv_time.append(t2-t1)

            att_success += (adv_pred_len > pred_len)
            total_pairs += 1

            if total_pairs >= max_num_samples:
                break


    # Summarize eval results
    ori_len = np.mean(ori_lens)
    adv_len = np.mean(adv_lens)
    ori_bleu = np.mean(ori_bleus)
    adv_bleu = np.mean(adv_bleus)
    ori_t = np.mean(ori_time)
    adv_t = np.mean(adv_time)
    print("Original output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}".format(ori_len, ori_t, ori_bleu))
    print("Adversarial output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}".format(adv_len, adv_t, adv_bleu))
    print("Attack success rate: {:.2f}%".format(100*att_success/total_pairs))

    # # Save generation files
    # with open(f"ori_gen_{max_per}.txt", "w") as f:
    #     for line in ori_lens:
    #         f.write(str(line) + "\n")

    # with open(f"adv_gen_{max_per}.txt", "w") as f:
    #     for line in adv_lens:
    #         f.write(str(line) + "\n")

    # with open("ori_bleu.txt", "w") as f:
    #     for line in ori_bleus:
    #         f.write(str(line) + "\n")

    # with open("adv_bleu.txt", "w") as f:
    #     for line in adv_bleus:
    #         f.write(str(line) + "\n")

    # with open("ori_time.txt", "w") as f:
    #     for line in ori_time:
    #         f.write(str(line) + "\n")

    # with open("adv_time.txt", "w") as f:
    #     for line in adv_time:
    #         f.write(str(line) + "\n")



if __name__ == "__main__":
    # max_num_samples = 10
    # for max_per in range(1, 11):
    #     print("Max perturbation times: {}".format(max_per))
    #     main(max_num_samples, max_per)

    # Demo
    max_num_samples = 5
    max_per = 1
    main(max_num_samples, max_per)


