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
    AutoModelForCausalLM,
)
from datasets import load_dataset
import evaluate
from DialogueAPI import dialogue
from attacker.my_attacker import WordAttacker, StructureAttacker
from attacker.PWWS import PWWSAttacker
from attacker.SCPN import SCPNAttacker
from attacker.VIPER import VIPERAttacker

def prepare_sent(text: str):
    return text.strip().capitalize()


def generation(
    instance, 
    tokenizer, 
    model, 
    attacker, 
    bleu,
    rouge,
    meteor,
    task,
    args,
    device,
):
    # Set up
    max_source_length = args.max_len
    max_target_length = args.max_len
    num_beams = args.num_beams 
    num_beam_groups = args.num_beam_groups

    num_entries = len(instance["free_messages"])
    persona_pieces = [
        f"<PS> {prepare_sent(instance['personas'][0])}",
        f"<PS> {prepare_sent(instance['personas'][1])}",
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
        free_message = prepare_sent(instance['free_messages'][entry_idx])
        guided_message = prepare_sent(instance['guided_messages'][entry_idx])
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
        if task == 'seq2seq':
            inputs = tokenizer(
                text, 
                max_length=max_source_length, 
                return_tensors="pt",
                truncation=True,
            )
        else:
            inputs = tokenizer(
                text + tokenizer.eos_token, 
                max_length=max_source_length, 
                return_tensors="pt",
                truncation=True,
            )
        input_ids = inputs.input_ids.to(device)
        t1 = time.time()
        outputs = dialogue(
            model, 
            input_ids,
            early_stopping=False, 
            num_beams=num_beams,
            num_beam_groups=num_beam_groups, 
            use_cache=True,
            max_length=max_target_length,
        )
        if task == 'seq2seq':
            output = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:], 
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        print("U--{} (y--{})".format(free_message, guided_message))
        print("G--{}".format(output))
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
        success, adv_his = attacker.run_attack(text, guided_message)
        new_text = adv_his[-1][0]
        new_free_message = new_text.split(tokenizer.eos_token)[1].strip()
        if success:
            print("U'--{}".format(new_free_message))
        else:
            print("Attack failed!")

        if task == 'seq2seq':
            inputs = tokenizer(
                new_text, 
                max_length=max_source_length, 
                return_tensors="pt",
                truncation=True,
            )
        else:
            inputs = tokenizer(
                new_text + tokenizer.eos_token, 
                max_length=max_source_length, 
                return_tensors="pt",
                truncation=True,
            )
        input_ids = inputs.input_ids.to(device)
        t1 = time.time()
        outputs = dialogue(
            model, 
            input_ids,
            early_stopping=False, 
            num_beams=num_beams,
            num_beam_groups=num_beam_groups, 
            use_cache=True,
            max_length=max_target_length,
        )
        if task == 'seq2seq':
            output = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:], 
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        print("G'--{}".format(output))
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




def main(args):
    random.seed(args.seed)
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    max_num_samples = args.max_num_samples
    max_len = args.max_len
    max_per = args.max_per
    num_beams = args.num_beams
    num_beam_groups = args.num_beam_groups

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path, num_beams=num_beams, num_beam_groups=num_beam_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path.lower():
        task = 'clm'
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        tokenizer.add_tokens(['<PS>'], special_tokens=True) ## this line is updated
        tokenizer.add_tokens(['<CTX>'], special_tokens=True) ## this line is updated
        tokenizer.add_tokens(['<SEP>'], special_tokens=True) ## this line is updated
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
    else:
        task = 'seq2seq'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # WORD_ATT_METHODS = ['word', 'structure', 'pwws']
    # if args.attack_strategy.lower() not in WORD_ATT_METHODS:
    #     max_per = 1

    # Load dataset
    bst_dataset = load_dataset(dataset)
    test_dataset = bst_dataset['test']
    ids = random.sample(range(len(test_dataset)), max_num_samples)
    sampled_test_dataset = test_dataset.select(ids)

    # Define attack method
    if args.attack_strategy.lower() == 'word':
        attacker = WordAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif args.attack_strategy.lower() == 'structure':
        attacker = StructureAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif args.attack_strategy.lower() == 'pwws':
        attacker = PWWSAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif args.attack_strategy.lower() == 'scpn':
        attacker = SCPNAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif args.attack_strategy.lower() == 'viper':
        attacker = VIPERAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    else:
        raise ValueError("Invalid attack strategy!")


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
                generation(
                    instance, tokenizer, model, attacker, bleu, rouge, meteor, task, args, device,
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
    import argparse
    import ssl
    # import nltk
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=1, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum length of generated sequence")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups")
    parser.add_argument("--model_name_or_path", type=str, 
                        default="results/bart", 
                        choices=['results/bart', 'results/t5', 'microsoft/DialoGPT-small'],
                        help="Path to model")
    parser.add_argument("--dataset", type=str, default="blended_skill_talk", help="Dataset to attack")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--attack_strategy", "--a", type=str, 
                        default='structure', 
                        choices=['structure', 'word', 'pwws', 'scpn', 'viper'], 
                        help="Attack strategy")
    args = parser.parse_args()
    main(args)