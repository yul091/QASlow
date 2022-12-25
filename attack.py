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


class DGAttack:
    def __init__(self, args, tokenizer, model, attacker, device, task, bleu, rouge, meteor):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.attacker = attacker
        self.dataset = args.dataset
        self.task = task
        self.device = device

        self.max_source_length = args.max_len
        self.max_target_length = args.max_len
        self.num_beams = args.num_beams 
        self.num_beam_groups = args.num_beam_groups

        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor

        self.ori_lens, self.adv_lens = [], []
        self.ori_bleus, self.adv_bleus = [], []
        self.ori_rouges, self.adv_rouges = [], []
        self.ori_meteors, self.adv_meteors = [], []
        self.ori_time, self.adv_time = [], []
        self.att_success = 0
        self.total_pairs = 0

    def prepare_sent(self, text: str):
        return text.strip().capitalize()

    def prepare_context(self, instance):
        if self.dataset == 'blended_skill_talk':
            if self.task == 'seq2seq':
                num_entries = len(instance["free_messages"])
                persona_pieces = [
                    # f"<PS>{prepare_sent(instance['personas'][0])}",
                    f"<PS>{self.prepare_sent(instance['personas'][1])}",
                ]
                if instance['context'] == "wizard_of_wikipedia":
                    additional_context_pieces = [f"<CTX>{self.prepare_sent(instance['additional_context'])}.<SEP>"]
                else:
                    additional_context_pieces = ["<SEP>"]
                context = ' '.join(persona_pieces + additional_context_pieces)
                prev_utt_pc = [self.prepare_sent(sent) for sent in instance["previous_utterance"]]
            else:
                num_entries = min(len(instance["free_messages"]), 2)
            total_entries = num_entries

        elif self.dataset == 'conv_ai_2':
            total_entries = len(instance['dialog'])
            if self.task == 'seq2seq':
                user_profile = ' '.join([''.join(x) for x in instance['user_profile']])
                persona_pieces = f"<PS>{user_profile}"
                num_entries = len([x for x in instance['dialog'] if x['sender_class'] == 'Human'])
                prev_utt_pc = [persona_pieces]
                context = persona_pieces
            else:
                num_entries = min(len([x for x in instance['dialog'] if x['sender_class'] == 'Human']), 2)
        else:
            raise ValueError("Dataset not supported.")

        return num_entries, total_entries, context, prev_utt_pc


    def prepare_entry(self, instance, entry_idx, context, prev_utt_pc, total_entries):
        if self.dataset == 'blended_skill_talk':
            free_message = self.prepare_sent(instance['free_messages'][entry_idx])
            guided_message = self.prepare_sent(instance['guided_messages'][entry_idx])
            if self.task == 'seq2seq':
                previous_utterance = '<SEP>'.join(prev_utt_pc)
                original_context = context + previous_utterance
            else:
                previous_utterance = ' '.join(prev_utt_pc)
                original_context = previous_utterance
        elif self.dataset == 'conv_ai_2':
            free_message = instance['dialog'][entry_idx*2]['text']
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx*2+1]['text']
            if self.task == 'seq2seq':
                original_context = '<SEP>'.join(prev_utt_pc)
            else:
                original_context = ' '.join(prev_utt_pc)
        else:
            raise ValueError("Dataset not supported.")

        return free_message, guided_message, original_context


    def get_prediction(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text 
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text, 
            max_length=self.max_source_length, 
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                self.model, 
                input_ids,
                early_stopping=False, 
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, 
                use_cache=True,
                max_length=self.max_target_length,
            )
        if self.task == 'seq2seq':
            output = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = self.tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:], 
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        return output, t2 - t1


    def eval_metrics(self, output, guided_message):
        if not output:
            return

        bleu_res = self.bleu.compute(
            predictions=[output], 
            references=[[guided_message]],
        )
        rouge_res = self.rouge.compute(
            predictions=[output],
            references=[guided_message],
        )
        meteor_res = self.meteor.compute(
            predictions=[output],
            references=[guided_message],
        )
        pred_len = bleu_res['translation_length']
        return bleu_res, rouge_res, meteor_res, pred_len
        
        
    def generation_step(self, instance):
        # Set up
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)

        for entry_idx in range(num_entries):
            free_message, guided_message, original_context = self.prepare_entry(
                instance, 
                entry_idx, 
                context, 
                prev_utt_pc,
                total_entries,
            )
            if guided_message is None:
                continue
            print("\nDialogue history: {}".format(original_context))
            prev_utt_pc += [
                free_message,
                guided_message,
            ]
            # Original generation
            if self.task == 'seq2seq':
                text = original_context + self.tokenizer.eos_token + free_message
            else:
                text = original_context + '<SEP>' + free_message

            output, time_gap = self.get_prediction(text)  
            print("U--{} (y--{})".format(free_message, guided_message))
            print("G--{}".format(output))
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, guided_message)
            print("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.ori_lens.append(pred_len)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)
            
            # Attack
            success, adv_his = self.attacker.run_attack(text, guided_message)
            new_text = adv_his[-1][0]
            if self.task == 'seq2seq':
                new_free_message = new_text.split(self.tokenizer.eos_token)[1].strip()
            else:
                new_free_message = new_text.split('<SEP>')[1].strip()
            if success:
                print("U'--{}".format(new_free_message))
            else:
                print("Attack failed!")

            output, time_gap = self.get_prediction(new_text)
            print("G'--{}".format(output))
            bleu_res, rouge_res, meteor_res, adv_pred_len = self.eval_metrics(output, guided_message)
            print("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                adv_pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.adv_lens.append(adv_pred_len)
            self.adv_bleus.append(bleu_res['bleu'])
            self.adv_rouges.append(rouge_res['rougeL'])
            self.adv_meteors.append(meteor_res['meteor'])
            self.adv_time.append(time_gap)
            # ASR
            self.att_success += (adv_pred_len > pred_len)
            self.total_pairs += num_entries


    def generation(self, test_dataset):
        for i, instance in tqdm(enumerate(test_dataset)):
            self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        return Ori_len, Adv_len, Ori_bleu, Adv_bleu, Ori_rouge, Adv_rouge, \
            Ori_meteor, Adv_meteor, Ori_t, Adv_t


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
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
    else:
        task = 'seq2seq'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # WORD_ATT_METHODS = ['word', 'structure', 'pwws']
    # if args.attack_strategy.lower() not in WORD_ATT_METHODS:
    #     max_per = 1

    # Load dataset
    all_datasets = load_dataset(dataset)
    if dataset == "conv_ai_2":
        test_dataset = all_datasets['train']
    else:
        test_dataset = all_datasets['test']
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

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Define DG attack
    dg = DGAttack(
        args=args,
        tokenizer=tokenizer,
        model=model,
        attacker=attacker,
        device=device,
        task=task,
        bleu=bleu,
        rouge=rouge,
        meteor=meteor,
    )
    Ori_len, Adv_len, Ori_bleu, Adv_bleu, Ori_rouge, Adv_rouge, \
        Ori_meteor, Adv_meteor, Ori_t, Adv_t = dg.generation(sampled_test_dataset)

    # Summarize eval results
    
    print("Original output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
        Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
    ))
    print("Perturbed output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
        Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
    ))
    print("Attack success rate: {:.2f}%".format(100*dg.att_success/dg.total_pairs))

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
    parser.add_argument("--max_num_samples", type=int, default=5, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups")
    parser.add_argument("--model_name_or_path", type=str, 
                        default="results/bart", 
                        choices=[
                            'results/bart', 
                            'results/t5', 
                            'results/dialogpt', 
                            'microsoft/DialoGPT-small',
                        ],
                        help="Path to model")
    parser.add_argument("--dataset", type=str, 
                        default="blended_skill_talk", 
                        choices=[
                            "blended_skill_talk",
                            "conv_ai_2",
                        ], 
                        help="Dataset to attack")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--attack_strategy", "--a", type=str, 
                        default='structure', 
                        choices=['structure', 'word', 'pwws', 'scpn', 'viper'], 
                        help="Attack strategy")
    args = parser.parse_args()
    main(args)