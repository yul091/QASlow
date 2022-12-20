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
import OpenAttack
from DialogueAPI import dialogue
from my_attacker import WordAttacker, StructureAttacker



def get_prediction(sentence, model, tokenizer, num_beams, num_beam_groups, max_len, device):
    def remove_pad(s):
        for i, tk in enumerate(s):
            if tk == tokenizer.eos_token_id and i != 0:
                return s[:i + 1]
        return s

    input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
    outputs = dialogue(
        model, 
        input_ids,
        early_stopping=False, 
        num_beams=num_beams,
        num_beam_groups=num_beam_groups, 
        use_cache=True,
        max_length=max_len,
    )
    
    seqs = outputs['sequences']
    seqs = [remove_pad(seq) for seq in seqs]
    out_scores = outputs['scores']
    pred_len = [compute_seq_len(seq, tokenizer) for seq in seqs]
    return pred_len, seqs, out_scores


def compute_seq_len(seq, tokenizer):
    if seq[0].eq(tokenizer.pad_token_id):
        return int(len(seq) - sum(seq.eq(tokenizer.pad_token_id)))
    else:
        return int(len(seq) - sum(seq.eq(tokenizer.pad_token_id))) - 1


def compute_score(text, model, tokenizer, num_beams, num_beam_groups, max_len, device):
    batch_size = len(text)
    index_list = [i * num_beams for i in range(batch_size + 1)]
    pred_len, seqs, out_scores = get_prediction(
        text, model, tokenizer, num_beams, num_beam_groups, max_len, device
    )
    scores = [[] for _ in range(batch_size)]
    for out_s in out_scores:
        for i in range(batch_size):
            current_index = index_list[i]
            scores[i].append(out_s[current_index: current_index + 1])
    scores = [torch.cat(s) for s in scores]
    scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
    return scores, seqs, pred_len


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
        # outputs = model.generate(input_ids, max_length=max_target_length, do_sample=False)
        outputs = dialogue(
            model, 
            input_ids,
            early_stopping=False, 
            num_beams=4,
            num_beam_groups=1, 
            use_cache=True,
            max_length=max_target_length,
        )
        output = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        t2 = time.time()
        print("U--{}".format(free_message))
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
        success, adv_his = attacker.run_attack(text)
        new_text = adv_his[-1][0]
        new_free_message = new_text.split(tokenizer.eos_token)[1].strip()
        if success:
            print("U'--{}".format(new_free_message))
        else:
            print("Attack failed!")

        input_ids = tokenizer(new_text, max_length=max_source_length, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        t1 = time.time()
        outputs = dialogue(
            model, 
            input_ids,
            early_stopping=False, 
            num_beams=4,
            num_beam_groups=1, 
            use_cache=True,
            max_length=max_target_length,
        )
        output = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # Load dataset
    bst_dataset = load_dataset(dataset)
    test_dataset = bst_dataset['test']
    ids = random.sample(range(len(test_dataset)), max_num_samples)
    sampled_test_dataset = test_dataset.select(ids)

    # Define attack method
    if args.attack_strategy == 'word':
        attacker = WordAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
        )
    else:
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


class MyClassifier(OpenAttack.Classifier):
    def __init__(self, model, tokenizer, num_beams, num_beam_groups, max_len, device):
        nltk.download('vader_lexicon')
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.max_len = max_len
        self.device = device

    def get_pred(self, input_):
        def remove_pad(s):
            for i, tk in enumerate(s):
                if tk == self.eos_token_id and i != 0:
                    return s[:i + 1]
            return s

        seqs = self.get_prob(input_).argmax(axis=-1) # (batch_size, max_len)
        seqs = [remove_pad(seq) for seq in seqs]
        return seqs
        
    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        """
        input_: a list of sentences
        """
        # ret = []
        # for sent in input_:
            # # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
            # res = self.model.polarity_scores(sent)
            # # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
            # # Adding 10^−6 is a trick to avoid dividing by zero.
            # prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)
            # ret.append(np.array([1 - prob, prob]))
            # input_ids = self.tokenizer(sent, return_tensors="pt").input_ids.to(self.device)
            # # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
            # outputs = dialogue(
            #     model, 
            #     input_ids,
            #     early_stopping=False, 
            #     num_beams=self.num_beams,
            #     num_beam_groups=self.num_beam_groups, 
            #     use_cache=True,
            #     max_length=self.max_len,
            # )
            # out_scores = outputs['scores']
            # print("sequences: ", outputs['sequences'].shape)
            # print("out_scores: ", [x.shape for x in out_scores])
        scores, seqs, pred_len = compute_score(
            input_, self.model, self.tokenizer, 
            self.num_beams, self.num_beam_groups, self.max_len, self.device
        )
        # print("scores: ", scores)
        # ret.append(scores.detach().cpu().numpy())
        scores = torch.stack(scores, dim=0).detach().cpu().numpy()
        print(scores.shape)

        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return scores



if __name__ == "__main__":
    import nltk
    import argparse
    import ssl
    from itertools import chain
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=1, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum length of generated sequence")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups")
    parser.add_argument("--model_name_or_path", type=str, default="results/bart", help="Path to model")
    parser.add_argument("--dataset", type=str, default="blended_skill_talk", help="Dataset to attack")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--attack_strategy", "--a", type=str, 
                        default='structure', 
                        choices=['structure', 'word', 'pwws'], 
                        help="Attack strategy")
    args = parser.parse_args()
    # main(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attacker = OpenAttack.attackers.GANAttacker()
    # victim = MyClassifier(model, tokenizer, args.num_beams, args.num_beam_groups, args.max_len, device)
    victim = OpenAttack.classifiers.TransformersClassifier(
        model=model,
        tokenizer=tokenizer,
        device=device,
        embedding_layer=model.get_input_embeddings(),
        max_length=args.max_len,
        batch_size=5,
        lang="english",
    )
    attack_eval = OpenAttack.AttackEval(
        attacker=attacker, 
        victim=victim,
        language="english",
        tokenizer=tokenizer,
        metrics=[OpenAttack.metric.BLEU(tokenizer), OpenAttack.metric.Fluency()],
    )
    bst_dataset = load_dataset(args.dataset)
    test_dataset = bst_dataset['test']
    column_names = bst_dataset['test'].column_names
    # Data processing
    def preprocess_bst(examples):
        num_entries = len(examples["free_messages"])
        persona_pieces = [
            f"<PS> {examples['personas'][0]}",
            f"<PS> {examples['personas'][1]}",
        ]
        if examples['context'] == "wizard_of_wikipedia":
            additional_context_pieces = [f"<CTX> {examples['additional_context']}. <SEP> "]
        else:
            additional_context_pieces = ["<SEP> "]

        previous_utterance_pieces = examples["previous_utterance"]
        inputs, labels = [], []
        for entry_idx in range(num_entries):
            free_message = examples['free_messages'][entry_idx]
            guided_message = examples['guided_messages'][entry_idx]

            previous_utterance = ' <SEP> '.join(previous_utterance_pieces)
            original_context = ' '.join(
                persona_pieces + additional_context_pieces
            ) + previous_utterance
            # Input & Output
            text = original_context + ' ' + tokenizer.eos_token + ' ' + free_message
            inputs.append(text)
            labels.append(guided_message)
            previous_utterance_pieces += [
                free_message,
                guided_message,
            ]

        return {
            "x": inputs,
            "y": labels,
        }


    def group_texts(examples):
        # ['input_ids', 'attention_mask', 'labels']
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        return concatenated_examples

    test_dataset = test_dataset.map(
        preprocess_bst,
        batched=False,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=False,
    )
    attack_eval.eval(test_dataset, visualize=False)


