import nltk

import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
)
from datasets import load_dataset
from my_attack import WordAttacker, StructureAttacker

nltk.download('averaged_perceptron_tagger')



def main():

    model_name_or_path = "results/" # "facebook/bart-base", "results/"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    bst_dataset = load_dataset("blended_skill_talk")
    test_dataset = bst_dataset['test']

    attacker = WordAttacker(
        device=device,
        tokenizer=tokenizer,
        model=model,
        config=config,
    )
    # attacker = StructureAttacker(
    #     device=device,
    #     tokenizer=tokenizer,
    #     model=model,
    #     config=config,
    # )

    # sentence = "Congratulations. Do you come from a big family?"
    for sentences in test_dataset['free_messages'][:2]:
        for sentence in sentences:
            print("\nOriginal sentence: ", sentence)
            input_ids = tokenizer(sentence, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            
            outputs = model.generate(input_ids, max_length=128, do_sample=False)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print("Generated sentence ({}): {}".format(outputs[0].size(-1), output))

            success, adv_his = attacker.run_attack(sentence)
            if success:
                print("Adversarial sentence: ", adv_his[-1][0])
            else:
                print("No adversarial sentence found!")

            input_ids = tokenizer(adv_his[-1][0], return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            
            outputs = model.generate(input_ids, max_length=128, do_sample=False)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print("Generated sentence ({}): {}".format(outputs[0].size(-1), output))