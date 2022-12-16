import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import nltk
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
)
import stanza
from nltk.corpus import wordnet as wn
from DialogueAPI import dialogue
from utils import GrammarChecker
import pdb


class BaseAttacker:
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3):
      
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.embedding = self.model.get_input_embeddings().weight
        self.special_token = self.tokenizer.all_special_tokens
        self.special_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_beams = self.model.config.num_beams
        self.num_beam_groups = self.model.config.num_beam_groups
        self.max_len = max_len
        self.max_per = max_per

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))

        return default

    def run_attack(self, text):
        pass

    def compute_loss(self, text):
        pass

    def compute_seq_len(self, seq):
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1

    def get_prediction(self, sentence):
        def remove_pad(s):
            for i, tk in enumerate(s):
                if tk == self.eos_token_id and i != 0:
                    return s[:i + 1]
            return s

        input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids.to(self.device)
        # ['sequences', 'sequences_scores', 'scores', 'beam_indices'] if num_beams != 1
        # ['sequences', 'scores'] if num_beams = 1
        # outputs = self.model.generate(
        #     input_ids, 
        #     num_beams=self.num_beams, 
        #     num_beam_groups=self.num_beam_groups,
        #     output_scores=True, 
        #     max_length=self.max_len,
        #     return_dict_in_generate=True
        # )
        outputs = dialogue(
            self.model, 
            input_ids,
            early_stopping=False, 
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups, 
            use_cache=True,
            max_length=self.max_len,
        )
        
        seqs = outputs['sequences']
        seqs = [remove_pad(seq) for seq in seqs]
        out_scores = outputs['scores']
        pred_len = [self.compute_seq_len(seq) for seq in seqs]
        return pred_len, seqs, out_scores

    def get_trans_string_len(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        return seqs[0], pred_len[0]

    def get_trans_len(self, text):
        pred_len, _, _ = self.get_prediction(text)
        return pred_len

    def get_trans_strings(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        out_res = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
        return out_res, pred_len
    
    def compute_score(self, text):
        batch_size = len(text)
        index_list = [i * self.num_beams for i in range(batch_size + 1)]
        pred_len, seqs, out_scores = self.get_prediction(text)


        scores = [[] for _ in range(batch_size)]
        for out_s in out_scores:
            for i in range(batch_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]
        scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len
        




class SlowAttacker(BaseAttacker):
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3):
        super(SlowAttacker, self).__init__(device, tokenizer, model, max_len, max_per)

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            s[:, self.pad_token_id] = 1e-12 # T X V
            eos_p = self.softmax(s)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss

    def leave_eos_target_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores): # s: T X V
            # if self.pad_token_id != self.eos_token_id:
            s[:, self.pad_token_id] = 1e-12
            softmax_v = self.softmax(s)
            eos_p = softmax_v[:pred_len[i], self.eos_token_id]
            target_p = torch.stack([softmax_v[idx, s] for idx, s in enumerate(seqs[i][1:])])
            target_p = target_p[:pred_len[i]]
            pred = eos_p + target_p
            pred[-1] = pred[-1] / 2
            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss

    @torch.no_grad()
    def select_best(self, new_strings, batch_size=30):
        """
        Select generated strings which induce longest output sentences.
        """
        pred_len = []
        # seqs = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1

        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            batch_strings = [x[1] for x in new_strings[st:ed]]
            inputs = self.tokenizer(batch_strings, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(self.device)
            outputs = self.model.generate(
                input_ids, 
                num_beams=self.num_beams, 
                max_length=self.max_len,
                return_dict_in_generate=True,
            )
            lengths = [self.compute_seq_len(seq) for seq in outputs['sequences']]
            # pdb.set_trace()
            pred_len.extend(lengths)
            
        # pred_len = np.array([self.compute_seq_len(torch.tensor(seq)) for seq in seqs])
        pred_len = np.array(pred_len)
        # pdb.set_trace()

        assert len(new_strings) == len(pred_len)
        return new_strings[pred_len.argmax()], max(pred_len)

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0] # original sentence length
        best_adv_text, best_len = deepcopy(text), ori_len
        current_adv_text, current_len = deepcopy(text), ori_len  # current_adv_text: List[str]
        return ori_len, (best_adv_text, best_len), (current_adv_text, current_len)

    def compute_loss(self, text):
        raise NotImplementedError

    def mutation(self, current_adv_text, grad, modified_pos):
        raise NotImplementedError

    def run_attack(self, text):
        """
        (1) Using gradient ascent to generate adversarial sentences -- mutation();
        (2) Select the best samples which induce longest output sentences -- select_best();
        (3) Save the adversarial samples -- adv_his.
        """
        assert len(text) != 1
        # torch.autograd.set_detect_anomaly(True)
        ori_len, (best_adv_text, best_len), (cur_adv_text, cur_len) = self.prepare_attack(text)
        # adv_his = [(deepcopy(cur_adv_text), deepcopy(cur_len), 0.0)]
        adv_his = []
        modify_pos = [] # record already modified positions (avoid recovering to the original token)
        pbar = tqdm(range(self.max_per))
        t1 = time.time()

        for it in pbar:
            loss_list = self.compute_loss([cur_adv_text])
            loss = sum(loss_list)
            self.model.zero_grad()
            loss.backward()
            grad = self.embedding.grad
            new_strings = self.mutation(cur_adv_text, grad, modify_pos)

            if new_strings:
                # print("new_strings: ", new_strings)
                (cur_pos, cur_adv_text), cur_len = self.select_best(new_strings)
                modify_pos.append(cur_pos)
                log_str = "%d, %d, %.2f" % (it, len(new_strings), best_len / ori_len)
                pbar.set_description(log_str)

                if cur_len > best_len:
                    best_adv_text = deepcopy(cur_adv_text)
                    best_len = cur_len
                t2 = time.time()
                adv_his.append((best_adv_text, int(best_len), t2 - t1))

        if adv_his:
            return True, adv_his
        else:
            return False, [(deepcopy(cur_adv_text), deepcopy(cur_len), 0.0)]



class WordAttacker(SlowAttacker):
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3):
        super(WordAttacker, self).__init__(device, tokenizer, model, max_len, max_per)


    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text) # list of [T X V], [T], [1]
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        return loss_list
    

    def token_replace_mutation(self, current_adv_text, grad, modified_pos):
        new_strings = []
        words = self.tokenizer.tokenize(current_adv_text)
        current_inputs = self.tokenizer(current_adv_text, return_tensors="pt", padding=True)
        current_ids = current_inputs.input_ids[0].to(self.device)
        base_ids = current_ids.clone()

        # masked_texts = self.get_masked_sentence(current_adv_text)
        # all_candidates = []
        # for masked_text in masked_texts:
        #     top_k_tokens = self.get_masked_predictions(masked_text)
        #     all_candidates.append(top_k_tokens)

        # current_text = self.tokenizer.decode(current_ids, skip_special_tokens=True)
        # print("current ids: ", current_ids)
        for pos, t in enumerate(current_ids):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        new_base_ids = base_ids.clone()
                        new_base_ids[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_ids, skip_special_tokens=True)
                        # if new_tag[pos][:2] == ori_tag[pos][:2]:
                        new_strings.append(candidate_s)
                        cnt += 1
                        if cnt >= 50:
                            break

        return new_strings


    def mutation(self, current_adv_text, grad, modified_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modified_pos)
        # print('new strings: ', new_strings)
        return new_strings




class StructureAttacker(SlowAttacker):
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3):
        super(StructureAttacker, self).__init__(device, tokenizer, model, max_len, max_per)

        # BERT initialization
        self.berttokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
        self.mask_token = self.berttokenizer.mask_token
        bertmodel = AutoModelForMaskedLM.from_pretrained('bert-large-uncased')
        self.bertmodel = bertmodel.eval().to(self.device)
        self.num_of_perturb = 50
        self.grammar = GrammarChecker()

        self.pos_dict = {'NOUN': 'n', 'VERB': 'v', 'ADV': 'r', 'ADJ': 'a'}
        self.pos_processor = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma')
        self.skip_pos_tags = ['DT', 'PDT', 'POS', 'PRP', 'PRP$', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'NNP']


    def get_pos(self, sentence, mask_index):
        processed_sentence = self.pos_processor(sentence)
        pos_list = []
        word_lemma = None

        for sentence in processed_sentence.sentences:
            for i, word in enumerate(sentence.words):
                pos_list.append(word.upos)
                if i == mask_index:
                    word_lemma = word.lemma

        return pos_list, word_lemma

    @staticmethod
    def get_word_antonyms(word):
        antonyms_lists = set()
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms_lists.add(l.antonyms()[0].name())
        return list(antonyms_lists)


    def get_synonyms(self, word, pos):
        if word is None:
            return []
        if pos not in self.pos_dict.keys():
            return []

        synonyms = set()
        for syn in wn.synsets(word):
            if syn.pos() == self.pos_dict[pos]:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())

        if word in synonyms:
            synonyms.remove(word)

        return list(synonyms)


    def formalize(self, text):
        tokens = self.berttokenizer.tokenize(text)

        if len(tokens) > self.max_len:
            tokens = tokens[0:self.max_len]

        string = self.berttokenizer.convert_tokens_to_string(tokens)
        return string


    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        return loss_list


    def get_token_type(self, input_tensor):
        # tokens = self.tree_tokenizer.tokenize(sent)
        tokens = self.tokenizer.convert_ids_to_tokens(input_tensor)
        # tokens = [tk.replace(self.space_token, '') for tk in tokens]
        pos_inf = nltk.tag.pos_tag(tokens)
        bert_masked_indexL = list()
        # collect the token index for substitution
        for idx, (word, tag) in enumerate(pos_inf):
            # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
            # if tag.startswith('NN') or tag.startswith('JJ'):
            #     tagFlag = tag[:2]
                # we do not perturb the first and the last token because BERT's performance drops on for those positions
            # if idx != 0 and idx != len(tokens) - 1:
            bert_masked_indexL.append((idx, tag))

        return tokens, bert_masked_indexL


    def perturbBert(self, cur_text, cur_tokens, cur_tags, cur_error, masked_index):
        new_sentences = []
        # invalidChars = set(string.punctuation)

        # For each idx, use BERT to generate k (i.e., num) candidate tokens
        cur_tok = cur_tokens[masked_index]
        low_tokens = [x.lower() for x in cur_tokens]
        low_tokens[masked_index] = self.mask_token

        # Get the pos tag & synonyms of the masked word
        # pos_list, word_lemma = self.get_pos(cur_text, masked_index)
        # masked_word_pos = pos_list[masked_index]
        # synonyms = self.get_synonyms(word_lemma, masked_word_pos)
        antonyms = self.get_word_antonyms(cur_tok)
        # print("antonyms: ", antonyms)

        # Try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            prediction = self.bertmodel(tokens_tensor)[0]

            # Skip the sentences that contain unknown words
            # Another option is to mark the unknow words as [MASK]; 
            # we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            return new_sentences

        # Get the similar words
        probs = F.softmax(prediction[0, masked_index], dim=-1)
        topk_Idx = torch.topk(probs, self.num_of_perturb, sorted=True)[1].tolist()
        topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)

        # Remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # This step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))
        topk_tokens = set(topk_tokens) - set(self.berttokenizer.all_special_tokens)
        topk_tokens = list(topk_tokens - set(antonyms))
        # topk_tokens = list(topk_tokens & set(synonyms))
        # print("topk_tokens: ", topk_tokens)

        # Generate similar sentences
        for tok in topk_tokens:
            # if any(char in invalidChars for char in t):
            #     continue
            cur_tokens[masked_index] = ' '+tok
            new_pos_inf = nltk.tag.pos_tag(cur_tokens)

            # Only use sentences whose similar token's tag is still the same
            if new_pos_inf[masked_index][1] == cur_tags[masked_index][1]:
                # print("[index {}] substituted: {}, pos tag: {}".format(
                #     masked_index, tok, new_pos_inf[masked_index][1],
                # ))
                # new_t = self.tokenizer.encode(tokens[masked_index], add_special_tokens=False)[0]
                # new_tensor = ori_tensors.clone()
                # new_tensor[masked_index] = new_t
                # new_sentence = self.tokenizer.decode(new_tensor, skip_special_tokens=True)
                # new_sentence = self.formalize(new_sentence)
                new_sentence = self.tokenizer.convert_tokens_to_string(cur_tokens)
                # print("new sentence: ", new_sentence)
                new_error = self.grammar.check(new_sentence)
                if new_error <= cur_error:
                    new_sentences.append((masked_index, new_sentence))

        cur_tokens[masked_index] = cur_tok
        return new_sentences


    def structure_mutation(self, cur_adv_text, grad, modified_pos):
        """
        cur_adv_text (string): the current adversarial text;
        grad (tensor[V X E]): the gradient of the current adversarial text.
        """
        all_new_strings = []
        important_tensor = (-grad.sum(1)).argsort() # sort token ids w.r.t. gradient
        important_tokens = self.tokenizer.convert_ids_to_tokens(important_tensor.tolist())
        cur_input = self.tokenizer(cur_adv_text, return_tensors="pt", add_special_tokens=False)
        cur_tensor = cur_input['input_ids'][0]
        cur_tokens, cur_tags = self.get_token_type(cur_tensor)
        cur_error = self.grammar.check(cur_adv_text)

        assert len(cur_tokens) == len(cur_tensor)
        assert len(cur_tokens) == len(cur_tags)


        # For each important token (w.r.t. gradient), perturb it using BERT
        # if it is in the current text
        for tok in important_tokens:
            if tok not in cur_tokens or tok in set(self.special_token):
                continue

            pos_list = [i for i, x in enumerate(cur_tokens) if x == tok]
            # print("\ncurrent key token: {}, pos tag: {}".format(tok, cur_tags[pos_list[0]][1]))

            for pos in pos_list:
                if (cur_tags[pos][1] not in self.skip_pos_tags) and (pos not in modified_pos):
                    new_strings = self.perturbBert(cur_adv_text, cur_tokens, cur_tags, cur_error, pos)
                    all_new_strings.extend(new_strings)
            if len(all_new_strings) > 2000:
                break

        return all_new_strings


    def mutation(self, current_adv_text, grad, modified_pos):
        new_strings = self.structure_mutation(current_adv_text, grad, modified_pos)
        return new_strings



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(preds, labels, metric, tokenizer):
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(labels, list):
        labels = [labels]
    preds, labels = postprocess_text(preds, labels)
    result = metric.compute(predictions=preds, references=labels)
    return result['score']


def inference(sentence, label, model, tokenizer, metric, device):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    t1 = time.time()
    outputs = model.generate(input_ids, max_length=64, do_sample=False)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    t2 = time.time()
    prediction_len = len(output.split())
    eval_score = compute_metrics(output, label, metric, tokenizer)

    success, adv_his = attacker.run_attack(sentence)
    print("\nU--{}".format(sentence))
    print("G--{}".format(output))
    print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(
        prediction_len, t2-t1, eval_score,
    ))

    if success:
        print("U'--{}".format(adv_his[-1][0]))
    else:
        print("Attack failed!")

    input_ids = tokenizer(adv_his[-1][0], return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    t1 = time.time()
    outputs = model.generate(input_ids, max_length=64, do_sample=False)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    t2 = time.time()
    prediction_len = len(output.split())
    print("G'--{}".format(output))
    eval_score = compute_metrics(output, label, metric, tokenizer)
    print("(length: {}, latency: {:.3f}, BLEU: {:.3f})".format(
        prediction_len, t2-t1, eval_score,
    ))



if __name__ == "__main__":

    from datasets import load_metric
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')

    model_name_or_path = "results/" # "facebook/bart-base", "results/"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # attacker = WordAttacker(
    #     device=device,
    #     tokenizer=tokenizer,
    #     model=model,
    #     max_len=64,
    #     max_per=1,
    # )

    attacker = StructureAttacker(
        device=device,
        tokenizer=tokenizer,
        model=model,
        max_len=128,
        max_per=5,
    )

    metric = load_metric("sacrebleu")

    # Demo 1
    # input_text = "Do you come from a big family?"
    # output_text = "I don't, just 2 siblings in a small family. "
    input_text = "Can't believe the kid grew up so quick."
    output_text = "Yeah, kids grow up so quickly."
    inference(input_text, output_text, model, tokenizer, metric, device)

    # Demo 2
    input_text = "How would I start rock climbing?"
    output_text = "You can google it. But I suggest you to find a local climbing gym and take a class."
    inference(input_text, output_text, model, tokenizer, metric, device)

    # Demo 3
    input_text = "How often do you use computers?"
    output_text = "Almost every week. I use them for work and personal use."
    inference(input_text, output_text, model, tokenizer, metric, device)
    


