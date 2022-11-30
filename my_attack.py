import torch
import torch.nn as nn
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
from DialogueAPI import dialogue


class BaseAttacker:
    def __init__(self, 
                 device,
                 model,
                 config):
      
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.embedding = self.model.get_input_embeddings().weight
        self.specical_token = self.tokenizer.all_special_tokens
        self.specical_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.model.config.eos_token_id
        self.pad_token_id = self.model.config.pad_token_id
        self.num_beams = self.model.config.num_beams
        self.num_beam_groups = self.model.config.num_beam_groups
        self.max_len = self._get_hparam(config, 'max_len', 128)
        self.max_per = self._get_hparam(config, 'max_per', 1000)

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))

        return default

    def run_attack(self, x):
        pass

    def compute_loss(self, x):
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
                 model,
                 config):
        super(SlowAttacker, self).__init__(device, model, config)

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            s[:, self.pad_token_id] = 1e-12
            eos_p = self.softmax(s)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss

    def leave_eos_target_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores):
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
        seqs = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            input_ids = self.tokenizer(new_strings[st:ed], return_tensors="pt", padding=True).input_ids
            input_ids = input_ids.to(self.device)
            
            # outputs = self.model.generate_with_grad(
            #     input_ids, 
            #     num_beams=self.num_beams, 
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
            seqs.extend(outputs['sequences'].tolist())

        pred_len = np.array([self.compute_seq_len(torch.tensor(seq)) for seq in seqs])
        assert len(new_strings) == len(pred_len)
        return new_strings[pred_len.argmax()], max(pred_len)

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0] # original sentence length
        best_adv_text, best_len = deepcopy(text[0]), ori_len
        current_adv_text, current_len = deepcopy(text[0]), ori_len  # current_adv_text: List[str]
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
        ori_len, (best_adv_text, best_len), (current_adv_text, current_len) = self.prepare_attack(text)
        adv_his = [(deepcopy(current_adv_text), deepcopy(current_len), 0.0)]
        modify_pos = []

        pbar = tqdm(range(self.max_per))
        t1 = time.time()
        for it in pbar:
            loss_list = self.compute_loss([current_adv_text])
            loss = sum(loss_list)
            # print('loss: ', loss)
            self.model.zero_grad()
            loss.backward()
            grad = self.embedding.grad
            # print("embedding grad: ", grad)
            new_strings = self.mutation(current_adv_text, grad, modify_pos)
            print('new strings: ', new_strings)
            try:
                if new_strings:
                    current_adv_text, current_len = self.select_best(new_strings)
                    log_str = "%d, %d, %.2f" % (it, len(new_strings), best_len / ori_len)
                    pbar.set_description(log_str)
                    if current_len > best_len:
                        best_adv_text = deepcopy(current_adv_text)
                        best_len = current_len
                    t2 = time.time()
                    adv_his.append((best_adv_text, int(best_len), t2 - t1))
                else:
                    return False, adv_his
            except:
                return False, adv_his
            
        return True, adv_his




class WordAttacker(SlowAttacker):
    def __init__(self, 
                 device,
                 model,
                 config):
        super(WordAttacker, self).__init__(device, model, config)

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        return loss_list
    

    def token_replace_mutation(self, current_adv_text, grad, modified_pos):
        new_strings = []
        current_ids = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        base_ids = current_ids.clone()
        for pos in modified_pos:
            t = current_ids[0][pos]
            grad_t = grad[t]
            score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
            index = score.argsort()
            for tgt_t in index:
                if tgt_t not in self.specical_token:
                    base_ids[pos] = tgt_t
                    break

        current_text = self.tokenizer.decode(current_ids)
        for pos, t in enumerate(current_ids):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        new_base_ids = base_ids.clone()
                        new_base_ids[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_ids)
                        # if new_tag[pos][:2] == ori_tag[pos][:2]:
                        new_strings.append(candidate_s)
                        cnt += 1
                        if cnt >= 50:
                            break

        return new_strings


    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)
        return new_strings




class StructureAttacker(SlowAttacker):
    def __init__(self, 
                 device,
                 model,
                 config):
        super(StructureAttacker, self).__init__(device, model, config)

        bertmodel = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.bertmodel = bertmodel.eval().to(self.device)
        self.num_of_perturb = 50

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


    def perturbBert(self, tokens, ori_tensors, masked_indexL, masked_index):
        new_sentences = list()
        # invalidChars = set(string.punctuation)

        # For each idx, use Bert to generate k (i.e., num) candidate tokens
        original_word = tokens[masked_index]
        low_tokens = [x.lower() for x in tokens]
        low_tokens[masked_index] = '[MASK]'

        # Try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(self.model.device)
            prediction = self.bertmodel(tokens_tensor)

            # Skip the sentences that contain unknown words
            # Another option is to mark the unknow words as [MASK]; we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            return new_sentences

        # get the similar words
        topk_Idx = torch.topk(prediction[0][0, masked_index], self.num_of_perturb)[1].tolist()
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_Idx)

        # Remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # This step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

        # Generate similar sentences
        for t in topk_tokens:
            # if any(char in invalidChars for char in t):
            #     continue
            tokens[masked_index] = t
            new_pos_inf = nltk.tag.pos_tag(tokens)

            # only use the similar sentences whose similar token's tag is still the same
            if new_pos_inf[masked_index][1][:2] == masked_indexL[masked_index][1][:2]:
                new_t = self.tokenizer.encode(tokens[masked_index])[0]
                new_tensor = ori_tensors.clone()
                new_tensor[masked_index] = new_t
                new_sentence = self.tokenizer.decode(new_tensor)
                new_sentences.append(new_sentence)
        tokens[masked_index] = original_word
        return new_sentences



    def structure_mutation(self, current_adv_text, grad):
        new_strings = []
        important_tensor = (-grad.sum(1)).argsort()
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        ori_tokens, ori_tag = self.get_token_type(current_tensor)

        assert len(ori_tokens) == len(current_tensor)
        assert len(ori_tokens) == len(ori_tag)
        
        current_tensor_list = current_tensor.tolist()
        for t in important_tensor:
            if int(t) not in current_tensor_list:
                continue
            pos_list = torch.where(current_tensor.eq(int(t)))[0].tolist()
            for pos in pos_list:
                new_string = self.perturbBert(ori_tokens, current_tensor, ori_tag, pos)
                new_strings.extend(new_string)
            if len(new_strings) > 2000:
                break

        return new_strings


    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.structure_mutation(current_adv_text, grad)
        return new_strings





if __name__ == "__main__":
    nltk.download('averaged_perceptron_tagger')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained("facebook/bart-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    # attacker = WordAttacker(
    #     device=device,
    #     model=model,
    #     config=config,
    # )
    attacker = StructureAttacker(
        device=device,
        model=model,
        config=config,
    )

    sentence = "Congratulations. Do you come from a big family?"
    adv_text = attacker.run_attack(sentence)
    print(adv_text)