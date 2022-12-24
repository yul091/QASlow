import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from DialogueAPI import dialogue


class BaseAttacker:
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3,
                 task='seq2seq'):
      
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.embedding = self.model.get_input_embeddings().weight
        self.special_token = self.tokenizer.all_special_tokens
        self.special_id = self.tokenizer.all_special_ids
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_beams = self.model.config.num_beams
        self.num_beam_groups = self.model.config.num_beam_groups
        self.max_len = max_len
        self.max_per = max_per
        self.task = task
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
        # print("sentence: ", sentence)
        def remove_pad(s):
            for i, tk in enumerate(s):
                if tk == self.eos_token_id and i != 0:
                    return s[:i + 1]
            return s

        if self.task == 'seq2seq':
            text = sentence
        else:
            if isinstance(sentence, list):
                text = [s + self.eos_token for s in sentence]
            elif isinstance(sentence, str):
                text = sentence + self.eos_token
            else:
                raise ValueError("sentence should be a list of string or a string")

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_len,
        )
        input_ids = inputs['input_ids'].to(self.device)
        # print("input_ids ({}): {}".format(input_ids.shape, input_ids))
        # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
        outputs = dialogue(
            self.model, 
            input_ids,
            early_stopping=False, 
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups, 
            use_cache=True,
            max_length=self.max_len,
        )
        if self.task == 'seq2seq':
            seqs = outputs['sequences'].detach()
        else:
            seqs = outputs['sequences'][:, input_ids.shape[-1]:].detach()
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

    def compute_batch_score(self, text):
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
    
    def compute_score(self, text, batch_size=None):
        total_size = len(text)
        if batch_size is None:
            batch_size = len(text)

        if batch_size < total_size:
            scores, seqs, pred_len = [], [], []
            for start in range(0, total_size, batch_size):
                end = min(start + batch_size, total_size)
                score, seq, p_len = self.compute_batch_score(text[start: end])
                pred_len.extend(p_len)
                seqs.extend(seq)
                scores.extend(score)
        else:
            scores, seqs, pred_len = self.compute_batch_score(text)
        return scores, seqs, pred_len



class SlowAttacker(BaseAttacker):
    def __init__(self, 
                 device,
                 tokenizer,
                 model,
                 max_len=64,
                 max_per=3,
                 task='seq2seq'):
        super(SlowAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )

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
            target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i][1:])])
            target_p = target_p[:pred_len[i]]
            pred = eos_p + target_p
            pred[-1] = pred[-1] / 2
            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss

    @torch.no_grad()
    def select_best(self, new_strings, batch_size=10):
        """
        Select generated strings which induce longest output sentences.
        """
        pred_len = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1

        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            if self.task == 'seq2seq':
                batch_strings = [x[1] for x in new_strings[st:ed]]
            else:
                batch_strings = [x[1] + self.eos_token for x in new_strings[st:ed]]
            inputs = self.tokenizer(
                batch_strings, 
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding=True,
            )
            input_ids = inputs.input_ids.to(self.device)
            outputs = dialogue(
                self.model, 
                input_ids,
                early_stopping=False, 
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, 
                use_cache=True,
                max_length=self.max_len,
            )
            lengths = [self.compute_seq_len(seq) for seq in outputs['sequences']]
            pred_len.extend(lengths)
            
        pred_len = np.array(pred_len)
        assert len(new_strings) == len(pred_len)
        return new_strings[pred_len.argmax()], max(pred_len)

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0] # original sentence length
        best_adv_text, best_len = deepcopy(text), ori_len
        cur_adv_text, cur_len = deepcopy(text), ori_len  # current_adv_text: List[str]
        return ori_len, (best_adv_text, best_len), (cur_adv_text, cur_len)

    def compute_loss(self, text):
        raise NotImplementedError

    def mutation(self, cur_adv_text, grad, modified_pos):
        raise NotImplementedError

    def run_attack(self, text, label):
        """
        (1) Using gradient ascent to generate adversarial sentences -- mutation();
        (2) Select the best samples which induce longest output sentences -- select_best();
        (3) Save the adversarial samples -- adv_his.
        """
        assert len(text) != 1
        torch.autograd.set_detect_anomaly(True)
        ori_len, (best_adv_text, best_len), (cur_adv_text, cur_len) = self.prepare_attack(text)
        ori_context = cur_adv_text.split(self.eos_token)[0].strip()
        adv_his = []
        modify_pos = [] # record already modified positions (avoid recovering to the original token)
        pbar = tqdm(range(self.max_per))
        t1 = time.time()

        for it in pbar:
            loss_list = self.compute_loss([cur_adv_text])
            if loss_list is not None:
                loss = sum(loss_list)
                self.model.zero_grad()
                loss.backward()
                grad = self.embedding.grad
            else:
                grad = None

            # Only mutate the part after eos_token
            cur_free_text = cur_adv_text.split(self.eos_token)[1].strip()
            new_strings = self.mutation(ori_context, cur_free_text, grad, label, modify_pos)
            # Pad the original context
            new_strings = [
                (pos, ori_context + " " + self.eos_token + " " + adv_text)
                for (pos, adv_text) in new_strings
            ]
            if new_strings:
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