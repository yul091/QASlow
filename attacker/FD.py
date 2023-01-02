from typing import Optional, List, Union
import numpy as np
import torch
from transformers import (
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from utils import ENGLISH_FILTER_WORDS
from .base import SlowAttacker
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute
from OpenAttack.exceptions import WordNotInDictionaryException


class FDAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(FDAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        self.unk_token = tokenizer.unk_token
        
        
    def compute_loss(self, text: list, labels: list):
        return 
    
    
    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        new_strings = []
        x_orig = x_orig.lower()
        sent = self.default_tokenizer.tokenize(x_orig, pos_tagging=False)
        curr_sent = context + self.sp_token + x_orig
        scores, seqs, pred_len = self.compute_score([curr_sent]) # list N of [T X V], [T], [1]
        
        iter_cnt = 0
        while True:
            idx = np.random.choice(len(sent)) # randomly choose a word
            iter_cnt += 1
            if iter_cnt > 5 * len(sent): # failed to find a substitute word
                return None
            if sent[idx] in self.filter_words:
                continue
            try: # find a substitute word
                reps = list(map(lambda x:x[0], self.substitute(sent[idx], None)))
            except WordNotInDictionaryException:
                continue
            reps = list(filter(lambda x: x in victim_embedding.word2id, reps))
            if len(reps) > 0:
                break
            
        prob, grad = victim.get_grad([sent], [goal.target])
        grad = grad[0]
        prob = prob[0]
        if grad.shape[0] != len(sent) or grad.shape[1] != victim_embedding.embedding.shape[1]:
            raise RuntimeError("Sent %d != Gradient %d" % (len(sent), grad.shape[0]))
        s1 = np.sign(grad[idx])
        
        mn = None
        mnwd = None
        
        for word in reps:
            s0 = np.sign(victim_embedding.transform(word, self.token_unk) - victim_embedding.transform(sent[idx], self.token_unk))
            v = np.abs(s0 - s1).sum()
            if goal.targeted:
                v = -v
            
            if (mn is None) or v < mn:
                mn = v
                mnwd = word

        if mnwd is None:
            return None
        sent[idx] = mnwd
        
        
        return new_strings
    