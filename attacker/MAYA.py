import pandas as pd
import abc
import copy
import torch
# from supar import Parser
from nltk.tree import Tree
from utils import SentenceEncoder, GrammarChecker
from .base import SlowAttacker
from typing import Optional, Union, List
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    PegasusTokenizer
)





class MAYAAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: BertTokenizer = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
        # fine_tune_path=None, #不知道是什么
        # save_paraphrase_label=None, #不知道是什么
    ):
        super(MAYAAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.parser = ConstituencyParser()
        self.sim = SentenceEncoder()
        self.grammar = GrammarChecker()
        self.paraphraser = T5()
        # self.fine_tune_path = fine_tune_path
        # self.save_paraphrase_label = save_paraphrase_label

        # collect supervisory singals for pretrained-agent Bert to fine-tune
        # 这是什么鬼？
        # if fine_tune_path:
        #     if os.path.exists(fine_tune_path):
        #         self.samples = pd.read_csv(fine_tune_path, sep='\t')
        #         self.sample_num = self.samples['index'].values[-1] + 1
        #     else:
        #         self.samples = pd.DataFrame(columns=['index', 'sentence', 'sentences', 'label'])
        #         self.sample_num = 0

        # if save_paraphrase_label:
        #     if os.path.exists(save_paraphrase_label):
        #         self.label_info = pd.read_csv(save_paraphrase_label, sep='\t')
        #     else:
        #         self.label_info = pd.DataFrame(columns=['sentence', 'phrase', 'label', 'length'])

        #     self.nlp = stanza.Pipeline('en', processors='tokenize,pos')

    def compute_loss(self, text: list, labels: list):
        return None, None

    # 将每个单词分别mask
    def get_masked_sentence(self, sentence:str):
        pos_info = None
        # if self.save_paraphrase_label:
        #     doc = self.nlp(sentence)
        #     pos_info = []
        #     for stc in doc.sentences:
        #         for word in stc.words:
        #             pos_info.append(word.pos)

        words = sentence.split(' ')
        masked_sentences = []
        indices = []

        for i in range(len(words)):
            word = words[i]
            words[i] = '[MASK]'
            tgt = ' '.join(x for x in words)
            masked_sentences.append(tgt)
            words[i] = word
            indices.append(i)

        return masked_sentences, pos_info, indices

    # 将句子统一用BertTokenizer格式化
    def formalize(self, sentences):
        formalized_sentences = []
        for ori in sentences:
            if ori is None:
                formalized_sentences.append(ori)
            else:
                tokens = self.tokenizer.tokenize(ori)

                if len(tokens) > 64:
                    tokens = tokens[0:64]

                string = self.tokenizer.convert_tokens_to_string(tokens)
                formalized_sentences.append(string)

        return formalized_sentences

    def get_best_sentences(self, sentence, paraphrases, info):
        ori_error = self.grammar.check(sentence)
        best_advs = []
        new_info = []
        # 获取同一个短语的不同paraphrase并取USE最高的
        for i in range(len(paraphrases[0])):
            advs = []

            # check the grammar error and filter out those which don't fit the restrictions
            for types in paraphrases:
                if types[i] is None:
                    continue

                adv_error = self.grammar.check(types[i])
                if adv_error <= ori_error:
                    advs.append(types[i])

            if len(advs) == 0:
                continue

            elif len(advs) == 1:
                best_advs.append(advs[0])

            else:
                best_adv = self.sim.find_best_sim(sentence, advs)[0]
                best_advs.append(best_adv)

            new_info.append(info[i])

        return best_advs, new_info

    # 对一个句子的预处理，获得所有可能的变形形式
    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        sentence = sentence.lower()
        masked_sentences, word_info, masked_indices = self.get_masked_sentence(sentence)

        masked_sentences = self.formalize(masked_sentences)
        masked_new_strings = []
        for i in range(len(masked_indices)):
            masked_new_strings.append(masked_indices[i])
            masked_new_strings.append(masked_sentences[i])

        root, nodes = self.parser(sentence)
        if len(nodes) == 0:
            return []

        phrases = [node[1] for node in nodes if node[3]]
        indices = [node[2] for node in nodes if node[3]]
        print(phrases)
        print(indices)
        info = [[node[1], node[3], node[4]] for node in nodes]

        paraphrases = []
        with torch.no_grad():
            if phrases:
                one_batch = self.paraphraser.paraphrase(phrases)
                if one_batch is not None:
                    paraphrases.append(one_batch)

        translated_sentence_list = []
        if len(paraphrases) > 0:
            for paraphrase_list in paraphrases:
                translated_sentences = []

                for i, phrase in enumerate(paraphrase_list):
                    tree = self.parser.get_tree(phrase)
                    try:
                        root_copy = copy.deepcopy(root)
                        root_copy[indices[i]] = tree[0]
                        modified_sentence = ' '.join(word for word in root_copy.leaves()).lower()
                        translated_sentences.append(modified_sentence)

                    except Exception as e:
                        translated_sentences.append(None)

                translated_sentences = self.formalize(translated_sentences)
                translated_sentence_list.append(translated_sentences)

        best = []
        # filter out paraphrases which don't fit the grammar strict and choose the most similar one.
        if len(translated_sentence_list) > 0:
            try:
                best, info = self.get_best_sentences(sentence, translated_sentence_list, info)

            except Exception as e:
                for i in translated_sentence_list:
                    print(len(i))
                print('error in getting best paraphrases!')
                best = []

        return list(set(masked_new_strings+best))

class ConstituencyParser:
    def __init__(self):
        self.parser = Parser.load('crf-con-en')

    @staticmethod
    def __sentence_to_list(sentence:str):
        word_list = sentence.strip().replace('(', '[').replace(')', ']').split(' ')

        while '' in word_list:
            word_list.remove('')

        return word_list

    def get_tree(self, sentence):
        word_list = self.__sentence_to_list(sentence)
        if len(word_list) == 0:
            return None

        try:
            prediction = self.parser.predict(word_list, verbose=False)
            return prediction.trees[0]

        except Exception as e:
            print('error: cannot get tree!')
            return None

    def __call__(self, sentence):
        # 返回句子的每个节点及索引
        root = self.get_tree(sentence)
        if root is None:
            return None, []

        node_list = pd.DataFrame(columns=['sub_tree', 'phrase', 'index', 'label', 'length'])

        for index in root.treepositions():
            sub_tree = root[index]
            if isinstance(sub_tree, Tree):
                if len(sub_tree.leaves()) > 1:
                    phrase = ' '.join(word for word in sub_tree.leaves())
                    node_list = node_list.append({'sub_tree': sub_tree,
                                                  'phrase': phrase,
                                                  'index': index,
                                                  'label': sub_tree.label(),
                                                  'length': len(sub_tree.leaves())}, ignore_index=True)

        node_list = node_list.drop_duplicates('phrase', keep='last')

        return root, node_list.values

class Paraphraser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def paraphrase(self, sentences):
        raise Exception("Abstract method 'substitute' method not be implemented!")

class T5(Paraphraser):
    def __init__(self, device='cuda'):
        super().__init__()
        model_name = 'tuner007/pegasus_paraphrase'
        self.max_length = 512
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name,
                                                                     max_length=self.max_length,
                                                                     max_position_embeddings=self.max_length).to(self.device)

    def paraphrase(self, sentences):
        with torch.no_grad():
            tgt_text = []
            for sentence in sentences:
                batch = self.tokenizer.prepare_seq2seq_batch([sentence],
                                                             truncation=True,
                                                             padding='longest',
                                                             max_length=int(len(sentence.split(' '))*1.2),
                                                             return_tensors="pt").to(self.device)

                translated = self.model.generate(**batch,
                                                 max_length=self.max_length,
                                                 min_length=int(len(sentence.split(' '))*0.8),
                                                 num_beams=1,
                                                 num_return_sequences=1,
                                                 temperature=1.5)

                tgt_text += self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            return tgt_text