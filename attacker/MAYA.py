import pandas as pd
import os
import torch
import stanza
from supar import Parser
from nltk.tree import Tree
from utils import SentenceEncoder, GrammarChecker
from .base import SlowAttacker
from typing import Optional, Union
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute





class MAYAAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
        paraphrasers=None,
        fine_tune_path=None,
        save_paraphrase_label=None,
    ):
        super(MAYAAttacker, self).__init__(
            device, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.parser = ConstituencyParser()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sim = SentenceEncoder()
        self.grammar = GrammarChecker()
        self.fine_tune_path = fine_tune_path
        self.save_paraphrase_label = save_paraphrase_label

        # collect supervisory singals for pretrained-agent Bert to fine-tune
        if fine_tune_path:
            if os.path.exists(fine_tune_path):
                self.samples = pd.read_csv(fine_tune_path, sep='\t')
                self.sample_num = self.samples['index'].values[-1] + 1
            else:
                self.samples = pd.DataFrame(columns=['index', 'sentence', 'sentences', 'label'])
                self.sample_num = 0

        if save_paraphrase_label:
            if os.path.exists(save_paraphrase_label):
                self.label_info = pd.read_csv(save_paraphrase_label, sep='\t')
            else:
                self.label_info = pd.DataFrame(columns=['sentence', 'phrase', 'label', 'length'])

            self.nlp = stanza.Pipeline('en', processors='tokenize,pos')

        # self.paraphrase_count = pd.DataFrame(columns=['back', 'gpt2', 'T5'])
        # self.back_count = 0
        # self.gpt2_count = 0
        # self.T5_count = 0

        self.substitution = substitution
        self.paraphrasers = paraphrasers

    # 将每个单词分别mask
    def get_masked_sentence(self, sentence):
        pos_info = None
        if self.save_paraphrase_label:
            doc = self.nlp(sentence)
            pos_info = []
            for stc in doc.sentences:
                for word in stc.words:
                    pos_info.append(word.pos)

        words = sentence.split(' ')
        masked_sentences = []

        for i in range(len(words)):
            word = words[i]
            words[i] = '[MASK]'
            tgt = ' '.join(x for x in words)
            masked_sentences.append(tgt)
            words[i] = word

        return masked_sentences, pos_info

    # 将句子统一用BertTokenizer格式化
    def formalize(self, sentences):
        formalized_sentences = []
        for ori in sentences:
            if ori is None:
                formalized_sentences.append(ori)
            else:
                tokens = self.tokenizer.tokenize(ori)

                if len(tokens) > 510:
                    tokens = tokens[0:510]

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
                # if max_index == 0:
                #     self.back_count += 1
                # elif max_index == 1:
                #     self.gpt2_count += 1
                # else:
                #     self.T5_count += 1
                best_adv = self.sim.find_best_sim(sentence, advs)[0]
                best_advs.append(best_adv)

            new_info.append(info[i])

        return best_advs, new_info

    # 对一个句子的预处理，获得所有可能的变形形式
    def sentence_process(self, sentence):
        sentence = sentence.lower()
        if self.substitution is None:
            masked_sentences, word_info = [], None
        else:
            masked_sentences, word_info = self.get_masked_sentence(sentence)

        masked_sentences = self.formalize(masked_sentences)

        root, nodes = self.parser(sentence)
        if len(nodes) == 0:
            return []

        phrases = [node[1] for node in nodes if node[3]]
        indices = [node[2] for node in nodes if node[3]]
        info = [[node[1], node[3], node[4]] for node in nodes]

        paraphrases = []
        with torch.no_grad():
            if phrases:
                for paraphraser in self.paraphrasers:
                    one_batch = paraphraser.paraphrase(phrases)
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

        return list(set(masked_sentences+best))

class ConstituencyParser:
    def __init__(self):
        self.parser = Parser.load('crf-con-en')

    @staticmethod
    def __sentence_to_list(sentence):
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