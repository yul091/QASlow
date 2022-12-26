import abc
import torch
import stanza
import language_tool_python
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForMaskedLM


ENGLISH_FILTER_WORDS = [
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
    'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
    'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
    'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
    "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
    'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
    'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
    'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
    "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
    'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
    'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
    'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
    'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
    'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
    'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
    'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
    'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
    'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
    'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
    "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves', 'have', 'be'
]



class Substitute(metaclass=abc.ABCMeta):
    def __init__(self, victim_model):
        self.victim_model = victim_model

    @abc.abstractmethod
    def substitute(self,    **kwargs):
        raise Exception("Abstract method 'substitute' method not be implemented!")


class SubstituteWithBert(Substitute):
    def __init__(self, victim_model, device='cpu'):
        super().__init__(victim_model)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.predictor = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.predictor.to(device)

    @staticmethod
    # Get antonyms of a word using WordNet
    def get_word_antonyms(word):
        antonyms_lists = set()
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms_lists.add(l.antonyms()[0].name())
        return list(antonyms_lists)

    def substitute(self, hypothesis, origin_sentence, masked_sentence, label, attack_type):
        info_dict = dict()
        info_dict['done'] = False
        info_dict['adv'] = None
        info_dict['suc_advs'] = None
        info_dict['advs'] = None
        info_dict['prob'] = 0
        info_dict['query'] = 0

        inputs = self.tokenizer(masked_sentence, return_tensors='pt')
        tokenized_sentence = inputs['input_ids']

        for i in range(tokenized_sentence.size()[1]):
            if tokenized_sentence[0][i] == 103:
                index = i

        # restrict max input length to 512
        if inputs['input_ids'].size()[1] > 512:
            inputs['input_ids'] = inputs['input_ids'][:, 0:512]
            inputs['token_type_ids'] = inputs['token_type_ids'][:, 0:512]
            inputs['attention_mask'] = inputs['attention_mask'][:, 0:512]

        with torch.no_grad():
            outputs = self.predictor(input_ids=inputs['input_ids'].to(self.predictor.device),
                                     token_type_ids=inputs['token_type_ids'].to(self.predictor.device),
                                     attention_mask=inputs['attention_mask'].to(self.predictor.device))
        logits = torch.softmax(outputs.logits[0][index], -1)

        # Filter out antonyms predicted by BERT
        mask_index = masked_sentence.split(' ').index('[MASK]')
        try:
            masked_word = origin_sentence.split(' ')[mask_index]
        except Exception as e:
            print(masked_sentence)
            print(origin_sentence)
            print(len(masked_sentence), len(origin_sentence))
            return info_dict

        antonyms_list = self.get_word_antonyms(masked_word)

        probs, indices = torch.topk(logits, 10)
        indices = indices.to('cpu').numpy().tolist()
        pred_list = self.tokenizer.convert_ids_to_tokens(indices)

        remove_list = []
        for i, word in enumerate(pred_list):
            if word in antonyms_list:
                remove_list.append(indices[i])

        for i in remove_list:
            indices.remove(i)

        info_dict['query'] += len(indices)

        # Substitute the original sentence with words predicted by BERT
        modified_sentences = []
        for i, location in enumerate(indices):
            tokenized_sentence[0][index] = location
            modified_sentence_ids = tokenized_sentence[0][1:-1]

            modified_sentences_tokens = self.tokenizer.convert_ids_to_tokens(modified_sentence_ids)
            modified_sentence = self.tokenizer.convert_tokens_to_string(modified_sentences_tokens)
            modified_sentences.append(modified_sentence)

        with torch.no_grad():
            if hypothesis:
                inputs = [[premise, hypothesis] for premise in modified_sentences]

            else:
                inputs = modified_sentences

            outputs = self.victim_model(sentences=inputs)

        suc_advs = []
        for i, pred_label in enumerate(outputs.pred_labels):
            if pred_label.item() != label:
                suc_advs.append(modified_sentences[i])

        if len(suc_advs) > 0:
            info_dict['done'] = True
            info_dict['suc_advs'] = suc_advs

        else:
            if attack_type == 'score':
                index = torch.argmin(outputs.probs[:, label], 0)
                prob = outputs.probs[index][label]
                info_dict['prob'] = prob
                info_dict['adv'] = modified_sentences[index]

            elif attack_type == 'decision':
                info_dict['advs'] = modified_sentences

        return info_dict


class SubstituteWithWordnet(Substitute):
    def __init__(self, victim_model):
        super().__init__(victim_model)
        self.pos_dict = {'NOUN': 'n', 'VERB': 'v', 'ADV': 'r', 'ADJ': 'a'}
        self.pos_processor = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma')

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

    def get_synonyms(self, word, pos):
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

    def substitute(self, hypothesis, origin_sentence, masked_sentence, label, attack_type):
        info_dict = dict()
        info_dict['done'] = False
        info_dict['adv'] = None
        info_dict['suc_advs'] = None
        info_dict['prob'] = 0
        info_dict['query'] = 0

        word_list = masked_sentence.split(' ')
        mask_index = word_list.index('[MASK]')

        pos_list, word_lemma = self.get_pos(origin_sentence, mask_index)
        masked_word_pos = pos_list[mask_index]

        synonyms = self.get_synonyms(word_lemma, masked_word_pos)
        if not synonyms:
            return info_dict

        modified_sentences = []
        for synonym in synonyms:
            word_list[mask_index] = synonym
            modified_sentence = ' '.join(word for word in word_list)
            modified_sentences.append(modified_sentence)

        info_dict['query'] += len(modified_sentences)
        with torch.no_grad():
            outputs = self.victim_model(sentences=modified_sentences)

        suc_advs = []
        for i, pred_label in enumerate(outputs.pred_labels):
            if pred_label.item() != label:
                suc_advs.append(modified_sentences[i])

        if len(suc_advs) > 0:
            info_dict['done'] = True
            info_dict['suc_advs'] = suc_advs

        else:
            if attack_type == 'score':
                index = torch.argmin(outputs.probs[:, label], 0)
                prob = outputs.probs[index][label]
                info_dict['prob'] = prob
                info_dict['adv'] = modified_sentences[index]

            elif attack_type == 'decision':
                info_dict['advs'] = modified_sentences

        return info_dict




class GrammarChecker:
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        '''
        :param sentence:  a string
        :return:
        '''
        matches = self.lang_tool.check(sentence)
        return len(matches)


