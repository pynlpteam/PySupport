# general
import os
import re
import inspect
import itertools
import dill as pickle
import string
from collections import Counter

# NLP tools
import enchant
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from pymystem3 import Mystem
import pymorphy2

# from Preprocessing.Dicts import word_lists, it_ru_dict

# Init project path
# inspect.getfile(inspect.currentframe()) # script filename (usually with path)
proj_path = '/'.join(inspect.getfile(inspect.currentframe()).split('/')[:-2])
# proj_path = '/'.join(os.path.dirname(os.getcwd()).split('/')[:-1])
print('project path (preprocessing): ', proj_path)

# Initialising Mystem
mystem = Mystem()

# Initialising dictionaties
en_dict = enchant.DictWithPWL("en_US", proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
ru_aot_dict = enchant.Dict("ru_RU")
# nltk.download()
# print(enchant.list_languages())

class Preprocessing():
    def __init__(self):
        self.re_signatures = [re.compile(each) for each in word_lists.signatures]
        self.mystem = Mystem()
        self.morph = pymorphy2.MorphAnalyzer()
        self.en_dict = enchant.DictWithPWL("en_US", proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
        self.ru_aot_dict = enchant.Dict("ru_RU")
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(word_lists.yandex_seo_stopwords)
        self.dataset =  self.load_dataset('/home/aviadmin/semantic_git/models/21/dataset')
        self.vocab = self.dataset._vocab

    # def __init__(self):

    def load_dataset(self, path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)

    def normalize(self, input_string):
        return input_string.lower().strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    def cut_by_signature(self, input_string):
        """ Find index of earliest signature index and cut it. """
        beginnings = []
        for each in self.re_signatures:
            try:
                beginnings.append(each.search(input_string).span()[0])
            except AttributeError:
                pass

        if beginnings:
            return input_string[:min(beginnings)]
        else:
            return input_string

    def tokenize(self, input_string):
        return nltk.word_tokenize(input_string)

    def remove_stopwords(self, tokenized_text):
        return [t for t in tokenized_text if t not in self.stop_words]

    def get_pymorphy_lemma(self, token):
        return self.morph.parse(token)[0].normal_form

    def scan_by_vocab(self, text):
        return [t for t in text if t in self.vocab]
