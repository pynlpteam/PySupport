import re
# import inspect
import itertools
import dill as pickle
import string
import collections
import numpy as np
import pandas as pd

# NLP tools
import enchant
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
# import pymorphy2
from alphabet_detector import AlphabetDetector
from Dictionary import stopword_lists

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Initialising dictionaries
# en_dict = enchant.DictWithPWL("en_US", proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
# ru_aot_dict = enchant.Dict("ru_RU")
# nltk.download()


class Preprocessing(object):
    def __init__(self):
        # Pre-loading objects
        self.re_signatures = [re.compile(each) for each in stopword_lists.signatures]
        self.mystem = Mystem()
        # self.morph = pymorphy2.MorphAnalyzer()

        self.mystem_lemma_dict = None
        self.ad = AlphabetDetector()

        # self.proj_path = '/'.join(inspect.getfile(inspect.currentframe()).split('/')[:-2])
        print('Preprocessing  loaded')

        # Dicts
        # self.en_dict = enchant.DictWithPWL("en_US", self.proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
        # self.ru_aot_dict = enchant.Dict("ru_RU")
        self.stop_words = set(stopword_lists.yandex_seo_stopwords +
                              stopword_lists.custom_stop_words +
                              stopwords.words('russian'))
        self.padding_punctuation = """!"#$%&\'()*+,;<=>?[\\]^`{|}~/��"""
        self.full_punctuation = string.punctuation + '��'

    # ======================================== #
    # ######## STRING PREPROCESSING ########## #
    # ======================================== #
    @staticmethod
    def normalize(input_string):
        return input_string.lower().strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    @staticmethod
    def cut_by_signature(input_string, signature_string):
        """ Cut by input search pattern (string) """
        p = re.compile(signature_string)
        search = p.search(input_string)
        try:
            start_index = search.span()[0]  # start index
            if start_index > 4:  # Do not cut from the beginning
                return input_string[:start_index]
            else:
                return input_string
        except AttributeError:
            return input_string

    def cut_by_signatures(self, input_string):
        """
        Find index of earliest signature
        with precompiled regex and cut it
        """
        beginnings = []
        for each in self.re_signatures:
            try:
                # Index of 1st found position
                beginnings.append(each.search(input_string).span()[0])
            except AttributeError:
                pass

        if beginnings:
            cut = min(beginnings)
            # Not in the beginning
            if cut > 5:
                return input_string[:cut]
            else:
                return input_string
        else:
            return input_string

    def pad_punctuation(self, input_string, punct_list=None):
        """ Used to control tokenization """
        normal_text = input_string.strip()
        padding_punctuation = punct_list if punct_list else self.padding_punctuation
        for char in padding_punctuation:
            normal_text = normal_text.replace(char, ' ' + char + ' ')
        return normal_text

    @staticmethod
    def tokenize(input_string):
        return nltk.word_tokenize(input_string)

    def get_vocab(self, series):
        return set(self.series_to_chain(series))

    def get_all_token_chain(self, series):
        return self.series_to_chain(series)

    def is_punct(self, token):
        """ True only if all chars are punct """
        for c in token:
            if c not in self.full_punctuation:
                return False
        return True

    def remove_punct(self, tokenlist):
        return [token for token in tokenlist if not self.is_punct(token)]

    @staticmethod
    def contains_digits(input_string):
        return any(char.isdigit() for char in input_string)

    def contains_punct(self, input_string):
        return any(self.is_punct(char) for char in input_string)

    def is_cyrillic(self, token):
        """
        Checks if string has only cyrillic letters
        """
        # return not(any(ord(c) < 128 for c in token))
        if self.contains_digits(token) or self.contains_punct(token):
            return False
        else:
            return self.ad.only_alphabet_chars(token, 'CYRILLIC')

    def remove_stopwords(self, tokenized_text, stopword_list=None):
        if not stopword_list:
            stopword_list = self.stop_words
        return [t for t in tokenized_text if t not in stopword_list]

    @staticmethod
    def remove_by_token_length(tokenized_text, min_len=1, max_len=25):
        return [t for t in tokenized_text if len(t) >= min_len and len(t) < max_len]

    # ======================================== #
    # ########### POS/LEMMATIZING ############ #
    # ======================================== #

    '''
    def get_pymorphy_lemma(self, token):
        return self.morph.parse(token)[0].normal_form
    '''

    def get_mystem_lemma(self, token):
        # Returns [POS-tag, lemma] for token
        response = self.mystem.analyze(token)
        analysis = response[0].get('analysis')
        try:
            the_one = analysis[0]
            lex = the_one.get('lex')
            return lex
        except:
            return token

    def get_mystem_pos_tags(self, token):
        response = self.mystem.analyze(token)
        analysis = response[0].get('analysis')
        try:
            the_one = analysis[0]
            tag = the_one.get('gr')
            return tag
        except:
            return None

    def lemmatize_series(self, series):
        if not self.mystem_lemma_dict:
            print('Building lemma-dictionary')
            vocab = self.get_vocab(series)
            self.mystem_lemma_dict = {token: self.get_mystem_lemma(token) for token in vocab}
        return series.apply(lambda tokenlist: [self.mystem_lemma_dict[token] for token in tokenlist])

    def get_nltk_pos_df(self, series):
        all_tokens = self.get_all_token_chain(series)
        nltk_tags_tuple = nltk.pos_tag(all_tokens, lang='rus')
        tags = set([each[1] for each in nltk_tags_tuple])

        def get_tokens_by_tag(tag):
            # Set of tokens by input tag
            token_tag_list = list(filter(lambda x: x[1] == tag, nltk_tags_tuple))  # [token, tag]
            return [each[0] for each in token_tag_list]  # [token]

        tag_dict = collections.OrderedDict(zip(tags, [get_tokens_by_tag(tag) for tag in tags]))
        return pd.DataFrame.from_dict(tag_dict, orient='index').transpose()

    def get_mystem_pos_df(self, series):
        all_tokens = self.get_all_token_chain(series)
        mystem_tags_dict = {token: self.get_mystem_pos_tags(token) for token in set(all_tokens)}
        # filter_dict(mystem_tags_dict)
        mystem_tags_dict = dict(filter(lambda item: item[1] is not None, mystem_tags_dict.items()))

        def get_tokens_by_mystem_tag(input_tag):
            matched_tokens = [(token, all_tokens.count(token)) for token, tags in mystem_tags_dict.items() if
                              input_tag in tags]
            return sorted(matched_tokens, key=lambda x: x[1], reverse=True)

        # {tag: (token, count), ...}
        mystem_tag_dict = collections.OrderedDict(zip(stopword_lists.forbidden_mystem_tags,
                                                      [get_tokens_by_mystem_tag(tag) for tag in
                                                       stopword_lists.forbidden_mystem_tags]))
        return pd.DataFrame.from_dict(mystem_tag_dict, orient='index').transpose()

    # ======================================== #
    # ########## Jupyter analysis ############ #
    # ======================================== #
    @staticmethod
    def stats_for_untokenized(series):
        """ Counts symbols in series of texts """
        return sum([len(each) for each in series])

    @staticmethod
    def series_to_chain(series):
        """Chained tokens in Series"""
        return list(itertools.chain.from_iterable(list(series.values)))

    def stats_for_series(self, series):
        """DF from Series stats"""
        empty_texts_indexes = list(series[series.astype(str) == '[]'].index)
        empty_texts = len(empty_texts_indexes)
        token_chain = self.series_to_chain(series)

        data = pd.DataFrame(data=[[len(token_chain),
                                   len(list(set(token_chain))),
                                   len(series),
                                   empty_texts,
                                   token_chain.count('')]],
                            index=['Count'],
                            columns=['Total tokens',
                                     'Unique tokens',
                                     'Total texts',
                                     'Empty texts',
                                     'Empty tokens'])
        return data

    @staticmethod
    def check_empty_texts(series, original_df=None):
        """Get unprocessed text for '[]' in Series"""
        empty_texts_indexes = list(series[series.astype(str) == '[]'].index)
        if original_df is not None:
            return original_df.loc[empty_texts_indexes]
        else:
            return empty_texts_indexes

    @staticmethod
    def drop_empty_text_rows(series):
        drop_indexes = series[series.astype(str) == '[]'].index
        return series.drop(drop_indexes)

    @staticmethod
    def plot_occurrences(series, str_expression):
        """
        Detects first occurrence of str expression in text.
        Plots index distribution of occurrences.
        """
        indexes = [text.index(str_expression) for text in series if str_expression in text]
        fig, ax = plt.subplots()
        ax.hist(indexes, range(0, 50))
        ax.set_xticks(np.arange(0, 51, 1))
        ax.set_xlabel('Position')
        ax.set_ylabel('Count')
        plt.title("Occurrence distribution")
        print(len(indexes), ' occurrences found')
        return ax

    def get_token_frequencies_df(self, series, topn=50):
        ctr = collections.Counter(self.series_to_chain(series))
        fdist_list = ctr.most_common(topn)
        tokens = [k for k, v in fdist_list]
        counts = [v for k, v in fdist_list]
        return pd.DataFrame({"token": tokens, "count": counts})

    def plot_token_frequencies(self, series, top_n=30):
        """ Plot frequency distribution over corpus for top_n tokens tokens """
        ctr = collections.Counter(list(self.series_to_chain(series)))
        fdist_list = ctr.most_common(top_n)
        tokens = [k for k, v in fdist_list]
        counts = [v for k, v in fdist_list]
        token_count = pd.DataFrame({"token": tokens, "count": counts})
        sns.barplot(x="count", y="token", data=token_count).set_xlabel('Token appearence')

    def plot_token_distribution(self, series):
        """ Overall tokens lenghts distribution for series """
        token_lenghts = [len(x) for x in self.series_to_chain(series)]
        bow_lenghts = [len(x) for x in series]

        # Unique lens
        fig, ax = plt.subplots(ncols=2)

        ax[0].hist(token_lenghts, bins=range(0, 25))
        ax[0].set_xticks(np.arange(0, 26, 1))
        ax[0].set_xlabel('Token length')
        ax[0].set_ylabel('Count')

        ax[1].hist(bow_lenghts, bins=range(0, 25))
        ax[1].set_xticks(np.arange(0, 26, 1))
        ax[1].set_xlabel('Tokens in docs')
        ax[1].set_ylabel('Count')

        return ax

    @staticmethod
    def most_common_in_df(df):
        result = dict()
        for col in df.columns:
            try:
                col_most_freq = df[col].value_counts().reset_index()
                tokens = col_most_freq['index']
                freqs = col_most_freq[col]
                result[col] = [(t, f) for t, f in zip(tokens, freqs)]
            except:
                result[col] = [None]
        return pd.DataFrame.from_dict(result, orient='index').transpose()

    # ======================================== #
    # ###### TOKEN SEQUENCE PROCESSING ####### #
    # ======================================== #
    @staticmethod
    def get_texts_with_token(series, token):
        return [text for text in series if token in text]

    @staticmethod
    def cut_after_token(tokenlist, token, pos=0):
        """ Truncate token list after input token position """
        if token in tokenlist:
            if tokenlist.index(token) > 1:
                return tokenlist[:tokenlist.index(token) + pos]
            else:
                return tokenlist
        else:
            return tokenlist

    @staticmethod
    def get_indexes_of_token(series, token):
        """ Indexes of the token in all documents """
        indexes = [text.index(token) for text in series if token in text]
        return indexes

    @staticmethod
    def token_scope(series, token, pos):
        """ Set of tokens going before or after (by position) the given token """
        found = series.apply(lambda x: x[x.index(token) + pos] if token in x else 0)
        token_set = list(set(found[found != 0]))
        return token_set

    @staticmethod
    def seq_in_series(series, seq):
        """ Return text if sequence is in token list """
        result = []
        for text in series:
            if seq[0] in text:
                index = text.index(seq[0])
                if seq == text[index:(index + len(seq))]:
                    result.append(text)
        return result

    def plot_indexes_of_token(self, series, token, x_range):
        indexes = self.get_indexes_of_token(series, token)
        fig, ax = plt.subplots()
        ax.hist(indexes, bins=range(0, x_range))
        ax.set_xticks(np.arange(0, x_range + 1, 1))
        ax.set_yticks(np.arange(0, 21, 1))
        ax.set_xlabel('Index')
        ax.set_ylabel('Count')
        plt.title(token)
        return ax

    @staticmethod
    def cut_after_seq(tokenlist, seq):
        """ Truncate document after token sequence """
        if seq[0] in tokenlist:  # if first element of seq is in text
            index = tokenlist.index(seq[0])
            if seq == tokenlist[index:(index + len(seq))]:  # if whole sequence is is
                return tokenlist[:tokenlist.index(seq[0])]
            else:
                return tokenlist
        else:
            return tokenlist

    @staticmethod
    def cut_seq(tokenlist, seq):
        """ Removes sequence from tokenized texts. """
        if seq[0] in tokenlist:
            index = tokenlist.index(seq[0])
            if seq == tokenlist[index:(index + len(seq))]:
                '''
                for each in seq:
                    del tokenlist[tokenlist.index(each)]
                return tokenlist
                '''
                return tokenlist[:index] + tokenlist[index + len(seq):]  # TODO: test it
            else:
                return tokenlist
        else:
            return tokenlist

    # ======================================== #
    # ################ OTHER ################# #
    # ======================================== #
    def separate_by_category(self, series):
        """
        Separates tokens by types of chars in it (punctuation, numbers, ...)
        :param series: series of tokenized texts
        :return: dict of {category:[tokenlist]}
        """
        vocab = self.series_to_chain(series)

        result = {'num_punct': [],
                  'alpha_num': [],
                  'alpha_punct': [],
                  'punct_tokens': [],
                  'numeric_tokens': [],
                  'alpha_tokens': [],
                  'alpha_num_punct': []}

        for token in vocab:
            # Add flag by symbol category
            punct = [1 for symbol in token if (symbol in self.full_punctuation)]
            numerics = [1 for symbol in token if (symbol.isnumeric())]
            alpha = [1 for symbol in token if (symbol.isalpha())]

            # If token contains all types
            if (punct and numerics) and alpha:
                result['alpha_num_punct'].append(token)

            # Double
            elif numerics and punct:
                result['num_punct'].append(token)

            elif numerics and alpha:
                result['alpha_num'].append(token)

            elif alpha and punct:
                result['alpha_punct'].append(token)

            # Simple
            elif punct:
                result['punct_tokens'].append(token)

            elif numerics:
                result['numeric_tokens'].append(token)

            elif alpha:
                result['alpha_tokens'].append(token)

        return result

    def get_categories_df(self, series):
        """
        Separates tokens by types of chars in it (punctuation, numbers, ...)
        in different categories and sort them by frequency
        """
        separated_categories_dict = self.separate_by_category(series)
        categories = pd.DataFrame.from_dict(separated_categories_dict, orient='index')
        return categories.transpose()

    # ======================================== #
    # ############## PIPELINES ############### #
    # ======================================== #
    def apply_pipeline(self, raw_string):
        """ Apply all the methods to raw string """
        normalized = self.normalize(raw_string)
        # print('normalized: ', normalized)
        signatures_cut = self.cut_by_signatures(normalized)
        # print('signatures_cut: ', signatures_cut)
        padded = self.pad_punctuation(signatures_cut)
        # print('padded: ', padded)
        tokenized = self.tokenize(padded)
        # print('tokenized: ', tokenized)
        no_punct = self.remove_punct(tokenized)
        # print('no_punct: ', no_punct)
        no_stops = self.remove_stopwords(no_punct)
        cut_by_len = [t for t in no_stops if len(t) < 25]
        lemmatized = [self.get_mystem_lemma(token) for token in cut_by_len]
        # print('lemmatized: ', lemmatized)
        return lemmatized

    def apply_short_pipeline(self, raw_string):
        """ Preprocessing for manual input in window form on client-side """
        normalized = self.normalize(raw_string)
        tokenized = self.tokenize(normalized)
        cut_by_len = [t for t in tokenized if len(t) < 25]
        lemmatized = [self.get_mystem_lemma(token) for token in cut_by_len]
        return lemmatized

    @staticmethod
    def pickle_save(data, path):
        with open(path, 'wb') as fp:
            print(type(data))
            pickle.dump(data, fp)
            print('Saved as ', path)