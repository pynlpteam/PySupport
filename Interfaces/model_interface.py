import gc
import random
import multiprocessing
import pandas as pd

import gensim
import preprocessing
from Interfaces.dataset_interface import *

gc.enable()
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# DEBUG = True

# Init project path
# proj_path = '/'.join(os.path.dirname(os.getcwd()).split('/')[:-1])
# print('Proj_path = ', proj_path)
cpu_count = multiprocessing.cpu_count()


class Doc2vecInterface(object):
    """
    Model management methods:
        * Preprocessing
        * Similarity queries
    """
    # models_dir = '/home/tonko22/semantic_git/semantic/models/'
    models_dir = '/home/aviadmin/semantic_git/models/'

    def __init__(self, dataset_path='', model_id=None, dataset_id=None, train_params=None, workers=cpu_count):
        self.workers = workers
        self.pp = preprocessing.Preprocessing()
        self.dataset_id = dataset_id
        self.model = None

        if model_id:
            self.model_id = str(model_id)
            print('Model {} loaded'.format(self.model_id))
            self.model = self.load_model_from_disk()
            self.model_vocab = set(self.model.wv.vocab.keys())
            self.dataset_interface = DatasetInterface(self.model_id)

        if self.dataset_id:
            self.dataset_interface = DatasetInterface(self.dataset_id)
            self.train_data = self.prepare_doc2vec_train_data(self.dataset_interface.dataset)
            # self.train_data = self.prepare_doc2vec_train_data_int(self.dataset_interface.dataset)
            self.init_and_train(**train_params)
        else:
            print('No params')

    # ======================================== #
    # ########### PREPARING DATA ############# #
    # ======================================== #
    def load_model_from_disk(self):
        model_path = self.models_dir + self.model_id + '/model.doc2vec'
        try:
            model = gensim.models.Doc2Vec.load(model_path)
            model.make_cum_table()
            return model
        except FileNotFoundError:
            print('Model file not found at {}'.format(model_path))
            return None

    @staticmethod
    def prepare_doc2vec_train_data(dataset):
        """ Make TaggedDocument from dataset dict """
        assert isinstance(dataset, dict)
        return [gensim.models.doc2vec.TaggedDocument(
            tags=[str(tag)], words=ntuple.preprocessed)
            for tag, ntuple in dataset.items()]

    @staticmethod
    def prepare_doc2vec_train_data_int(dataset):
        """ Make TaggedDocument from dataset dict """
        assert isinstance(dataset, dict)
        return [gensim.models.doc2vec.TaggedDocument(
            tags=[int(tag)], words=ntuple.preprocessed)
            for tag, ntuple in dataset.items()]

    # ======================================== #
    # ###### EXTRACTING DATA FROM MODEL ###### #
    # ======================================== #
    def get_random_doctag(self):
        return random.choice(self.model.docvecs.offset2doctag)

    def get_tokens_not_in_vocab(self, tokens):
        return list(filter(lambda x: x not in self.model.wv.vocab.keys(), tokens))

    def doctag_not_in_model(self, doctag):
        return doctag not in self.model.docvecs.doctags.keys()

    def get_model_params_df(self):
        model_params_dict = self.model.__dict__
        useful_params_dict = dict((key, value) for key, value in model_params_dict.items() if key in
                                  ('alpha', 'corpus_count', 'cbow_mean', 'dbow_words', 'dm_concat',
                                   'hs', 'iter', 'layer1_size', 'min_alpha', 'min_alpha_yet_reached',
                                   'min_count', 'negative', 'sample', 'seed', 'sg', 'train_count',
                                   'vector_size', 'window'))
        model_params = pd.DataFrame.from_dict(useful_params_dict, orient='index').reset_index()
        # model_params.columns = ['param', str(model.id)]
        return model_params

    # ======================================== #
    # ############# INIT/TRAIN ############### #
    # ======================================== #
    def init_and_train(self, dm, size, window, alpha, min_count, sample, iter, hs, negative, dm_mean, dm_concat,
                       dbow_words, seed=None):
        """ Model trains on the init stage. """
        print('\nTraining model with {} epochs, {} alpha'.format(iter, alpha))
        if not self.train_data:
            raise ValueError('No TRAIN DATA')
        random.shuffle(self.train_data)
        if not seed:
            seed = 99
        self.model = gensim.models.Doc2Vec(
            documents=self.train_data,
            workers=self.workers,
            dm=dm,
            size=size,
            window=window,
            alpha=alpha,
            seed=seed,
            min_count=min_count,
            sample=sample,
            iter=iter,
            hs=hs,
            negative=negative,
            dm_mean=dm_mean,
            dm_concat=dm_concat,
            dbow_words=dbow_words
        )
        print('Training complete')

    # ======================================== #
    # ########## SIMILARITY QUERIES ########## #
    # ======================================== #
    def get_similarity_by_doctag(self, doctag, topn=1):
        assert type(doctag) == str, 'Input doctag is not str\n'
        assert len(doctag) == 10, 'Doctag is not standart: len < 10\n'
        print('Getting similarity for doctag: ', doctag)
        return self.model.docvecs.most_similar(positive=[doctag], topn=topn)

    def get_similarity_by_tokens(self, preprocessed_tokens, topn=1):
        assert type(preprocessed_tokens) == list, 'input must be list of tokens'
        vec = self.model.infer_vector(preprocessed_tokens, steps=100)
        similarity = self.model.docvecs.most_similar(positive=[vec], topn=topn)
        return similarity

    def get_similarity_by_raw_text(self, raw_text, topn=1):
        assert type(raw_text) == str, 'input text must be str\n'
        preprocessed_tokens = self.pp.apply_pipeline(raw_text)
        assert type(preprocessed_tokens) == list, 'preprocessed_tokens must be list of tokens\n'
        vec = self.model.infer_vector(preprocessed_tokens, steps=100)
        similarity = self.model.docvecs.most_similar(positive=[vec], topn=topn)
        return similarity
