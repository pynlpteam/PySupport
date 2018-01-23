import gc
import dill as pickle
import collections

gc.enable()

class DatasetInterface(object):
    """
    Dataset management methods:
        * Loading from disk
        * Extracting raw text
        * Extracting preprocessed text
    """


    def __init__(self, dataset_id):
        # self.dataset_id = dataset_id
        self.dataset = self.load_dataset_from_disk(dataset_id)
        print('Dataset {} loaded'.format(dataset_id))

    # ======================================== #
    # ############ LOADING DATA ############## #
    # ======================================== #
    def load_dataset_from_disk(self, path):
        """ Pickled DF """
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    # ======================================== #
    # ########### DATA EXTRACTION ############ #
    # ======================================== #
    """
    def get_raw_text_by_doctag(self, doctag):
        assert isinstance(doctag, str), 'Input doctag is not str'
        return self.dataset[doctag].description

    def get_preprocessed_text_by_doctag(self, doctag):
        assert isinstance(doctag, str), 'Input doctag is not str'
        return self.dataset[doctag].preprocessed

    def doctag_not_in_dataset(self, doctag):
        assert isinstance(doctag, str), 'Input doctag is not str'
        return doctag not in self.dataset

    def doctag_in_dataset(self, doctag):
        assert isinstance(doctag, str), 'Input doctag is not str'
        return doctag in self.dataset
    """

