import gc
import dill as pickle
import collections

gc.enable()

# NamedTuple to load dataset dict; Use "from optimization import *" to load it
incident = collections.namedtuple('incident', ['description', 'preprocessed'])


class DatasetInterface(object):
    """
    Dataset management methods:
        * Loading from disk
        * Extracting raw text
        * Extracting preprocessed text
    """
    # datasets_dir = '/home/tonko22/semantic_git/semantic/models/'
    datasets_dir = '/home/aviadmin/semantic_git/models/'

    def __init__(self, dataset_id):
        # self.dataset_id = dataset_id
        self.dataset = self.load_dataset_from_disk(dataset_id)
        print('Dataset {} loaded'.format(dataset_id))

    # ======================================== #
    # ############ LOADING DATA ############## #
    # ======================================== #
    def load_dataset_from_disk(self, dataset_id):
        print("DATASET ID: {}".format(str(dataset_id)))
        dataset_path = self.datasets_dir + str(dataset_id) + '/dataset'
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

        '''
        def normalize(num):
            # Format doc ids for backward compatibility with old dataset format
            num = str(num)
            while len(num) < 10:
                num = '0' + num
            return num

        if int(dataset_id) >= 20:
            dataset_path = datasets_dir + str(dataset_id) + '/dataset'
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
                return dataset

        # Old-style dataset
        else:
            dataset_path = datasets_dir + str(dataset_id) + '/raw_descriptions'
            with open(dataset_path, 'rb') as f:
                raw_descriptions = pickle.load(f)
                dataset = {normalize(each[0]): each[1] for each in raw_descriptions}
                return dataset
        '''

    # ======================================== #
    # ########### DATA EXTRACTION ############ #
    # ======================================== #
    def get_raw_text_by_doctag(self, doctag):
        assert isinstance(doctag, str), 'Input doctag is not str'
        # Old style formatting
        # while len(doctag) < 10:
        #     doctag = '0'+doctag
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


'''
class Dataset(dict):
    """
    Constructed from dict {'doctag': incident_NamedTuple}.
    incident is a namedtuple with fields ['description', 'preprocessed', 'solution']
    id_num is dataset unique id and folder name on disk at the same time.
    """
    incident = collections.namedtuple('incident', ['description', 'preprocessed', 'solution'])

    class DataError(Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)

    def __init__(self, data=None, description='', id_num=None):

        # Creating new dataset
        if isinstance(data, dict):
            dict.__init__(self, data)
            self.creation_date = str(datetime.datetime.now())
            self.description = description
            if id_num:
                self.id_num = str(id_num)

        elif isinstance(data, pd.core.frame.DataFrame):
            dict.__init__(self, data)
            self.creation_date = str(datetime.datetime.now())
            self.description = description
            if id_num:
                self.id_num = str(id_num)
        else:
            if id_num:
                self.id_num = str(id_num)
                self.path = '/home/aviadmin/semantic_git/models/' + self.id_num + '/dataset'

        # Set of tokens
        self._vocab = None  
        self._preprocessed = None

    # TODO: Custom iterators
    """
    def __iter__(self):
        for line in open('datasets/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

    @property # Lazy-init vocab
    def preprocessed(self):
        print('Lazy init vocab')
        if not self._vocab:
            self._vocab = set(chain.from_iterable([each.preprocessed for each in self.values()]))
        return self._vocab
    """

    @property # Lazy-init vocab
    def vocab(self):
        print('Lazy init vocab')
        if not self._vocab:
            self._vocab = set(chain.from_iterable([each.preprocessed for each in self.values()]))
        return self._vocab

    def save(self):
        with open(self.path, 'wb') as fp:
            pickle.dump(self, fp)
            print('Saved to ', self.path)

'''
