# -*- coding: utf-8 -*-
###############################################
# imports
###############################################
import os
from bioservices.uniprot import UniProt
import pandas as pd
from io import StringIO
import re
import math
from goatools.obo_parser import GODag as _GoDag
from goatools.godag.go_tasks import get_go2ancestors
import json
import socket
from data.filters import filter as dataset_filter
import itertools
from collections import defaultdict
import numpy as np
from util.constants import amino_acid_alphabet
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from datetime import date
from util.fasta import fasta_to_list
from tqdm import tqdm
import pickle
import requests

tqdm.pandas()
pd.options.mode.chained_assignment = None

PATH = os.path.dirname(os.path.realpath(__file__))

###############################################
# datasets configuration class
###############################################
class Config():
    '''
    Reads the configuration and creates new datasets if necessary.
    '''
    def __init__(self):
        self._config = None
        self.levels = ['raw','datasets']
        if not self.load(): self.create()
        if not os.path.exists(PATH+'/datasets'): os.mkdir(PATH+'/datasets')
        if not os.path.exists(PATH+'/godag'): os.mkdir(PATH+'/godag')

    def load(self):
        if os.path.exists(PATH+'/config.json'):
            with open(PATH+'/config.json','r') as file:
                self._config = json.load(file)
            return True
        else: return False

    def create(self):
        print('WARNING: No dataset configuration found. Creating an empty one. Please specify a configuration.')
        self._config = []
        with open(PATH+'/config.json','w') as file:
            json.dump(self._config, file)

    def exists(self, names):
        names = names.split(' ')
        for raw in self._config:
            if len(names) == 1:
                if raw['name'] == names[0]:
                    return True
            elif len(names) == 2:
                for dataset in raw['datasets']:
                    if dataset['name'] == names[1]:
                        return True
        return False

    def checkout(self):
        for raw in self._config:
            if not os.path.exists(PATH+'/datasets/'+raw['name']):
                RawData.create(name=raw['name'], query=raw['query'], godag=raw['godag'], verbose=True)
            for dataset in raw['datasets']:
                if not os.path.exists(PATH+'/datasets/'+raw['name']+'/'+dataset['name']):
                    Dataset.create(raw=raw['name'], name=dataset['name'], filter=dataset['filter'], verbose=True)

    def get_godag(self, names):
        names = names.split(' ')
        for raw in self._config:
            if raw['name'] == names[0]:
                return raw['godag']
        return False

CONFIG = Config()


###############################################
# GO DAG class
###############################################
class GoDag():
    '''
    GO DAG class to store the GO hierarchy.
    '''
    def __init__(self, name):
        if not os.path.exists(PATH+'/godag/'+name):
            raise Exception('GO DAG {} does not exist. Please download it and place it in the data/godag directory or use the name of a RawDataset (e.g. "base.obo").'.format(name))
        self.GODAG = _GoDag(PATH+'/godag/'+name, prt=None)

    def get(self, term):
        return self.GODAG[term]

    def get_go_lineage_of(self, terms):
        g = [self.GODAG[i] for i in terms]
        g = get_go2ancestors(g, False)
        gos = []
        for key in g:
            gos.append(key)
            gos.extend(g[key])
        return list(set(gos))


###############################################
# base dataset class
###############################################
class BaseDataset():
    '''
    Base Dataset class. Implements data loading and sequence/label embedding, along with a few helper functions.
    '''
    def __init__(self, names=False):
        if names:
            path, godag = self.load(names)
            self.names = names
            self.path = path
            self.df = pd.read_csv(path+'/data.csv', converters={'labels': lambda x: x.split('; ')})
            self.godag = GoDag(godag)
            self._godag = godag
            self.terms = pd.read_csv(path+'/terms.csv') if os.path.exists(path+'/terms.csv') else self.calculate_term_frequency()
            self.goterms = self.terms['term'].tolist()

    def copy(self):
        ds = BaseDataset().from_df(self.df.copy(), self._godag)
        return ds

    def from_df(self, df, godag='base.obo', path=None):
        assert ('id' in df.columns and 'sequence' in df.columns), 'Dataframe format not correct. Please provide data with an \'id\' and \'sequence\' column.'
        self.path = path
        self.df = df
        self.godag = GoDag(godag)
        self._godag = godag
        self.terms = self.calculate_term_frequency(save=False)
        self.goterms = self.terms['term'].tolist()
        return self

    def from_csv_file(self, file, godag='base.obo'):
        df = pd.read_csv(file, converters={'labels': lambda x: x.split(' ')})
        df['sequence'] = df['sequence'].map(lambda x: re.sub('-', '', x))
        if len(df.index) == 0:
            print('Error: Empty file given to Dataset.')
            return False
        return self.from_df(df, godag)

    def from_fasta(self, fasta, godag='base.obo', path=None):
        def fasta2list(f):
            l = []
            name, seq = False, ''
            for line in f.splitlines():
                line = line.rstrip()
                if line.startswith(">"):
                    if name: l.append([name,seq])
                    name = line[1:]
                    seq = ''
                else: seq += line
            l.append([name,seq])
            return l
        df = pd.DataFrame(fasta2list(fasta), columns=['labels','sequence'])
        if len(df.index) == 0:
            print('Error: Empty file given to Dataset.')
            return False
        df['id'] = df['labels'].map(lambda x: x.split(' ')[0]).astype(str)
        df['labels'] = df['labels'].map(lambda x: x.split(' ')[1:])
        df['sequence'] = df['sequence'].map(lambda x: re.sub('-', '', x))
        self.path = path
        self.df = df
        self.godag = GoDag(godag)
        self.terms = self.calculate_term_frequency(save=False)
        return self

    def from_fasta_file(self, file, godag='base.obo', path=None):
        with open(file,'r') as file:
            fasta = file.read()
        return self.from_fasta(fasta, godag, path)

    def load(self, names):
        assert CONFIG.exists(names), 'No such data found! Please make sure the name is spelled correctly and check the dataset configuration.'
        path = PATH+'/datasets/'+'/'.join(names.split(' '))
        godag = CONFIG.get_godag(names)
        return path, godag


    def calculate_term_frequency(self, save=True):
        '''
        Calculates the number of occurences for each term.
        '''
        # term list
        terms = self.df['labels'].tolist()
        terms = list(itertools.chain(*terms))
        # term counts
        sf = pd.Series(terms).value_counts()
        df = pd.DataFrame({'term':sf.index, 'count':sf.values})
        df = df[df.term != 'GO:0003674'] # filter MF
        # level
        df['level'] = df.apply(lambda row: self.godag.get(row['term']).level, axis=1)
        # name
        df['name'] = df.apply(lambda row: self.godag.get(row['term']).name, axis=1)
        if save: df.to_csv(self.path+'/terms.csv', header=True, index=False)
        return df

    def fasta(self, labels_as_description=False):
        fasta = ''
        for i, row in self.df.iterrows():
            if labels_as_description:
                fasta += '>'+row['id']+' '+','.join(row['labels'])+'\n'+row['sequence']+'\n'
            else:
                fasta += '>'+row['id']+'\n'+row['sequence']+'\n'
        return fasta

    def ravel(self, *columns):
        df = self.df[list(columns)]
        cols = df.columns
        for c in cols:
            df = df.explode(c)
        return df

    def pad_to(self, length):
        self.df['raw_sequence'] = self.df['sequence']
        self.df['sequence'] = self.df.apply(lambda row: row['sequence'].ljust(length,'-')[:length], axis=1)
        return self

    def tokenize_sequences(self):
        import time
        tk = Tokenizer(num_words=None, char_level=True,lower=False)
        tk.word_index = {aa:i+1 for i,aa in enumerate(amino_acid_alphabet)}
        tk.word_index['-'] = 0
        tk.index_word = {i:aa for aa,i in tk.word_index.items()}
        self.df['sequence_tokenized'] = list(tk.texts_to_sequences(self.df['sequence'].tolist()))
        self.df['sequence_tokenized'] = self.df.apply(lambda row: np.array(row['sequence_tokenized'], dtype=np.uint8), axis=1)
        self.tokenizer = tk
        self.alphabet_size = len(self.tokenizer.word_index)
        return self

    def embed_labels(self, binarizer=None):
        if binarizer is None:
            mlb = MultiLabelBinarizer()
            self.df['labels_onehot'] = list(mlb.fit_transform(self.df['labels'].tolist()))
        else:
            mlb = binarizer
            self.df['labels_onehot'] = list(mlb.transform(self.df['labels'].tolist()))
        self.df['labels_onehot'] = self.df['labels_onehot'].map(lambda x: np.array(x, dtype=np.float32))
        self.labelBinarizer = mlb
        self.num_classes = len(self.labelBinarizer.classes_)
        return self

    def tf(self):
        def generator():
            for index, row in self.df.iterrows():
                yield (row['sequence_tokenized'], row['labels_onehot'])
        ds = tf.data.Dataset.from_generator(generator, output_types=(tf.uint8, tf.float32))
        return ds

    def raw(self):
        return (self.df['sequence_tokenized'].map(lambda x: np.asarray(x, dtype=np.uint8)).to_list(), self.df['labels_onehot'].map(lambda x: np.asarray(x, dtype=np.float32)).to_list())


###############################################
# raw data class
###############################################
class RawData(BaseDataset):
    '''
    Mainly used to download data from UniProt and the GO DAG from obolibrary.org.
    '''
    def __init__(self, names):
        super().__init__(names)

    @classmethod
    def create(cls, name, query, godag, verbose=True):
        if verbose: print('No data with name \'{}\' found. Downloading... (This may take a while)'.format(name))
        # download current GO DAG
        r = requests.get('http://purl.obolibrary.org/obo/go/go-basic.obo', allow_redirects=True)
        with open(PATH+'/godag/'+name+'.obo','wb') as file:
            file.write(r.content)
        # setup UniProt API
        up = UniProt()
        up.settings.TIMEOUT = None
        # load GO DAG
        godag = GoDag(name=godag)
        # download
        result = up.search(query, columns='id, sequence, go(molecular function)')
        df = pd.read_csv(StringIO(result), delimiter='\t')
        # filter go terms present in godag
        df['Gene ontology (molecular function)'] = df['Gene ontology (molecular function)'].map(lambda labels: [l for l in re.findall('GO:\d{7}',str(labels)) if l in godag.GODAG])
        df = df[df['Gene ontology (molecular function)'].map(lambda l: len(l) > 0)]
        # annotate full go
        df['Gene ontology (molecular function)'] = df['Gene ontology (molecular function)'].map(lambda labels: godag.get_go_lineage_of(labels))
        # clean
        df = cls.clean(df)
        # save
        df['labels'] = df.apply(lambda row: '; '.join(row['labels']), axis=1)
        os.mkdir(PATH+'/datasets/'+name)
        df.to_csv(PATH+'/datasets/'+name+'/data.csv', index=False)
        with open(PATH+'/datasets/'+name+'/info.txt', 'w') as file:
            file.write('Downloaded: {}'.format(date.today()))
        if verbose: print('Raw data successfully downloaded.')

    @classmethod
    def clean(cls, df):
        df.rename(columns={
            'Entry':'id',
            'Sequence':'sequence',
            'Gene ontology (molecular function)': 'labels'
            },
            inplace=True)
        return df

###############################################
# preprocessed data class
###############################################
class Dataset(BaseDataset):
    '''
    Creates a Dataset. Takes care of filtering according to data/filters.py
    '''
    def __init__(self, names):
        super().__init__(names)

    @classmethod
    def create(cls, raw, name, filter, verbose=False):
        if verbose: print('No dataset with name \'{}\' found. Preprocessing...'.format(name))
        # load raw data
        raw_data = RawData(raw)
        # filter
        for f in filter:
            filter_fx = f['filter_fx']
            filter_args = f['filter_args']
            assert filter_fx in dataset_filter.keys(), 'Filter \'{}\' not found. Please specify a valid filter.'.format(filter)
            df_filter = dataset_filter[filter_fx](*filter_args)
            raw_data.df = df_filter(raw_data)
        df = raw_data.df
        df = df[df['labels'].map(lambda l: len(l) > 0)]
        # save
        df['labels'] = df['labels'].map(lambda labels: '; '.join(labels))
        os.mkdir(PATH+'/datasets/'+raw+'/'+name)
        df.to_csv(PATH+'/datasets/'+raw+'/'+name+'/data.csv', index=False)
        # generate split candidates
        ds = cls(raw+' '+name)
        if verbose: print('Dataset successfully preprocessed.')


###############################################
# load and check configuration
###############################################
CONFIG.checkout()
