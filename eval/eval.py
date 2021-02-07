# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import pickle
from data.dataset import Dataset, BaseDataset
import json
import random
from util.tools import NumpyEncoder
from util.constants import amino_acid_alphabet
from joblib import Parallel, delayed
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PATH = os.path.dirname(os.path.abspath(__file__))

class FeatureSpace():
    '''
    This class implements the routines to do sequence mappings
    with the spectrum kernel and MMD estimates of two sequence sets.
    '''
    def __init__(self, spectrum=3):
        self.spectrum = spectrum
        self.mode = 'indicate'
        self.kmers = [''.join(i) for i in itertools.product(amino_acid_alphabet, repeat = spectrum)]
        self.trie = self.parse_dictionary(self.kmers)

    def cache(self, dataset):
        if not os.path.exists(PATH+'/cache'):
            os.mkdir(PATH+'/cache')
        name = '_'.join(list(dataset))
        path = PATH+'/cache/{}.npy'.format(name)
        if os.path.exists(path):
            return np.load(path)
        else:
            data = Dataset(dataset)
            mean = self.mean_map(data.df['sequence'])
            np.save(path, mean)
            return mean

    def matches(self, substring, trie):
        d = trie
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['words']

    def parse_dictionary(self, dictionary):
        dictionary_trie = {}
        for i,word in enumerate(dictionary):
            tmp_trie = dictionary_trie
            for letter in word:
                if letter not in tmp_trie:
                    tmp_trie[letter] = {}
                if 'words' not in tmp_trie[letter]:
                    tmp_trie[letter]['words'] = []
                tmp_trie[letter]['words'].append(i)
                tmp_trie = tmp_trie[letter]
        return dictionary_trie

    def map(self, sequence):
        vector = np.zeros(len(self.kmers))
        for i in range(len((sequence))-self.spectrum+1):
            for j in self.matches(sequence[i:i+self.spectrum],self.trie):
                if self.mode == 'count':
                    vector[j] += 1
                elif self.mode == 'indicate':
                    vector[j] = 1
        return np.array(vector)

    def mean_map(self, seqs):
        m = len(seqs)
        x = np.zeros(len(self.kmers))
        for seq in seqs:
            feat = self.map(seq)
            norm = np.sqrt(np.dot(feat,feat))
            if norm != 0:
                feat /= norm
            x += feat
        x /= m
        return x

    def mmd(self, x, y):
        if type(x) == list:
            x = self.mean_map(x)
        if type(y) == list:
            y = self.mean_map(y)
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))
        return MMD


def TrainTestValHoldout(dataset, sample_size, random_seed, return_holdouts=False):
    name = '_'.join(dataset.split(' ')+[str(sample_size),str(random_seed)])
    ds = Dataset(dataset)
    names = ds.names
    df = ds.df

    # these are manually selected label combinations for holdouts A-E
    combinations = [
        ['GO:0008144', 'GO:0022857'],
        ['GO:0003677', 'GO:0003723'],
        ['GO:0043169', 'GO:0015075'],
        ['GO:0036094', 'GO:0016301', 'GO:0140096', 'GO:0038023'],
        ['GO:0003824', 'GO:0043168', 'GO:0048037']
    ]

    holdouts = []
    for comb in combinations:
        _labels = set(comb)
        mask = df['labels'].map(lambda x: _labels.issubset(x))
        holdouts.append(df[mask])

    for comb in combinations:
        _labels = set(comb)
        mask = df['labels'].map(lambda x: _labels.issubset(x))
        df = df[~mask]

    ds = BaseDataset().from_df(df)
    #print(len(ds.df.index))
    #print(ds.terms['count'].min())
    ds.names = names

    if return_holdouts:
        return combinations, holdouts
    else:
        return _TrainTestVal(ds, sample_size, random_seed)

def TrainTestVal(dataset, sample_size, random_seed):
    ds = Dataset(dataset)
    return _TrainTestVal(ds, sample_size, random_seed)

def _TrainTestVal(data, sample_size, random_seed):
    '''
    Returns (and if necessary generates) the train, test and validation splits
    from <dataset> such that each label is represented with at least <sample_size> sequences.
    '''
    df = data.df.copy()
    terms = data.terms['term'].to_list()
    godag = data.godag
    name = '_'.join(data.names.split(' ')+[str(sample_size),str(random_seed)])
    test = TestingSet(data.names, df, sample_size, random_seed, name='test', terms=terms, godag=godag)
    df = data.df.copy()
    df.drop(df[df['id'].isin(test.df['id'])].index, inplace=True)
    val = TestingSet(data.names, df, sample_size, random_seed, name='val', terms=terms, godag=godag)
    if os.path.exists(PATH+'/traintestval/{}/train.csv'.format(name)):
        train = pd.read_csv(PATH+'/traintestval/{}/train.csv'.format(name), converters={'labels': lambda x: x.split('; ')})
        train = BaseDataset().from_df(train)
    else:
        train_df = data.df.copy()
        train_df.drop(train_df[train_df['id'].isin(test.df['id'])].index, inplace=True)
        train_df.drop(train_df[train_df['id'].isin(val.df['id'])].index, inplace=True)
        train = BaseDataset().from_df(train_df)
        train_df['labels'] = train_df['labels'].map(lambda labels: '; '.join(labels))
        train_df.to_csv(PATH+'/traintestval/{}/train.csv'.format(name), index=False)
    train.names = data.names
    return train, test, val


class TestingSet:
    '''
    Base class for a testing set. Implements creating, loading, and evaluation.
    '''
    def __init__(self, dataset, pool, sample_size, random_seed, name, terms, godag):
        pool = pool.sample(frac=1, random_state=random_seed)
        self.name = name
        self.seed = random_seed
        self.dataset = dataset
        self.sample_size = sample_size
        self.df = None
        self.means = None
        self.space = FeatureSpace()
        self.pool = pool
        self.data = BaseDataset().from_df(pool)
        self.terms = terms
        self.godag = godag
        self.load()

    def load(self):
        name = '_'.join(self.dataset.split(' ')+[str(self.sample_size),str(self.seed)])
        if os.path.exists(PATH+'/traintestval/{}/{}'.format(name, self.name)):
            TESTSET = pd.read_csv(PATH+'/traintestval/{}/{}/df.csv'.format(name, self.name), converters={'labels': lambda x: x.split('; ')})
            with open(PATH+'/traintestval/{}/{}/means.pkl'.format(name, self.name),'rb') as file:
                MEANS = pickle.load(file)
            with open(PATH+'/traintestval/{}/{}/global_mean.pkl'.format(name, self.name),'rb') as file:
                self.global_mean = pickle.load(file)
            with open(PATH+'/traintestval/{}/{}/metrics.json'.format(name, self.name),'r') as file:
                self.metrics = json.load(file)
            self.df = TESTSET
            self.terms = list(MEANS.keys())
            self.means = np.array(list(MEANS.values()))
        else:
            print('Set not found. Creating...')
            if not os.path.exists(PATH+'/traintestval'):
                os.mkdir(PATH+'/traintestval')
            if not os.path.exists(PATH+'/traintestval/{}'.format(name)):
                os.mkdir(PATH+'/traintestval/{}'.format(name))
            if not os.path.exists(PATH+'/traintestval/{}/{}'.format(name, self.name)):
                os.mkdir(PATH+'/traintestval/{}/{}'.format(name, self.name))

            test_means, TESTSET = self.create()
            self.df = TESTSET
            self.means = test_means
            self.global_mean = self.space.mean_map(TESTSET['sequence'].to_list())
            MEANS = {term:mean for term,mean in zip(self.terms,test_means)}

            self.metrics = self.evaluate_testset()

            TESTSET['labels'] = TESTSET['labels'].map(lambda labels: '; '.join(labels))
            TESTSET.to_csv(PATH+'/traintestval/{}/{}/df.csv'.format(name, self.name))
            with open(PATH+'/traintestval/{}/{}/means.pkl'.format(name, self.name),'wb') as file:
                pickle.dump(MEANS, file)
            with open(PATH+'/traintestval/{}/{}/global_mean.pkl'.format(name, self.name),'wb') as file:
                pickle.dump(self.global_mean, file)

    def create(self):
        TESTSET, TESTIDS = self.sample_set()
        test_means = self.set_means(TESTSET, TESTIDS)
        return test_means, TESTSET

    def get_group_ids(self, df, terms):
        df = df.reset_index(drop=True)
        ids = {key:[] for key in terms}
        for index,row in df.iterrows():
            for label in row['labels']:
                if label in ids:
                    ids[label].append(index)
                else:
                    print('Warning: Unidentified label {}'.format(label))
        return ids

    def set_means(self, set, ids):
        return np.array([self.space.mean_map(set.iloc[ids[term]]['sequence'].to_list()[:self.sample_size]) for term in self.terms])

    def sample_set(self):
        TESTSET = None
        sample_ids = {key:[] for key in self.terms}
        for term in self.terms:
            term_pool = self.pool[self.pool['labels'].map(lambda x: term in x)]
            remainder = self.sample_size-len(sample_ids[term])
            if len(term_pool.index) < remainder:
                raise Exception('Not enough sequences for term {}.'.format(term))
            if remainder > 0:
                sample = term_pool.sample(n=remainder)
                self.pool = self.pool.drop(sample.index)
                if TESTSET is None: TESTSET = sample
                else: TESTSET = TESTSET.append(sample)
            sample_ids = self.get_group_ids(TESTSET, self.terms)
            m = min([len(sample_ids[term]) for term in self.terms])
            print('\r{}/{}'.format(m,self.sample_size), end='')
        print()
        return TESTSET, sample_ids

    def evaluate_testset(self):
        n_rep = int((self.data.terms['count'].min()-self.sample_size)/self.sample_size)
        if n_rep < 1:
            raise Exception('Not enough sequences in dataset to evaluate testset with sample size {}.'.format(self.sample_size))
        #n_rep = 10 if n_rep > 10 else n_rep
        n_rep = 1
        val = []
        rnd = []
        for i in range(n_rep):
            VALSET, VALIDS = self.sample_set()
            RNDSET = VALSET.copy()
            RNDSET['sequence'] = np.random.permutation(RNDSET['sequence'].values)
            val.append(self.evaluate(VALSET, group_ids=VALIDS))
            rnd_eval = self.evaluate(RNDSET, group_ids=VALIDS)
            RNDSET['sequence'] = [['L']*2048]*len(RNDSET.index) # constant sequence to maximize MMD
            rnd_eval['global_distance'] = self.global_mmd(RNDSET)
            rnd.append(rnd_eval)
        val_metrics = dict(pd.DataFrame(val).mean())
        rnd_metrics = dict(pd.DataFrame(rnd).mean())
        val_metrics = {'val_'+key:val_metrics[key] for key in val_metrics}
        rnd_metrics = {'rnd_'+key:rnd_metrics[key] for key in rnd_metrics}
        metrics = {**val_metrics, **rnd_metrics}
        metrics['n_rep'] = n_rep
        for m in ['global_distance', 'mean_reciprocal_rank', 'mean_reciprocal_rank_wo_parents', 'mean_reciprocal_rank_wo_childs', 'mean_reciprocal_rank_wo_both']:
            metrics['val_'+m+'_std'] = np.std([v[m] for v in val])
            metrics['rnd_'+m+'_std'] = np.std([r[m] for r in rnd])

        name = '_'.join(self.dataset.split(' ')+[str(self.sample_size),str(self.seed)])
        for m in ['val_term_distances','val_reciprocal_ranks','val_reciprocal_ranks_wo_parents','val_reciprocal_ranks_wo_childs','val_reciprocal_ranks_wo_both','rnd_term_distances','rnd_reciprocal_ranks','rnd_reciprocal_ranks_wo_parents','rnd_reciprocal_ranks_wo_childs','rnd_reciprocal_ranks_wo_both']:
            if 'distance' in m: kwargs = {'vmin': max(metrics[m]),'vmax':0}
            else: kwargs = {}
            plot_dag(self.terms, metrics[m], self.godag, PATH+'/traintestval/{}/{}/{}.png'.format(name, self.name, m), **kwargs)

        with open(PATH+'/traintestval/{}/{}/metrics.json'.format(name, self.name), 'w') as file:
            json.dump(metrics, file, cls=NumpyEncoder)

        return metrics

    def evaluate(self, df, group_ids=None):
        metrics = {}
        if group_ids is None:
            means = self.set_means(df, self.get_group_ids(df, self.terms))
        else:
            means = self.set_means(df, group_ids)
        metrics['global_distance'] = self.global_mmd(df)
        metrics['term_distances'] = self.term_mmd(means)
        metrics['reciprocal_ranks'] = self.rr(means)
        metrics['reciprocal_ranks_wo_parents'] = self.rr(means, ignore_parents=True)
        metrics['reciprocal_ranks_wo_childs'] = self.rr(means, ignore_childs=True)
        metrics['reciprocal_ranks_wo_both'] = self.rr(means, ignore_parents=True, ignore_childs=True)
        metrics['mean_reciprocal_rank'] = np.mean(metrics['reciprocal_ranks'])
        metrics['mean_reciprocal_rank_wo_parents'] = np.mean(metrics['reciprocal_ranks_wo_parents'])
        metrics['mean_reciprocal_rank_wo_childs'] = np.mean(metrics['reciprocal_ranks_wo_childs'])
        metrics['mean_reciprocal_rank_wo_both'] = np.mean(metrics['reciprocal_ranks_wo_both'])
        return metrics

    def global_mmd(self, seqs):
        if type(seqs) == list or type(seqs) == np.ndarray:
            return self.space.mmd(seqs,self.global_mean)
        else:
            return self.space.mmd(seqs['sequence'].to_list(),self.global_mean)

    def term_mmd(self, means):
        return np.array([self.space.mmd(x,y) for x,y in zip(means,self.means)])

    def rr(self, means, ignore_parents=False, ignore_childs=False):
        positions = []
        for x,term in zip(means,self.terms):
            d = [self.space.mmd(x,y) for y in self.means]
            s = np.argsort(d)
            t = [self.terms[i] for i in s]
            if ignore_parents:
                t = [_t for _t in t if not self.godag.get(term).has_parent(_t)]
            if ignore_childs:
                t = [_t for _t in t if not self.godag.get(term).has_child(_t)]
            positions.append(t.index(term))
        rr = [1/(r+1) for r in positions]
        return np.array(rr)

    def mrr(self, means, ignore_parents=False, ignore_childs=False):
        rr = self.rr(means, ignore_parents=ignore_parents, ignore_childs=ignore_parents)
        mrr = np.mean(rr)
        return mrr

    def pval(self, x, y):
        m = len(x)
        original_mmd = self.space.mmd(x,y)
        aggregated = x+y
        t = 1000
        space_mmd = self.space.mmd

        def random_mmd():
            random.shuffle(aggregated)
            x_hat = aggregated[:m]
            y_hat = aggregated[m:]
            return space_mmd(x_hat,y_hat)

        if t > 100:
            MMDs = Parallel(n_jobs=20)(delayed(random_mmd)() for i in tqdm(range(t)))
        else:
            MMDs = [random_mmd() for i in range(t)]
        rank = float(sum([mmd<=original_mmd for mmd in MMDs]))+1
        pval = (t+1-rank)/(t+1)
        return pval

    def global_pval(self, df):
        x = df['sequence'].to_list()
        y = self.df['sequence'].to_list()
        return self.pval(x,y)

    def term_pval(self, df):
        print('Computing pValues...')
        ids = self.get_group_ids(df, self.terms)
        testset_ids = self.get_group_ids(self.df, self.terms)
        pvals = []
        for term in tqdm(self.terms):
            term_seqs = df.iloc[ids[term]]['sequence'].to_list()[:self.sample_size]
            testset_term_seqs = self.df.iloc[testset_ids[term]]['sequence'].to_list()[:self.sample_size]
            pvals.append(self.pval(term_seqs,testset_term_seqs))
        return np.array(pvals)
