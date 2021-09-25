import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import os, json
from data.util import tokenize, tokenize_labels, detokenize_sequences
from util.fasta import save_as_fasta
from metrics.similarity import mmd
from metrics.conditional import mrr
import itertools
from collections import defaultdict
import math, time
sns.set_style("whitegrid")

def batch_dont_shuffle(data, batch_size):
    sequences, labels = data
    return [[sequences[i:i+batch_size], labels[i:i+batch_size]] for i in range(0, len(sequences), batch_size)]

def batch_and_shuffle(data, batch_size):
    sequences, labels = data
    transposed = list(zip(sequences, labels))
    random.shuffle(transposed)
    shuffled = list(zip(*transposed))
    batched = batch_dont_shuffle(shuffled, batch_size)
    return batched

class BaseTrainer():

    '''
    Base class for training a model. Takes care of the training loop,
    sequence generation and plotting. Subclass BaseTrainer and implement
    the indicated methods for your own model.
    '''

    def __init__(self,
            train_data = None,
            test_data = None,
            val_data = None,
            batch_size = None,
            path = '.',
            plot_every = 1,
            checkpoint_every = 100,
            pad_sequence = False,
            pad_labels = False,
            add_eos_token = False,
            short_val = False,
            config = None,
            ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.steps_per_epoch = 0

        pad_sequence_to = False if not pad_sequence else config['seq_length']

        if not train_data is None:
            self.tokenized_train = tokenize(train_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)
            self.steps_per_epoch = math.ceil(len(train_data)/batch_size)
            self.plot_interval = math.ceil(self.steps_per_epoch*plot_every)
            self.checkpoint_interval = int(self.steps_per_epoch*checkpoint_every)
        if not test_data is None:
            self.tokenized_test = tokenize(test_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)
        if not val_data is None:
            self.tokenized_val = tokenize(val_data, pad_sequence_to=pad_sequence_to, add_eos_token=add_eos_token, pad_labels=pad_labels)

        self.path = path if path.endswith('/') else path+'/'
        os.makedirs(self.path, exist_ok=True)
        with open(self.path+'config.json','w') as file:
            json.dump(config,file)
        self.step = 0
        self.metrics = []
        self.best = {'MMD':0, 'MRR':0, 'ratio':0}
        self.gen_length = config['seq_length'] if not short_val else 300

    def train(self, epochs, progress=False, restore=True):
        self.epochs = epochs
        if restore:
            self.restore()
        else:
            self.create_optimizer()
        last_epoch = int(self.step/self.steps_per_epoch)
        if restore:
            print('Restored model from epoch {}'.format(last_epoch))
        if progress:
            it = tqdm(range(last_epoch,epochs+1), total=epochs, initial=last_epoch)
        else:
            it = range(last_epoch,epochs+1)
        for epoch in it:
            self.train_epoch()
        self.store()

    def train_epoch(self):
        x = batch_and_shuffle(self.tokenized_train, self.batch_size)
        losses = defaultdict(lambda: [])
        for batch in batch_and_shuffle(self.tokenized_train, self.batch_size):
            loss = self.train_batch(batch)
            self.step += 1
            losses = {k: losses[k]+[v] for k,v in loss.items()}
            if self.step % self.plot_interval == 0:
                losses = {k:np.mean(v) for k,v in losses.items()}
                metrics = {'Step':self.step, **losses, **self.eval(seed=0)}
                self.metrics.append(metrics)
                self.plot()
                losses = defaultdict(lambda: [])
                if metrics['MRR']/metrics['MMD'] > self.best['ratio']:
                    self.store(best=True)
                    self.best = {
                        'MMD': metrics['MMD'],
                        'MRR': metrics['MRR'],
                        'ratio': metrics['MRR']/metrics['MMD']
                    }
            if self.step % self.checkpoint_interval == 0:
                self.store()

    def create_optimizer(self):
        '''
        Implement yourself. Should create an optimizer for the model.
        '''
        raise NotImplementedError

    def train_batch(self, batch):
        '''
        Implement yourself. Should train a single batch and return a dict of losses.
        '''
        raise NotImplementedError

    def predict_batch(self, batch, seed=None):
        '''
        Implement yourself. Should return a matrix of tokenized sequences.
        '''
        raise NotImplementedError

    def eval(self, seed=None):
        val_labels = self.val_data['labels'].tolist()
        val_seqs = self.val_data['sequence'].tolist()
        generated = []
        labels = []
        for batch in batch_dont_shuffle(self.tokenized_val, self.batch_size):
            seqs = self.predict_batch(batch, seed=seed)
            seqs = detokenize_sequences(seqs)
            generated.extend(seqs)
        save_as_fasta(zip(generated,val_labels), self.path+'generated.fasta')
        metrics = {
            'MMD': mmd(generated, val_seqs),
            'MRR': mrr(generated, val_labels, val_seqs, val_labels, warning=False),
        }
        return metrics

    def plot(self):
        df = pd.DataFrame(self.metrics)
        plt.figure()
        colors = ['#f44336','#b23c17','#2196f3','#00bcd4','#ffc107','#9c27b0','#009688','#8bc34a']
        names = sorted(list(set(df.columns) - set(('MMD','MRR','Step'))))
        for i, name in enumerate(names):
            ax = sns.lineplot(data=df, x='Step', y=name, legend=False, color=colors[i+2])
        plt.xlabel('Parameter Update', fontdict = {'size': 12})
        plt.ylabel('Loss', fontdict = {'size': 12})
        ax.grid(False)
        ax2 = ax.twinx()
        ax = sns.lineplot(data=df, x='Step', y='MMD', legend=False, ax=ax2, color=colors[0])
        ax = sns.lineplot(data=df, x='Step', y='MRR', legend=False, ax=ax2, color=colors[1])
        plt.ylabel('MMD & MRR')
        lines = [Line2D([0], [0], color=colors[i], lw=2) for i, name in enumerate(['MMD','MRR']+names)]
        ax.legend(lines, ['MMD','MRR']+names, bbox_to_anchor=(0,0.98,1,0.2), loc="upper left", mode="expand", borderaxespad=0, ncol=3, frameon=False)
        plt.tight_layout()
        plt.savefig(self.path+'plot.png', dpi=350)
        plt.close()

    def generate(self, labels=None, progress=False, return_df=False, seed=None):
        if labels is None:
            labels = self.test_data['labels'].tolist()
        tokenized_labels = [tokenize_labels(l) for l in labels]
        data = ([[]]*len(tokenized_labels),tokenized_labels)
        batched_data = batch_dont_shuffle(data, self.batch_size)
        it = tqdm(batched_data) if progress else batched_data
        generated = [self.predict_batch(batch, seed=seed) for batch in it]
        generated = np.array(list(itertools.chain(*generated)))
        generated = detokenize_sequences(generated)
        if return_df:
            return pd.DataFrame({'sequence':generated, 'labels':labels})
        else:
            return generated

    def store(self, best=False):
        dir = 'best/' if best else 'checkpoint/'
        os.makedirs(self.path+dir, exist_ok=True)
        props = {
                'best': self.best,
                'step': self.step
                }
        with open(self.path+dir+'props.json','w') as file:
            json.dump(props, file)
        with open(self.path+dir+'metrics.json','w') as file:
            json.dump(self.metrics, file)
        self.store_model(self.path+dir)

    def restore(self, path=None, best=False):
        path = self.path if path is None else path
        path = path if path.endswith('/') else path+'/'
        dir = 'best/' if best else 'checkpoint/'
        if not os.path.exists(path+dir):
            raise 'Checkpoint for restoring the Trainer not found.'
        with open(path+dir+'props.json','r') as file:
            props = json.load(file)
        with open(path+dir+'metrics.json','r') as file:
            self.metrics = json.load(file)
        self.best = props['best']
        self.step = props['step']
        self.create_optimizer()
        self.restore_model(path+dir)

    def store_model(self, path):
        '''
        Implement yourself. Should save all model-related files and optimizer state.
        '''
        raise NotImplementedError

    def restore_model(self, path):
        '''
        Implement yourself. Should reload all model-related files and optimizer state.
        '''
        raise NotImplementedError
