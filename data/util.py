import os, json
import pandas as pd
import numpy as np
from util.constants import amino_acid_alphabet

PATH = os.path.dirname(os.path.realpath(__file__))

with open(PATH+'/labels.txt') as file:
    labels = file.read().split()
with open(PATH+'/labels100.txt') as file:
    labels100 = file.read().split()
with open(PATH+'/labels200.txt') as file:
    labels200 = file.read().split()

seq_tokens = list('-'+amino_acid_alphabet)
label_tokens = labels+['<PAD>']
all_tokens = np.array(seq_tokens + label_tokens)

aa_map = {k:v for v,k in enumerate(seq_tokens)}
seq_token_map = {v:k for k,v in aa_map.items()}
label_map = {k:v for v,k in enumerate(label_tokens)}
label_map['<PAD>'] = 0

def load(dataset=None, split=None, path=None):
    '''
    Loads a data split into a pd.DataFrame. Protein sequences are in the 'sequence' column, labels are lists of GO terms in 'labels'
    '''
    if not path is None:
        return pd.read_csv(path, converters={'labels':lambda x: x.split(' ')}, na_filter=False)
    assert dataset in ['train','test','val','holdout','all']
    if dataset == 'all':
        return pd.read_csv(PATH+'/all.csv', converters={'labels':lambda x: x.split(' ')}, na_filter=False)
    elif dataset == 'holdout':
        return pd.read_csv(PATH+'/holdout/{}.csv'.format(split), converters={'labels':lambda x: x.split(' ')}, na_filter=False)
    else:
        return pd.read_csv(PATH+'/split_{}/{}.csv'.format(split,dataset), converters={'labels':lambda x: x.split(' ')}, na_filter=False)

def save(df, path):
    '''
    Saves a generated DataFrame to csv. Takes care of encoding the lists of labels.
    '''
    df['labels'] = df['labels'].map(lambda x: ' '.join(x))
    df.to_csv(path)



def tokenize_sequence(sequence, pad_to=False, add_eos_token=False):
    if add_eos_token:
        sequence += '$'
    if pad_to:
        sequence += '-'*(pad_to-len(sequence))
        sequence = sequence[:pad_to]
    return np.array([aa_map[aa] for aa in sequence])

def detokenize_sequence(sequence, remove_padding=True):
    sequence = ''.join([seq_token_map.get(int(i),'?') for i in sequence])
    if remove_padding:
        sequence = sequence.replace('-','')
        sequence = sequence.replace('?','')
        sequence = sequence.replace('$','')
    return sequence

def detokenize_sequences(sequences, remove_padding=True, remove_labels=True):
    if remove_labels:
        sequences[sequences >= len(seq_tokens)] = 0
    sequences = all_tokens[np.array(sequences)]
    if remove_padding:
        join_fx = lambda s: ''.join(s).replace('-','').replace('?','').replace('$','')
    else:
        join_fx = lambda s: ''.join(s)
    sequences = [join_fx(s) for s in sequences]
    return sequences

def tokenize_labels(labels, pad=False):
    if pad:
        all_labels = list(filter(lambda x: x != '<PAD>', label_map.keys()))
        labels = [k if k in labels else '<PAD>' for k in all_labels] #labels + ['<PAD>']*(50-len(labels))
    return np.array([label_map[label] for label in labels])

def tokenize(df, pad_sequence_to=False, add_eos_token=False, pad_labels=False):
    tokenized_sequences = [tokenize_sequence(seq, pad_to=pad_sequence_to, add_eos_token=add_eos_token) for seq in df['sequence'].tolist()]
    tokenized_labels = [tokenize_labels(seq, pad=pad_labels) for seq in df['labels'].tolist()]
    return tokenized_sequences, tokenized_labels

def embed_sequence(tokens):
    return np.eye(len(aa_map))[tokens]

def embed_labels(tokens):
    return np.sum(np.eye(len(labels))[tokens], axis=0)

def embed_and_tokenize_sequence(sequence, pad_to=False):
    tokens = tokenize_sequence(sequence, pad_to=pad_to)
    return embed_sequence(tokens)

def embed_and_tokenize_labels(labels):
    tokens = tokenize_labels(labels)
    return embed_labels(tokens)



from goatools.obo_parser import GODag as _GoDag
from goatools.godag.go_tasks import get_go2ancestors

class GoDag():
    '''
    GO DAG class to represent the ontology. Contains helper functions to get all parents or leaf nodes.
    '''
    def __init__(self, path=PATH+'/godag.obo'):
        self.GODAG = _GoDag(path, prt=None)

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

    def get_leaf_nodes(self, terms):
        gos = set(terms)
        if len(gos) > 1:
            leaves = []
            for go in gos:
                childs = set([i.id for i in self.get(go).children])
                inter = gos.intersection(childs)
                if len(inter) == 0:
                    leaves.append(go)
            if len(leaves) > 1:
                gos = set(leaves)
        return list(gos)
