import numpy as np
from util.constants import amino_acid_alphabet
import itertools
from tqdm import tqdm

def make_kmer_trie(k):
    '''
    For efficient lookup of k-mers.
    '''
    kmers = [''.join(i) for i in itertools.product(amino_acid_alphabet, repeat = k)]
    kmer_trie = {}
    for i,kmer in enumerate(kmers):
        tmp_trie = kmer_trie
        for aa in kmer:
            if aa not in tmp_trie:
                tmp_trie[aa] = {}
            if 'kmers' not in tmp_trie[aa]:
                tmp_trie[aa]['kmers'] = []
            tmp_trie[aa]['kmers'].append(i)
            tmp_trie = tmp_trie[aa]
    return kmer_trie

three_mer_trie = make_kmer_trie(3)

def spectrum_map(sequences, k=3, mode='count', normalize=True, progress=False):
    '''
    Maps a set of sequences to k-mer vector representation.
    '''

    if isinstance(sequences, str):
        sequences = [sequences]
    if k==3:
        trie = three_mer_trie
    else:
        trie = make_kmer_trie(k)

    def matches(substring):
        d = trie
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['kmers']

    def map(sequence):
        vector = np.zeros(len(amino_acid_alphabet)**k)
        for i in range(len((sequence))-k+1):
            for j in matches(sequence[i:i+k]):
                if mode == 'count':
                    vector[j] += 1
                elif mode == 'indicate':
                    vector[j] = 1
        feat = np.array(vector)
        if normalize:
            norm = np.sqrt(np.dot(feat,feat))
            if norm != 0:
                feat /= norm
        return feat

    it = tqdm(sequences) if progress else sequences
    return np.array([map(seq) for seq in it], dtype=np.float32)
