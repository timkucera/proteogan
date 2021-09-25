from embed.spectrum import spectrum_map
import numpy as np
from scipy.stats import entropy as scipy_entropy

def entropy(seq1=None, seq2=None, emb1=None, emb2=None, embedding='spectrum', **kwargs):
    '''
    Calculates the average entropy over embedding dimensions between two sets of sequences. Optionally takes embeddings of sequences if these have been precomputed for efficiency.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map
    if embedding == 'profet':
        raise NotImplementedError
    if embedding == 'unirep':
        raise NotImplementedError

    if not seq1 is None:
        emb1 = embed(seq1, **kwargs)
    if not seq2 is None:
        emb2 = embed(seq2, **kwargs)

    lo = np.min(np.vstack((emb1,emb2)),axis=0)
    hi = np.max(np.vstack((emb1,emb2)),axis=0)

    def _entropy(emb):
        res = 0
        for i,col in enumerate(emb.T):
            hist, _ = np.histogram(col, bins=1000, range=(lo[i],hi[i]))
            res += scipy_entropy(hist, base=2)
        res = res / emb.shape[1]
        return res

    return _entropy(emb2)-_entropy(emb1)


def distance(seq1=None, seq2=None, emb1=None, emb2=None, embedding='spectrum'):
    '''
    Calculates the average pairwise distance between two sets of sequences in mapping space. Optionally takes embeddings of sequences if these have been precomputed for efficiency.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map
    if embedding == 'profet':
        raise NotImplementedError
    if embedding == 'unirep':
        raise NotImplementedError

    if not seq1 is None:
        emb1 = embed(seq1, **kwargs)
    if not seq2 is None:
        emb2 = embed(seq2, **kwargs)

    def _distance(emb):
        res = np.sum(emb ** 2, axis=1, keepdims=True) + np.sum(emb ** 2, axis=1, keepdims=True).T - 2 * np.dot(emb, emb.T)
        res = np.mean(res)
        return res

    return _distance(emb2)-_distance(emb1)
