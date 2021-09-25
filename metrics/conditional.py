import os
import numpy as np
from metrics.similarity import mmd
from embed.spectrum import spectrum_map
from data.util import labels as terms, GoDag

PATH = os.path.dirname(os.path.realpath(__file__))

def mrr(seq1=None, labels1=None, seq2=None, labels2=None, emb1=None, emb2=None, n=1300, embedding='spectrum', ignore_parents=False, ignore_childs=False, godag=PATH+'/../data/godag.obo', return_ranks=False, terms=terms, warning=True, **kwargs):
    '''
    Calculates MRR between two sets of sequences and associated labels. <seq2> and <labels2> is the reference set. Optionally takes embeddings of sequences if these have been precomputed for efficiency. <n> is the size of each label bucket. <ignore_parents> and <ignore_childs> are flags to ignore off-target effects in related terms in the label hierarchy.
    '''

    if ignore_parents or ignore_childs:
        godag = GoDag(godag)

    if embedding == 'spectrum':
        embed = spectrum_map
    if embedding == 'profet':
        raise NotImplementedError
    if embedding == 'unirep':
        raise NotImplementedError

    if not seq1 is None and emb1 is None:
        emb1 = embed(seq1, **kwargs)
    if not seq2 is None and emb2 is None:
        emb2 = embed(seq2, **kwargs)

    def group(embs, labels):
        groups = {term:[] for term in terms}
        for emb,lab in zip(embs, labels):
            for label in lab:
                if label in groups:
                    groups[label].append(emb)
        return groups

    def mean_embedding(groups):
        means = {}
        for term, emb in groups.items():
            means[term] = np.mean(emb[:n], axis=0)
        return means

    groups1 = group(emb1,labels1)
    groups2 = group(emb2,labels2)

    len1 = [len(v) for k,v in groups1.items()]
    len2 = [len(v) for k,v in groups2.items()]
    if warning and not (
            all(len(v)>=n for k,v in groups1.items())
            and
            all(len(v)>=n for k,v in groups2.items())
            ):
        print('Warning: Not enough sequences for each label to calculate MRR. It is still returned, but not correct.')

    # filter terms with no sequence samples
    terms1 = [term for term in terms if len(groups1[term]) > 0]
    terms2 = [term for term in terms if len(groups2[term]) > 0]
    groups1 = {term:groups1[term] for term in terms1}
    groups2 = {term:groups2[term] for term in terms2}

    means1 = mean_embedding(groups1)
    means2 = mean_embedding(groups2)

    ranks = {}
    for term1 in terms1:
        distances = [mmd(mean1=means1[term1], mean2=means2[term2]) for term2 in terms2]
        term_ranks = np.argsort(distances)
        ranked_terms = [terms2[i] for i in term_ranks]
        if ignore_parents:
            ranked_terms = [t for t in ranked_terms if not godag.get(term1).has_parent(t)]
        if ignore_childs:
            ranked_terms = [t for t in ranked_terms if not godag.get(term1).has_child(t)]
        ranks[term1] = ranked_terms.index(term1)+1
    if return_ranks:
        return ranks
    reciprocal_ranks = [1/r for t,r in ranks.items()]
    return np.mean(reciprocal_ranks)
