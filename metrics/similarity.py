import numpy as np
from embed.spectrum import spectrum_map
from falkon import Falkon, kernels
import torch
from tqdm import tqdm

def mmd(seq1=None, seq2=None, emb1=None, emb2=None, mean1=None, mean2=None, embedding='spectrum', kernel='linear', return_pvalue=False, progress=False, **kwargs):
    '''
    Calculates MMD between two sets of sequences. Optionally takes embeddings or mean embeddings of sequences if these have been precomputed for efficiency. If <return_pvalue> is true, a Monte-Carlo estimate (1000 iterations) of the p-value is returned. Note that this is compute-intensive and only implemented for the linear kernel.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map
    if embedding == 'profet':
        raise NotImplementedError
    if embedding == 'unirep':
        raise NotImplementedError

    if mean1 is None and emb1 is None:
        emb1 = embed(seq1, progress=progress, **kwargs)
    if mean2 is None and emb2 is None:
        emb2 = embed(seq2, progress=progress, **kwargs)

    if mean1 is None:
        x = np.mean(emb1, axis=0)
    else:
        x = mean1
    if mean2 is None:
        y = np.mean(emb2, axis=0)
    else:
        y = mean2

    if kernel == 'linear':
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))
        if return_pvalue:
            m = len(emb1)
            agg = np.concatenate((emb1,emb2),axis=0)
            mmds = []
            it = tqdm(range(1000)) if progress else range(1000)
            for i in it:
                np.random.shuffle(agg)
                _emb1 = agg[:m]
                _emb2 = agg[m:]
                mmds.append(mmd(emb1=_emb1, emb2=_emb2))
            rank = float(sum([x<=MMD for x in mmds]))+1
            pval = (1000+1-rank)/(1000+1)
            return MMD, pval
        else:
            return MMD

    elif kernel == 'gaussian':
        gauss = kernels.GaussianKernel(sigma=1.0)
        x = torch.from_numpy(emb1)
        y = torch.from_numpy(emb2)
        m = float(len(emb1))
        n = float(len(emb2))
        Kxx = gauss(x,x).numpy()
        Kxy = gauss(x,y).numpy()
        Kyy = gauss(y,y).numpy()
        return np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )
