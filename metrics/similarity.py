import numpy as np
from embed.spectrum import spectrum_map
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel

def mmd(seq1=None, seq2=None, emb1=None, emb2=None, mean1=None, mean2=None, embedding='spectrum', kernel='linear', kernel_args={}, return_pvalue=False, progress=False, **kwargs):
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

    if not mean1 is None and not mean2 is None:
        MMD = np.sqrt(np.dot(mean1,mean1) + np.dot(mean2,mean2) - 2*np.dot(mean1,mean2))
        return MMD

    if kernel == 'linear':
        x = np.mean(emb1, axis=0)
        y = np.mean(emb2, axis=0)
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))
    elif kernel == 'gaussian':
        x = np.array(emb1)
        y = np.array(emb2)
        m = x.shape[0]
        n = y.shape[0]
        Kxx = rbf_kernel(x,x, **kernel_args)#.numpy()
        Kxy = rbf_kernel(x,y, **kernel_args)#.numpy()
        Kyy = rbf_kernel(y,y, **kernel_args)#.numpy()
        MMD = np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )

    if return_pvalue:
        agg = np.concatenate((emb1,emb2),axis=0)
        mmds = []
        it = tqdm(range(1000)) if progress else range(1000)
        for i in it:
            np.random.shuffle(agg)
            _emb1 = agg[:m]
            _emb2 = agg[m:]
            mmds.append(mmd(emb1=_emb1, emb2=_emb2, kernel=kernel, kernel_args=kernel_args))
        rank = float(sum([x<=MMD for x in mmds]))+1
        pval = (1000+1-rank)/(1000+1)
        return MMD, pval
    else:
        return MMD
