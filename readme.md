## ProteoGAN
This repository contains code accompanying [**"Conditional generative modeling for *de novo* protein design with hierarchical functions"**](https://doi.org/10.1093/bioinformatics/btac353). It provides the data used to train and evaluate the model, the evaluation metrics, and ProteoGAN itself with pretrained weights.

Note that, due to conflicts with our license, evaluations with UniRep and ProFET embeddings are not included in this repository. However, it is relatively straightforward to include any alternative embedding by providing a mapping function to the modules in `proteogan.metrics`. (See also bibliography notes below)

To train ProteoGAN, run `python -m train.train proteogan 1`, where `1` is the split number (1 to 5). Loss curves, generated sequences and model weights are saved to `train/split_1/proteogan/`. To evaluate the metrics MMD and MRR, run `python -m train.eval proteogan 1`. Pretrained weights for split 1 are provided with this repository.

You may use the MMD and MRR metrics to evaluate your own generated sequences. Note that the metric values depend on the reference sequence set they are applied to. We provide the reference sets of the paper in this repository. Load them with:

```python
from data.util import load

split = 1
train = load('train', split)
test = load('test', split)
val = load('val', split)
```

The dataset contains primary sequences of 150.000 proteins and their annotations of 50 GO molecular function labels (compare paper).

Once you have generated some sequences, provide them to MMD and MMR. MRR needs the labels the generated sequences were conditioned on as a list of lists. Usually these would be the labels of a test set you want to compare to.

```python
from metrics.similarity import mmd
from metrics.conditional import mrr

generated = ['MLAVEGSALLVS...', 'MALSEGALSVGELA...', ...] # use your model to generate these, conditioned on test['labels']

MMD = mmd(generated, test['sequence'])
MRR = mrr(generated, test['labels'], test['sequence'], test['labels'])

```

You may also use `train.util.BaseTrainer` to implement your own model and reuse our training and evaluation loop. See `train/train.py` for details.


## Citation
If you use code from this repository in your work we kindly ask to cite accordingly:

> Kucera, T., Togninalli, M., & Meng-Papaxanthos, L. (2022). Conditional generative modeling for de novo protein design with hierarchical functions. Bioinformatics, 38(13), 3454-3461.

```
@article{10.1093/bioinformatics/btac353,
    author = {Kucera, Tim and Togninalli, Matteo and Meng-Papaxanthos, Laetitia},
    title = "{Conditional generative modeling for de novo protein design with hierarchical functions}",
    journal = {Bioinformatics},
    volume = {38},
    number = {13},
    pages = {3454-3461},
    year = {2022},
    month = {05},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac353},
    url = {https://doi.org/10.1093/bioinformatics/btac353},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/13/3454/44268843/btac353.pdf},
}
```


## Bibliography

The code in this repository is based on the following works:

```

Abadi, Martín, et al. "Tensorflow: A system for large-scale machine learning." 12th {USENIX} symposium on operating systems design and implementation ({OSDI} 16). 2016.

Da Costa-Luis, Casper, et al. "tqdm: A fast, Extensible Progress Bar for Python and CLI." Zenodo. Apr (2021).

Harris, Charles R., et al. "Array programming with NumPy." Nature 585.7825 (2020): 357-362.

Hunter, John D. "Matplotlib: A 2D graphics environment." Computing in science & engineering 9.03 (2007): 90-95.

Klopfenstein, D. V., et al. "GOATOOLS: A Python library for Gene Ontology analyses." Scientific reports 8.1 (2018): 1-17.

McKinney, Wes. "Data structures for statistical computing in python." Proceedings of the 9th Python in Science Conference. Vol. 445. 2010.

Meanti, Giacomo, et al. "Kernel methods through the roof: handling billions of points efficiently." arXiv preprint arXiv:2006.10350 (2020).

Paszke, Adam, et al. "Pytorch: An imperative style, high-performance deep learning library." Advances in neural information processing systems 32 (2019): 8026-8037.

Virtanen, Pauli, et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python." Nature methods 17.3 (2020): 261-272.

Waskom, Michael L. "Seaborn: statistical data visualization." Journal of Open Source Software 6.60 (2021): 3021.

```

Although not used in this repository, we refer to the following works as they have been very helpful to obtain the results in the paper:

```

Alley, Ethan C., et al. "Unified rational protein engineering with sequence-based deep representation learning." Nature methods 16.12 (2019): 1315-1322.

Ma, Eric J., and Arkadij Kummer. "Reimplementing Unirep in JAX." bioRxiv (2020).

Ofer, Dan, and Michal Linial. "ProFET: Feature engineering captures high-level protein functions." Bioinformatics 31.21 (2015): 3429-3436.

```
