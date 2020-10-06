## ProteoGAN
This repository contains code accompanying "Conditional Generative Modeling for De Novo Hierarchical Multi-Label Functional Protein Design".


## Installation
First clone and cd into the repository. You can install our preconfigured environment via conda:
```sh
conda env create --file environment.yml
conda activate proteogan
pip install bioservices # conda can throw a compatibility error here, hence install with pip
```
or install the requirements yourself:
```sh
conda install -c anaconda tensorflow-gpu=2.0 # Has only been tested with 2.0
conda install pandas numpy matplotlib seaborn scikit-learn
conda install -c bioconda goatools
conda install -c conda-forge tqdm
pip install bioservices # conda can throw a compatibility error here, hence install with pip
```
We included our training data in this repository. Extract it with:
```sh
tar -xzvf ./data/datasets/base/data.csv.tar.gz -C ./data/datasets/base/
```
Then execute like:
```sh
python -m examples.generate
```


## Generate and evaluate sequences with ProteoGAN
```python
# load data splits
train, test, val = TrainTestVal('base L50', 300)

# load model
proteogan = Trainable(train, val, logdir='./test')

# load trained weights
proteogan.load('./weights/ckpt')

# generate some sequences based on test set labels
df = proteogan.generate(test)

# evaluate metrics
metrics = test.evaluate(df)
```
Or run:
```sh
python -m examples.generate
```

## Train ProteoGAN
```python
# load data splits
train, test, val = TrainTestVal('base L50', 300)

# load model
proteogan = Trainable(train, val, logdir='./test')

# training loop
# train for at least 20 epochs to get reasonable results
for epoch in tqdm(range(5)):
    proteogan.train_epoch()

# save a final checkpoint
proteogan.save('./test/ckpt')

```
Or run:
```sh
python -m examples.train
```


## Create own dataset
We provide the data we used to train _ProteoGAN_, but you can pull an updated version from UniProt with our _data.dataset_ API. You can also create your own dataset with other filters. Please first create a configuration in the _data/config.json_ file:
```python
[
    { # each object in the array creates a data.dataset.RawDataset with the raw data from UniProt. If the data does not exist, it will be downloaded automatically.
        "name": "myNewDataset", # give it a name
        "query": "(existence:\"evidence at transcript level\" OR existence:\"evidence at protein level\") goa:(evidence:manual) go:0003674", # specify the query that should be used to pull data from UniProt. We recommend leaving this setting as is, but you could include other ontologies or more sequences.
        "godag": "myNewDataset.obo", # specify the GO DAG. Should be the same as the RawData name above (will be downloaded along with it automatically), but you can specify your own GO DAG definition.
        "datasets": [{ # specify a dataset. This will create the actual dataset from the raw data according to the filters you specify below
            "name": "100labels", # specify a name. The dataset can then be loaded with data.dataset.Dataset('myNewDataset 100labels')
            "filter": [ # you can implement your own filters in data/filters.py
                {"filter_fx": "maximum_length", "filter_args": [2048]}, # maximal protein sequence length
                {"filter_fx": "n_largest_classes", "filter_args": [100]}, # choose the 100 largest GO classes
                {"filter_fx": "amino_acids", "filter_args": []} # only use the standard amino acids
            ]
        }]
    }
]
```
You can then simply load your data with
```python
from data.dataset import Dataset
ds = Dataset('myNewDataset 100labels')
```
The data will be automatically downloaded and preprocessed. Remember to regenerate train, test, validation splits if you create a new dataset (see below). Also, change the constants in model.constants accordingly.


## Create own test and validation sets
Simply specify the dataset from which the splits should be created, and the minimum sequence number per label:
```python
from eval.eval import TrainTestVal
train, test, val = TrainTestVal('myNewDataset 100labels', 300)
```
This will create the splits and evaluate them to define positive and negative control values. They are saved in _eval/traintestval/myNewDataset_100labels_300/{test or val}/metrics.json_.


## Implement your own model
If you would like to implement your own model but use our data, training and evaluation pipeline, feel free to subclass model.model.GenerativeModel and use with model.train.Trainable.


## Citation
If you use code from this repository in your work we kindly ask to cite accordingly:
```
@inproceedings{
    anonymous2021conditional,
    title={Conditional Generative Modeling for De Novo Hierarchical Multi-Label Functional Protein Design},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=eHg0cXYigrT},
    note={under review}
}
```
