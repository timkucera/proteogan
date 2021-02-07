# -*- coding: utf-8 -*-
import os
from model.train import Trainable
from eval.eval import TrainTestValHoldout
from tqdm import tqdm

# load data splits
# first argument is the dataset name, then the minimum members per class, and the split number
train, test, val = TrainTestValHoldout('base L50', 1300, 1)

# specify a logging directory
# the trainable will dump some info and plots into the logdir directory, as well as the best model checkpoint
log_dir = './test'

# load model
proteogan = Trainable(train, val, logdir=log_dir)

# training loop
# train for at least 20 epochs to get reasonable results
for epoch in tqdm(range(20)):
    proteogan.train_epoch()

# save a final checkpoint
proteogan.save(log_dir+'/ckpt')

# generate some sequences based on test set labels
df = proteogan.generate(test)

# evaluate metrics
metrics = test.evaluate(df)
print()
print('MMD: ', metrics['global_distance'])
print('MRR: ', metrics['mean_reciprocal_rank'])
