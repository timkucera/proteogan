# -*- coding: utf-8 -*-
from model.train import Trainable
from eval.eval import TrainTestVal

# load data splits
train, test, val = TrainTestVal('base L50', 300)

# load model
proteogan = Trainable(train, val, logdir='./test')

# load trained weights
proteogan.load('./weights/ckpt')

# generate some sequences based on test set labels
df = proteogan.generate(test)
print('Some example sequences:\n', df['sequence'].head(5))

# evaluate metrics
metrics = test.evaluate(df)
print()
print('MMD: ', metrics['global_distance'])
print('MRR: ', metrics['mean_reciprocal_rank'])
