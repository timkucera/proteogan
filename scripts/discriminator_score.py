"""
Script to score protein sequences with a trained discriminator/critic ('source' output). In the Wasserstein GAN this relates to the distance of generated samples to real samples.

This is a follow up to the discussion in https://github.com/timkucera/proteogan/issues/1 to see if these values correlate with stability. Please note: THIS IS NOT PART OF THE PAPER NOR HAS IT BEEN VERIFIED IN ANY WAY. It's just a wild guess.

Parameters:
--in_csv    A csv file of sequences and labels. See https://github.com/timkucera/proteogan/blob/master/data/split_1/test.csv for an example of the formatting.
--out_csv   The path to store the output csv file, with discriminator scores added as a new column.
--weights   The path to the weights of the trained model.
"""

import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from models.proteogan import ProteoGAN
from models.config import proteogan_config
from data.util import tokenize, embed_labels, save, load
from train.util import batch_dont_shuffle

parser = argparse.ArgumentParser(description='Score sequences with discriminator')
parser.add_argument('--in_csv', type=str, help='Input csv file', default='data/split_1/test.csv')
parser.add_argument('--out_csv', type=str, help='Output csv file', default='disc_scores.csv')
parser.add_argument('--weights', type=str, help='Path to trained model weights', default='train/split_1/proteogan/best')
args = parser.parse_args()

df = load(path=args.in_csv)
data = tokenize(df, pad_sequence_to=2048)
data = batch_dont_shuffle(data, proteogan_config['batch_size'])

model = ProteoGAN(proteogan_config)
model.generator.load_weights(args.weights+'/gen')
model.discriminator.load_weights(args.weights+'/dis')

disc_output = []
for batch in tqdm(data):
    seq, labels = batch
    seq = tf.one_hot(tf.cast(seq, tf.int32), depth=proteogan_config['seq_dim'], axis=-1, dtype=tf.dtypes.float32)
    labels = np.array([embed_labels(tokens) for tokens in labels])
    src, _ = model.discriminator([seq,labels], training=False)
    disc_output.extend(src.numpy().squeeze())

df['discriminator_score'] = disc_output
save(df, args.out_csv)
