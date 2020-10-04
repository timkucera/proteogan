# -*- coding: utf-8 -*-
'''
Parameters for protein design as in the paper. Hyperparameters are set as in the final model ProteoGAN.
'''

z_dim = 100
label_dim = 50 # number of classes
seq_dim = 21 # length of amino acid alphabet + 1 (for padding)
seq_length = 2048
batch_size = 128
plot_interval = 2 # times per epoch
