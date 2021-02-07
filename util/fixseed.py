# -*- coding: utf-8 -*-
SEED = 0

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import tensorflow as tf
import random
import numpy as np

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
