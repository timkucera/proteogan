# -*- coding: utf-8 -*-

class GenerativeModel():
    '''
    Base class for a generative model to use with model.train.Trainable
    '''

    def __init__(self):
        self.step = 0

    def sample_z(self, batch_size, seed=None):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def evaluate(self, batch):
        raise NotImplementedError

    def plot(self, dir):
        raise NotImplementedError

    def get_loss_trajectory(self):
        raise NotImplementedError

    def save(self, dir):
        raise NotImplementedError

    def load(self, dir):
        raise NotImplementedError
