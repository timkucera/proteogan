import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from train.util import BaseTrainer
from data.util import embed_labels
from util.constants import amino_acid_alphabet
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Conv1D, Concatenate, Dense, BatchNormalization, Input, ReLU, LeakyReLU, Softmax, Flatten, Dot, Add, Layer, Lambda, Conv2DTranspose, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
import numpy as np
from contextlib import redirect_stdout

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


# Conv1DTranspose layer from: https://github.com/tensorflow/tensorflow/issues/6724 by Ryan Peach
class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)


class ProteoGAN():
    '''
    Base class for ProteoGAN, which is based on a Wasserstein Generative Adversarial Network with Gradient Penalty (Gulrajani et al., 2017).
    '''
    def __init__(self, config):

        self.config = config
        self.seq_length = self.config['seq_length']
        self.seq_dim = self.config['seq_dim']
        self.label_dim = self.config['label_dim']
        self.z_dim = self.config['z_dim']
        self.strides = self.config['strides']
        self.kernel_size = self.config['kernel_size']

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.step = 0
        self.gen_loss = []
        self.gen_loss_buffer = []
        self.dis_loss = []
        self.dis_loss_buffer = []

    def build_generator(self):
        '''
        Builds the generator.
        '''

        z_input = Input(shape=(self.z_dim,))
        c_input = Input(shape=(self.label_dim,))
        x = Concatenate(axis=1)([z_input,c_input])
        x = Dense(units=self.seq_length*self.seq_dim, activation='relu')(x)
        x = BatchNormalization()(x)
        L = int(self.seq_length/(self.strides**2))
        f = int(self.seq_length*self.seq_dim/L)
        x = Reshape(target_shape=(L,f))(x)
        L = int(self.seq_length/(self.strides**1))
        f = int(self.seq_length*self.seq_dim/L)
        x = Conv1DTranspose(filters=f, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv1DTranspose(filters=self.seq_dim, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = ReLU()(x)
        x = Softmax(axis=-1)(x)
        output = x
        return KerasModel([z_input, c_input],[output])

    def build_discriminator(self):
        '''
        Builds the discriminator.
        '''

        projections = []
        def project(x):
            x = Flatten()(x)
            x = Dense(self.label_dim)(x)
            c = c_input
            dot = Dot(axes=1)([x,c])
            x = Dense(1)(x)
            output = Add()([dot,x])
            projections.append(output)

        x_input = Input(shape=(self.seq_length, self.seq_dim))
        c_input = Input(shape=(self.label_dim,))
        L = int(self.seq_length/(self.strides**1))
        f = int(self.seq_length*self.seq_dim/L)
        x = Conv1D(filters=f, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x_input)
        x = LeakyReLU(alpha=0.2)(x)
        project(x)
        x = Conv1D(filters=256, kernel_size=self.kernel_size, strides=self.strides, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        project(x)
        x = Flatten()(x)
        output_source = Add()(projections)
        output_labels = Dense(units=self.label_dim, activation='sigmoid')(x)
        return KerasModel([x_input, c_input],[output_source,output_labels])


class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = kwargs['config']
        self.seq_length = self.config['seq_length']
        self.seq_dim = self.config['seq_dim']
        self.label_dim = self.config['label_dim']
        self.z_dim = self.config['z_dim']
        self.strides = self.config['strides']
        self.kernel_size = self.config['kernel_size']
        self.ac_weight = self.config['ac_weight']

        self.model = ProteoGAN(self.config)

    def train_batch(self, batch):
        '''
        Train one batch on generator and discriminator.
        '''
        seq, labels = batch
        labels = np.array([embed_labels(tokens) for tokens in labels])
        z = self.sample_z(labels.shape[0])
        dis_losses = self.discriminator_train_step([seq, labels])
        gen_losses = self.generator_train_step([z, labels])
        losses = {**gen_losses, **dis_losses}
        losses = {k:float(v.numpy()) for k,v in losses.items()}
        return losses

    def create_optimizer(self):
        '''
        Creates the optimizer.
        '''
        self.generator_optimizer = Adam(4e-4, beta_1=0.0, beta_2=0.9)
        self.discriminator_optimizer = Adam(4e-4, beta_1=0.0, beta_2=0.9)

    def predict_batch(self, batch, seed=None):
        '''
        Generate a sample with a given <batch> of latent variables and conditioning labels.
        '''
        seq, labels = batch
        labels = np.array([embed_labels(tokens) for tokens in labels])
        z = self.sample_z(labels.shape[0], seed=seed)
        output = self._predict_batch([z, labels])
        return np.array(tf.argmax(output, axis=-1)).astype(int)

    @tf.function
    def _predict_batch(self, batch):
        return self.model.generator(batch, training=False)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, real_label_output, fake_label_output, real_labels, gradient, L=10):
        '''
        WGAN-GP loss.
        '''
        norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)
        gradient_penalty = (norm - 1.)**2
        w_loss = K.mean(fake_output) - K.mean(real_output) + K.mean(gradient_penalty)
        ac_loss = self.ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, real_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    @tf.function
    def generator_loss(self, fake_output, fake_label_output, real_labels):
        '''
        WGAN-GP loss.
        '''
        w_loss = K.mean(-fake_output)
        ac_loss = self.ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, fake_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    @tf.function
    def sample_z(self, batch_size, seed=None):
        '''
        Generates a latent noise vector of <batch_size> instances. Optionally with fixed <seed> for reproducibility.
        '''
        return tf.random.normal((batch_size, self.z_dim), seed=seed)

    #@tf.function
    def classify(self, batch, c):
        '''
        Classifiy a <batch> of sequences and labels with the auxiliary classifier.
        '''
        output, label_output = self.model.discriminator([batch,c], training=False)
        return label_output

    @tf.function
    def generator_train_step(self, batch):
        '''
        A generator train step. Returns losses.
        '''
        c = batch[1]
        data_size = c.shape[0]
        z = self.sample_z(data_size)
        with tf.GradientTape() as gen_tape:
            generated_data = self.model.generator([z,c], training=True)
            fake_output, fake_label_output = self.model.discriminator([generated_data,c], training=True)
            gen_loss, w_loss, ac_loss  = self.generator_loss(fake_output, fake_label_output, c)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator.trainable_variables))
        return {'Total Loss (G)': gen_loss, 'W. Loss (G)': w_loss, 'AC Loss (G)': ac_loss}

    @tf.function
    def discriminator_train_step(self, batch):
        '''
        A discriminator train step. Returns losses.
        '''
        c = batch[1]
        data = tf.one_hot(tf.cast(batch[0], tf.int32), depth=self.seq_dim, axis=-1, dtype=tf.dtypes.float32)
        data_size = data.shape[0]
        e_shape = (data_size,)
        for i in data.shape[1:]:
            e_shape = e_shape + (1,)
        z = self.sample_z(data_size)
        with tf.GradientTape() as disc_tape:
            generated_data = self.model.generator([z,c], training=True)
            real_output, real_label_output = self.model.discriminator([data,c], training=True)
            fake_output, fake_label_output = self.model.discriminator([generated_data,c], training=True)
            epsilon = K.random_uniform(e_shape, dtype=tf.dtypes.float32)
            random_weighted_average = (epsilon * data) + ((1 - epsilon) * generated_data)
            # calculate gradient for penalty
            with tf.GradientTape() as norm_tape:
                norm_tape.watch(random_weighted_average)
                average_output = self.model.discriminator([random_weighted_average,c], training=True)
            gradient = norm_tape.gradient(average_output, random_weighted_average)
            disc_loss, w_loss, ac_loss = self.discriminator_loss(real_output, fake_output, real_label_output, fake_label_output, c, gradient, L=10)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.discriminator.trainable_variables))
        return {'Total Loss (D)': disc_loss, 'W. Loss (D)': w_loss, 'AC Loss (D)': ac_loss}

    def store_model(self, path):
        '''
        Save a model checkpoint.
        '''
        self.model.generator.save_weights(path+'gen')
        self.model.discriminator.save_weights(path+'dis')

    def restore_model(self, path):
        '''
        Restore a model checkpoint.
        '''
        self.model.generator.load_weights(path+'gen')
        self.model.discriminator.load_weights(path+'dis')
