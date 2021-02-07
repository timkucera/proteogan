# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Conv1D, Concatenate, Dense, BatchNormalization, Input, ReLU, LeakyReLU, Softmax, Flatten, Dot, Add, Layer, Lambda, Conv2DTranspose, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
import numpy as np
from contextlib import redirect_stdout
from model.model import GenerativeModel
from model.constants import *


kernel_size = 12
strides = 8
ac_weight = 135

# Conv1DTranspose layer from: https://github.com/tensorflow/tensorflow/issues/6724
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


class ProteoGAN(GenerativeModel):
    '''
    Base class for ProteoGAN, which is based on a Wasserstein Generative Adversarial Network with Gradient Penalty (Gulrajani et al., 2017).
    '''
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator_optimizer = Adam(0.00041200571521910433,beta_1=0.0,beta_2=0.9)
        self.discriminator_optimizer = Adam(0.0004011676739902282,beta_1=0.0,beta_2=0.9)
        self.step = 0
        self.gen_loss = []
        self.gen_loss_buffer = []
        self.dis_loss = []
        self.dis_loss_buffer = []

    def build_generator(self):
        '''
        Builds the generator.
        '''
        z_input = Input(shape=(z_dim,))
        c_input = Input(shape=(label_dim,))
        x = Concatenate(axis=1)([z_input,c_input])
        x = Dense(units=seq_length*seq_dim, activation='relu')(x)
        x = BatchNormalization()(x)
        L = int(seq_length/(strides**2))
        f = int(seq_length*seq_dim/L)
        x = Reshape(target_shape=(L,f))(x)
        L = int(seq_length/(strides**1))
        f = int(seq_length*seq_dim/L)
        x = Conv1DTranspose(filters=f, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv1DTranspose(filters=seq_dim, kernel_size=kernel_size, strides=strides, padding='same')(x)
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
            x = Dense(label_dim)(x)
            c = c_input
            dot = Dot(axes=1)([x,c])
            x = Dense(1)(x)
            output = Add()([dot,x])
            projections.append(output)

        x_input = Input(shape=(seq_length, seq_dim))
        c_input = Input(shape=(label_dim,))
        L = int(seq_length/(strides**1))
        f = int(seq_length*seq_dim/L)
        x = Conv1D(filters=f, kernel_size=kernel_size, strides=strides, padding='same')(x_input)
        x = LeakyReLU(alpha=0.2)(x)
        project(x)
        x = Conv1D(filters=256, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        project(x)
        x = Flatten()(x)
        output_source = Add()(projections)
        output_labels = Dense(units=label_dim, activation='sigmoid')(x)
        return KerasModel([x_input, c_input],[output_source,output_labels])

    def discriminator_loss(self, real_output, fake_output, real_label_output, fake_label_output, real_labels, gradient, L=10):
        '''
        WGAN-GP loss.
        '''
        norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)
        gradient_penalty = (norm - 1.)**2
        w_loss = K.mean(fake_output) - K.mean(real_output) + K.mean(gradient_penalty)
        ac_loss = ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, real_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    def generator_loss(self, fake_output, fake_label_output, real_labels):
        '''
        WGAN-GP loss.
        '''
        w_loss = K.mean(-fake_output)
        ac_loss = ac_weight * K.mean(tf.keras.losses.binary_crossentropy(real_labels, fake_label_output, from_logits=False))
        total_loss = w_loss + ac_loss
        return total_loss, w_loss, ac_loss

    def sample_z(self, batch_size, seed=None):
        '''
        Generates a latent noise vector of <batch_size> instances. Optionally with fixed <seed> for reproducibility.
        '''
        return tf.random.normal((batch_size, z_dim), seed=seed)

    def train_step(self, batch):
        '''
        A single training step of the GAN. Input is a <batch> with embedded sequences and labels, generated from the data.dataset.Dataset class.
        '''
        self.step += 1
        disc_loss = self.discriminator_train_step(batch)
        self.dis_loss_buffer.append(np.array([l.numpy() for l in disc_loss]))
        gen_loss = self.generator_train_step(batch)
        self.gen_loss_buffer.append(np.array([l.numpy() for l in gen_loss]))
        if self.step % self.plot_interval == 0: # average over the plot interval
            gen_avg_loss = np.array(list(zip(np.array([self.step]*len(self.gen_loss_buffer[0])), np.mean(self.gen_loss_buffer,0))))
            self.gen_loss.append(gen_avg_loss)
            self.gen_loss_buffer = []
            dis_avg_loss = np.array(list(zip(np.array([self.step]*len(self.dis_loss_buffer[0])), np.mean(self.dis_loss_buffer,0))))
            self.dis_loss.append(dis_avg_loss)
            self.dis_loss_buffer = []

    @tf.function
    def evaluate(self, batch):
        '''
        Generate a sample with a given <batch> of latent variables and conditioning labels.
        '''
        z = batch[0]
        c = batch[1]
        return self.generator([z,c], training=False)

    @tf.function
    def classify(self, batch, c):
        '''
        Classifiy a <batch> of sequences and labels with the auxiliary classifier.
        '''
        output, label_output = self.discriminator([batch,c], training=False)
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
            generated_data = self.generator([z,c], training=True)
            fake_output, fake_label_output = self.discriminator([generated_data,c], training=True)
            gen_loss, w_loss, ac_loss  = self.generator_loss(fake_output, fake_label_output, c)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return (gen_loss, w_loss, ac_loss)

    @tf.function
    def discriminator_train_step(self, batch):
        '''
        A discriminator train step. Returns losses.
        '''
        c = batch[1]
        data = tf.one_hot(batch[0], depth=seq_dim, axis=-1, on_value=1, off_value=0)
        data = tf.cast(data, dtype=tf.dtypes.float32)
        data_size = data.shape[0]
        e_shape = (data_size,)
        for i in data.shape[1:]:
            e_shape = e_shape + (1,)
        z = self.sample_z(data_size)
        with tf.GradientTape() as disc_tape:
            generated_data = self.generator([z,c], training=True)
            real_output, real_label_output = self.discriminator([data,c], training=True)
            fake_output, fake_label_output = self.discriminator([generated_data,c], training=True)
            epsilon = K.random_uniform(e_shape, dtype=tf.dtypes.float32)
            random_weighted_average = (epsilon * data) + ((1 - epsilon) * generated_data)
            # calculate gradient for penalty
            with tf.GradientTape() as norm_tape:
                norm_tape.watch(random_weighted_average)
                average_output = self.discriminator([random_weighted_average,c], training=True)
            gradient = norm_tape.gradient(average_output, random_weighted_average)
            disc_loss, w_loss, ac_loss = self.discriminator_loss(real_output, fake_output, real_label_output, fake_label_output, c, gradient, L=10)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return (disc_loss, w_loss, ac_loss)

    def plot(self, dir):
        '''
        Plot a model overview and a text description in <dir>.
        '''
        plot_model(self.generator, to_file=dir+'/gen.png')
        plot_model(self.discriminator, to_file=dir+'/dis.png')
        with open(dir+'/gen_model.txt','w') as file:
            with redirect_stdout(file):
                self.generator.summary()
        with open(dir+'/dis_model.txt','w') as file:
            with redirect_stdout(file):
                self.discriminator.summary()

    def get_loss_trajectory(self):
        '''
        Convert losses into a plottable format.
        '''
        gen_loss = np.array(self.gen_loss).transpose(1,0,2)
        dis_loss = np.array(self.dis_loss).transpose(1,0,2)
        gen, gen_w, gen_ac = gen_loss
        dis, dis_w, dis_ac = dis_loss
        return [
            {'value':gen[:,1],'step':gen[:,0],'color':'#4caf50','label':'Generator Loss'},
            {'value':gen_ac[:,1],'step':gen_ac[:,0],'color':'#c5e1a5','label':'Generator-AC Loss'},
            {'value':gen_w[:,1],'step':gen_w[:,0],'color':'#E6EE9C','label':'Generator-W Loss'},
            {'value':dis[:,1],'step':dis[:,0],'color':'#2196f3','label':'Discriminator Loss'},
            {'value':dis_ac[:,1],'step':dis_ac[:,0],'color':'#4fc3f7','label':'Discriminator-AC Loss'},
            {'value':dis_w[:,1],'step':dis_w[:,0],'color':'#b2ebf2','label':'Discriminator-W Loss'}
        ]

    def save(self, dir):
        '''
        Save a model checkpoint.
        '''
        self.generator.save_weights(dir+'/gen')
        self.discriminator.save_weights(dir+'/dis')
        np.save(dir+'/gen_loss.npy', np.array(self.gen_loss))
        np.save(dir+'/dis_loss.npy', np.array(self.dis_loss))

    def load(self, dir):
        '''
        Restore a model checkpoint.
        '''
        self.generator.load_weights(dir+'/gen')
        self.discriminator.load_weights(dir+'/dis')
        self.gen_loss = list(np.load(dir+'/gen_loss.npy'))[:int(self.step/self.plot_interval)]
        self.dis_loss = list(np.load(dir+'/dis_loss.npy'))[:int(self.step/self.plot_interval)]
