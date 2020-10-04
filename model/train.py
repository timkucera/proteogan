# -*- coding: utf-8 -*-
from util import fixseed
import os, sys, time, pickle, gc, shutil, glob, json, math
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import seaborn as sns
import numpy as np
import pandas as pd
from data.dataset import Dataset, BaseDataset
import matplotlib.pyplot as plt
import matplotlib
from util.constants import amino_acid_alphabet
from eval.eval import TrainTestVal, TestingSet
from matplotlib.lines import Line2D
from model.gan import ProteoGAN
from model.constants import *


class Trainable():
    '''
    Trainable class that implements all routines for training, such as
    train steps, checkpointing and plotting.
    '''

    def __init__(self, train, val, logdir='.'):
        self.batch_size = 128
        self.shuffle_buffer_size = 5000
        self.start_time = time.time()
        self.logdir = logdir
        if not os.path.exists(logdir):
            try:
                os.mkdir(logdir)
            except:
                print('Could not find nor create log directory {}. Please make sure it exists and try again.'.format(logdir))
                raise

        # data for training and duality gap
        train.pad_to(seq_length).tokenize_sequences().embed_labels()
        self.tokenizer = train.tokenizer
        self.godag = train._godag
        self.labelBinarizer = train.labelBinarizer
        self.names = train.names
        data_train, data_adv, data_adv_test = np.split(train.df, [int(.98 * len(train.df)), int(.99 * len(train.df))])
        self.data = BaseDataset().from_df(data_train).tf()
        self.adversary_finding_set = BaseDataset().from_df(data_adv).tf().batch(self.batch_size)
        self.adversary_testing_set = BaseDataset().from_df(data_adv_test).tf().batch(self.batch_size)
        self.valset = val
        self.datasetSize = len(data_train.index)

        # plot related (for losses)
        self.plot_curves = {'generator': [], 'discriminator':[], 'dualitygap':[], 'mmd': [], 'mrr':[]}
        self.iter_per_epoch = math.ceil(self.datasetSize/batch_size)
        self.plot_interval = int(self.iter_per_epoch/plot_interval)

        # specify the model
        self._model = ProteoGAN # the model class is also used to instantiate the duality gap models. Change here if you'd like to implement your own model.
        self.model = self._model()
        self.model.plot(self.logdir)
        self.model.plot_interval = self.plot_interval

        # define a fixed sample to generate for evaluations during training, to avoid confoundings with the random variable
        sample_to_generate = BaseDataset().from_df(self.valset.df.copy(), godag=self.godag)
        sample_to_generate.names = self.names
        sample_to_generate = sample_to_generate.embed_labels(binarizer=self.labelBinarizer)
        self.embedded_labels_to_generate = np.stack(sample_to_generate.df['labels_onehot'].to_numpy())
        self.labels_to_generate = sample_to_generate.df['labels'].to_list()
        self.latents_to_generate = self.model.sample_z(len(sample_to_generate.df.index))

        # checkpoints for best model and snapshots in duality gap
        self.best = None
        self.best_dir = self.logdir+'/best'
        if not os.path.exists(self.best_dir): os.mkdir(self.best_dir)
        self.snapshots = []
        self.snapshot_path = self.logdir+'/snapshots'
        if not os.path.exists(self.snapshot_path): os.mkdir(self.snapshot_path)
        self.max_models_for_dualitygap = 10
        self.plot_curves['dualitygap'] = []


    def train_epoch(self):
        '''
        Defines the loop for training an epoch.
        '''
        metrics = None
        data = self.data.shuffle(self.shuffle_buffer_size).batch(self.batch_size)
        for batch in data:
            self.model.train_step(batch)
            if self.model.step % self.plot_interval  == 0:
                generated = self.generate()
                generated.to_csv(self.logdir+'/generated.csv')
                metrics = self.valset.evaluate(generated)
                MMD = metrics['global_distance']
                MRR = metrics['mean_reciprocal_rank']
                with open(self.logdir+'/result.json','w') as file:
                    json.dump({'ratio':MRR/MMD, 'mmd':MMD, 'mrr':MRR},file)
                if self.best is None or MRR/MMD > self.best:
                    self.best = MRR/MMD
                    self.model.save(self.best_dir)
                    with open(self.best_dir+'/result.json','w') as file:
                        json.dump({'ratio':MRR/MMD, 'mmd':MMD, 'mrr':MRR, 'iteration': self.model.step},file)
                self.plot_curves['dualitygap'].append([self.model.step, self.dualityGap(self.model.step)])
                self.plot_curves['mmd'].append(np.array([self.model.step,MMD]))
                self.plot_curves['mrr'].append(np.array([self.model.step,MRR]))
                self.plot()
        return metrics

    def save(self, checkpoint_dir):
        '''
        Saves a checkpoint of the model and losses/evaluations into <checkpoint_dir>.
        '''
        if not os.path.exists(checkpoint_dir):
            try:
                os.mkdir(checkpoint_dir)
            except:
                print('Could not create directory {}. Please make sure it exists and try again.'.format(checkpoint_dir))
                raise
        self.model.save(checkpoint_dir)
        with open(checkpoint_dir+'/state.pkl','wb') as file:
            state = {'step': self.model.step, 'time':(time.time() - self.start_time)}
            pickle.dump(state, file)
        np.save(checkpoint_dir+'/mrr.npy', np.array(self.plot_curves['mrr']))
        np.save(checkpoint_dir+'/mmd.npy', np.array(self.plot_curves['mmd']))
        np.save(checkpoint_dir+'/dualitygap.npy', np.array(self.plot_curves['dualitygap']))
        # save a generated sample
        df = self.generate()
        df['id'] = ['generated_{}'.format(i+1) for i in range(len(df.index))]
        with open(self.logdir+'/generated.fasta','w') as file:
            for index,row in df.iterrows():
                file.write('>'+row['id']+' '+' '.join(row['labels'])+'\n'+row['sequence']+'\n')
        df.to_csv(self.logdir+'/generated.csv', index=False)
        #self.classify()

    def load(self, checkpoint_dir):
        '''
        Loads a model checkpoint and losses/evaluations from <checkpoint_dir>.
        '''
        if os.path.exists(checkpoint_dir+'/state.pkl'):
            with open(checkpoint_dir+'/state.pkl','rb') as file:
                state = pickle.load(file)
                self.model.step = state['step']
                self.start_time = time.time() - state['time']
                self.plot_curves['mmd'] = list(np.load(checkpoint_dir+'/mmd.npy'))[:int(self.model.step/self.plot_interval )]
                self.plot_curves['mrr'] = list(np.load(checkpoint_dir+'/mrr.npy'))[:int(self.model.step/self.plot_interval )]
                if callable(getattr(self, "dualityGap", None)): self.plot_curves['dualitygap'] = list(np.load(checkpoint_dir+'/dualitygap.npy'))[:int(self.model.step/self.plot_interval )]
        if os.path.exists(self.best_dir+'/result.json'):
            with open(self.best_dir+'/result.json','r') as file:
                self.best = json.load(file)['ratio']
        self.model.load(checkpoint_dir)

    def generate(self, dataset=None, latent_seed=None):
        '''
        Generates a sample given a <dataset> with labels. If not given,
        the evaluation set will be used. Optionally with fixed <latent_seed>.
        '''
        if dataset is None:
            labels = self.embedded_labels_to_generate
            latents = self.latents_to_generate
            original_labels = self.labels_to_generate
        else:
            if isinstance(dataset, pd.DataFrame):
                sample_to_generate = BaseDataset().from_df(dataset.copy(), godag=self.godag)
            elif isinstance(dataset, BaseDataset):
                sample_to_generate = dataset.copy()
            elif isinstance(dataset, TestingSet):
                sample_to_generate = BaseDataset().from_df(dataset.df.copy(), godag=self.godag)
            sample_to_generate = sample_to_generate.embed_labels(binarizer=self.labelBinarizer)
            labels = np.stack(sample_to_generate.df['labels_onehot'].to_numpy())
            original_labels = sample_to_generate.df['labels'].to_list()
            latents = self.model.sample_z(labels.shape[0], seed=latent_seed)

        ds_size = math.ceil(len(labels)/self.batch_size)
        labels = tf.data.Dataset.from_tensor_slices(labels)
        latents = tf.data.Dataset.from_tensor_slices(latents)
        ds = tf.data.Dataset.zip((latents, labels)).batch(self.batch_size)
        gen = []
        for batch in ds:
            _result = self.model.evaluate(batch)
            result = [r.numpy() for r in _result]
            result = self.tokenizer.sequences_to_texts(list(np.argmax(result, axis=-1)))
            gen.extend(result)
        gen = [s.replace(' ', '') for s in gen]
        gen = [s.replace('-', '') for s in gen]
        df = pd.DataFrame({'sequence': gen, 'labels': original_labels})
        return df

    def plot(self):
        '''
        Plots losses and evaluations.
        '''
        losses = self.model.get_loss_trajectory()
        dg = np.array(self.plot_curves['dualitygap'])
        if len(losses[0]['step']) > 0:
            with sns.axes_style("whitegrid"):
                plt.figure()
                for loss in losses:
                    ax = sns.lineplot(x=loss['step'], y=loss['value'], label=loss['label'], legend=False, color=loss['color'])
                ax = sns.lineplot(x=dg[:,0].ravel(), y=dg[:,1].ravel(), label='Duality Gap', legend=False, color='#e91e63')
                ax = plt.gca()
                if len(self.plot_curves['mmd']) > 0:
                    mrr = np.array(self.plot_curves['mrr'])
                    mmd = np.array(self.plot_curves['mmd'])
                    ax2 = ax.twinx()
                    ax2.grid(None)
                    ax_mmd = sns.lineplot(x=mmd[:,0].ravel(), y=mmd[:,1].ravel(), label='MMD', legend=False, color='#ffa500', ax=ax2)
                    ax_mrr = sns.lineplot(x=mrr[:,0].ravel(), y=mrr[:,1].ravel(), label='MRR', legend=False, color='#FFD700', ax=ax2)
                colors = [loss['color'] for loss in losses]+['#e91e63','#ffa500','#FFD700']
                names = [loss['label'] for loss in losses]+['Duality Gap', 'MMD','MRR']
                ncol = 2 if len(names) == 4 else 3
                bbox_top = 0.85 + ((len(losses)+2)//ncol)*0.07
                lines = [Line2D([0], [0], color=color, lw=2) for color in colors]
                ax.legend(lines, names, bbox_to_anchor=(0,bbox_top,1,0.2), loc="upper left", mode="expand", borderaxespad=0, ncol=ncol, frameon=False)
                ax.set_xlabel('Parameter Update', fontdict = {'size': 12})
                axes = plt.gca()
                plt.tight_layout()
                plt.savefig(self.logdir+'/loss.png', dpi=300)
                plt.close()

    def classify(self, ds=None):
        '''
        Classifies a data.dataset.BaseDataset with the auxiliary classifier. If not given, it classifies the validation set.
        '''
        if ds is None:
            ds = BaseDataset().from_df(self.valset.df, godag=self.godag)
        ds.names = self.names
        ids = ds.df['id'].to_list()
        ds_size = math.ceil(len(ds.df.index)/self.batch_size)
        ds = ds.pad_to(seq_length).tokenize_sequences().embed_labels(binarizer=self.labelBinarizer)
        ds = ds.tf().batch(self.batch_size)
        df = []
        idx = 0
        go_terms = np.ones((1,label_dim))
        go_terms = list(self.labelBinarizer.inverse_transform(go_terms)[0])
        for batch in ds:
            data = tf.one_hot(batch[0], depth=seq_dim, axis=-1)
            c = batch[1]
            _result = self.model.classify(data,c)
            result = np.array([r.numpy() for r in _result])
            for i,labels in enumerate(result):
                for j, score in enumerate(labels):
                    df.append({'id':ids[idx], 'score':score, 'label':go_terms[j]})
                idx += 1
        df = pd.DataFrame(df)
        df.to_csv(self.logdir+'/classified.csv', index=False)
        return df

    def dualityGap(self, step):
        '''
        Estimates the duality gap (Grnarova et al., ‎2019) based on past snapshots.
        '''
        # make snapshot
        self.model.generator.save_weights(self.snapshot_path+'/snapshot_gen_{}'.format(step))
        self.model.discriminator.save_weights(self.snapshot_path+'/snapshot_dis_{}'.format(step))
        self.snapshots.append(step)
        Mu = np.inf
        Mv = -np.inf
        u_worst = self.model.generator
        v_worst = self.model.discriminator
        for snap in self.snapshots[-self.max_models_for_dualitygap:]:
            # build models
            model =self._model()
            model.generator.load_weights(self.snapshot_path+'/snapshot_gen_{}'.format(snap))
            model.discriminator.load_weights(self.snapshot_path+'/snapshot_dis_{}'.format(snap))
            # evaluate against current model
            ## v worst
            M = self.adversarial_loss(
                generator = self.model.generator,
                discriminator = model.discriminator,
                dataset = self.adversary_finding_set
            )
            ## choose worst v
            if M > Mv:
                Mv = M
                v_worst = model.discriminator
            ## u worst
            M = self.adversarial_loss(
                generator = model.generator,
                discriminator = self.model.discriminator,
                dataset = self.adversary_finding_set
            )
            ## choose worst u
            if M < Mu:
                Mu = M
                u_worst = model.generator
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        # calculate DG
        Mu = self.adversarial_loss(
            generator = u_worst,
            discriminator = self.model.discriminator,
            dataset = self.adversary_testing_set
        )
        Mv = self.adversarial_loss(
            generator = self.model.generator,
            discriminator = v_worst,
            dataset = self.adversary_testing_set
        )
        # delete not needed snapshots
        self.deleteSnapshots(snaps=self.snapshots[:-self.max_models_for_dualitygap])
        return Mv - Mu

    #@tf.function
    def adversarial_loss(self, generator, discriminator, dataset):
        '''
        Adversarial loss for duality gap.
        '''
        M = []
        for batch in dataset:
            c = batch[1]
            data = tf.one_hot(batch[0], depth=seq_dim, axis=-1, on_value=1, off_value=0)
            data = tf.cast(data, dtype=tf.dtypes.float32)
            data_size = data.shape[0]
            z = self.model.sample_z(data.shape[0])
            generated_data = generator([z,c], training=False)
            real_output,_ = discriminator([data,c], training=False)
            fake_output,_ = discriminator([generated_data,c], training=False)
            M.append(np.mean(real_output)-np.mean(fake_output))
        return np.mean(M)


    def deleteSnapshots(self, snaps=False):
        '''
        Delete snapshots not used anymore by duality gap.
        '''
        if snaps == False:
            shutil.rmtree(self.snapshot_path)
        else:
            for snap in snaps:
                for path in glob.glob(self.snapshot_path+'/snapshot_*_{}.*'.format(snap)):
                    os.remove(path)
