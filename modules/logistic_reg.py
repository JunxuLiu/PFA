# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger

#from models.dnn_cifar10 import conv_net
from modules.models import Model

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = {'mnist':(28,28,1), 'fmnist':(28,28,1), 'cifar10':(32,32,3) }


class LogisticRegression(Model):

    def __init__(self, dataset, lr, lr_decay):
        self.dataset = dataset
        self.lr = lr
        self.lr_decay = lr_decay

        self.dpsgd = False

    
    def set_dpsgd_params(self, l2_norm_clip, num_microbatches, noise_multiplier):
        self.dpsgd = True
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches
        self.noise_multiplier = noise_multiplier


    def loss(self, logits, labels):
        labels = tf.cast(labels, dtype=tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')
            

    def evaluation(self, logits, labels):
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
    
    def init_placeholder(self):
        img_size = IMAGE_SIZE[self.dataset]
        img_pixels = img_size[0] * img_size[1] * img_size[2]
        
        self.data_placeholder = tf.placeholder(tf.float32, shape=(None,img_pixels), name='images_placeholder')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')
        self.labels_placeholder = tf.cast(labels_placeholder, dtype=tf.int64)

        return self.data_placeholder, self.labels_placeholder

    def __lr_mnist(self, features):
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        logits = tf.nn.softmax(tf.matmul(features, W) + b)

        return logits

    def build_model(self, features):
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            return self.__lr_mnist(features)
        else:
            raise ValueError('No model matches the required dataset.')
        '''
        # another architecture
        else:
            return self.__lr__xx(features)
        '''

    def train_model(self):

        # - global_step : A Variable, which tracks the amount of steps taken by the clients:
        global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

        # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
        if self.lr_decay:
            learning_rate = tf.train.exponential_decay(learning_rate=self.lr, 
                                                        global_step=global_step,
                                                        decay_steps=27000, 
                                                        decay_rate=0.1,
                                                        staircase=True, 
                                                        name='learning_rate')

            print('decay lr: start at {}'.format(self.lr))

        else:
            learning_rate = self.lr
            print('constant lr: {}'.format(self.lr))

        # Create the gradient descent optimizer with the given learning rate.
        if self.dpsgd:
            optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=learning_rate)

            train_op = optimizer.minimize(loss=self.vector_loss, global_step=global_step)

        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

            train_op = optimizer.minimize(loss=self.scalar_loss, global_step=global_step)
            
        return train_op


    def eval_model(self):

        # - logits : output of the [fully connected neural network] when fed with images.
        logits = self.build_model(self.data_placeholder)

        # - loss : when comparing logits to the true labels.
        # Calculate loss as a vector (to support microbatches in DP-SGD).
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits)

        # Define mean of loss across minibatch (for reporting through tf.Estimator).
        scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

        # - eval_correct : when run, returns the amount of labels that were predicted correctly.
        eval_op = self.evaluation(logits, self.labels_placeholder)

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', scalar_loss)

        self.vector_loss = vector_loss
        self.scalar_loss = scalar_loss

        return eval_op, vector_loss, scalar_loss
    