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
from scipy.stats import truncnorm

import numpy as np
import math

import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger

#from models.dnn_cifar10 import conv_net

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = {'mnist':(28,28,1), 'fmnist':(28,28,1), 'cifar10':(32,32,3) }

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.cast(labels, dtype=tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def init_placeholder(dataset):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      data_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    img_size = IMAGE_SIZE[dataset]
    img_pixels = img_size[0] * img_size[1] * img_size[2]
    
    data_placeholder = tf.placeholder(tf.float32, shape=(None,img_pixels), name='images_placeholder')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')

    return data_placeholder, labels_placeholder


def lr_mnist(x):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    logits = tf.nn.softmax(tf.matmul(x, W) + b)

    return logits


def cnn_mnist(features):
    """tensorflow/privacy"""
    """Given input features, returns the logits from a simple CNN model."""
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    y = tf.keras.layers.Conv2D(
      16, 8, strides=2, padding='same', activation='relu').apply(input_layer)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    y = tf.keras.layers.Conv2D(
      32, 4, strides=2, padding='valid', activation='relu').apply(y)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    y = tf.keras.layers.Flatten().apply(y)
    y = tf.keras.layers.Dense(32, activation='relu').apply(y)
    logits = tf.keras.layers.Dense(10).apply(y)
    return logits

def cnn_cifar10(features):
    """Given input features, returns the logits from a simple CNN model."""
    '''
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    y = tf.keras.layers.Conv2D(
      32, (3,3), strides=1, padding='same', activation='relu').apply(input_layer)
    y = tf.keras.layers.MaxPool2D(2, 2).apply(y)
    y = tf.keras.layers.Conv2D(
      64, (3,3), strides=1, padding='same', activation='relu').apply(y)
    y = tf.keras.layers.MaxPool2D(2, 2).apply(y)
    y = tf.keras.layers.Conv2D(
      64, (3,3), strides=2, padding='same', activation='relu').apply(y)
    y = tf.keras.layers.Flatten().apply(y)
    y = tf.keras.layers.Dense(64, activation='relu').apply(y)
    logits = tf.keras.layers.Dense(10).apply(y)
    '''
    #input_layer = tf.reshape(features, [-1, 32, 32, 3])
    #logits = conv_net(input_layer)
    #logits = ResNet20ForCIFAR10(input_shape=(32, 32, 3), classes=num_classes, weight_decay=weight_decay)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model

def train_model(hparams, dataset, model, opt_loss, 
                lr, lr_decay=False, noise_multiplier=None):

    # - global_step : A Variable, which tracks the amount of steps taken by the clients:
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
    if lr_decay:
      learning_rate = tf.train.exponential_decay(learning_rate=hparams.lr, 
                                                 global_step=global_step,
                                                 decay_steps=27000, decay_rate=0.1,
                                                 staircase=True, name='learning_rate')
      print('decay lr: start at {}'.format(lr))

    else:
      learning_rate = lr
      print('constant lr: {}'.format(learning_rate))

    '''
    ledger = privacy_ledger.PrivacyLedger(
          population_size=6000,
          selection_probability=(FLAGS.client_batch_size / 6000))
    '''
    # Create the gradient descent optimizer with the given learning rate.
    if noise_multiplier is None:
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

      #gradient_op = optimizer.compute_gradients(loss=opt_loss)
      #gradient_op_list = [gradient_op] * FLAGS.N
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    else:
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=hparams.l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=hparams.num_mbs,
        learning_rate=learning_rate)

      #var_list = tf.trainable_variables()
      #gradient_op = optimizer.compute_gradients(loss=opt_loss, var_list=None)
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    return train_op

def eval_model(hparams, dataset, model, 
          data_placeholder, labels_placeholder):

    # Define logistic regression and CNN architecture
    # - logits : output of the [fully connected neural network] when fed with images.
    if model == 'lr' and (dataset == 'mnist' or dataset == 'fmnist'):
      logits = lr_mnist(data_placeholder)
    elif model == 'cnn' and (dataset == 'mnist' or dataset == 'mnist'):
      logits = cnn_mnist(data_placeholder)
    else:
      raise ValueError('No model matches the required model and dataset.')

    # - loss : when comparing logits to the true labels.
    # Calculate loss as a vector (to support microbatches in DP-SGD).
    labels_placeholder = tf.cast(labels_placeholder, dtype=tf.int64)
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)

    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

    # - eval_correct : when run, returns the amount of labels that were predicted correctly.
    eval_op = evaluation(logits, labels_placeholder)

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', scalar_loss)

    return eval_op, vector_loss, scalar_loss


def cifar10_model(FLAGS, eps_list=None, noise_multiplier=None):

    classifier = cnn_cifar10(data_placeholder)

    classifier.fit(target_X_train,
           target_y_train,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=[target_X_valid, target_y_valid],
           verbose=1)


    # Create the gradient descent optimizer with the given learning rate.
    if FLAGS.dpsgd:
      #gradient_op_list = []
      train_op_list = []

      for i in range(FLAGS.N):
        optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=noise_multiplier[i],
          num_microbatches=FLAGS.num_microbatches,
          learning_rate=learning_rate)
        opt_loss = vector_loss

        #var_list = tf.trainable_variables()
        #gradient_op = optimizer.compute_gradients(loss=opt_loss, var_list=None)
        #gradient_op_list.append(gradient_op)
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        train_op_list.append(train_op)

    else:

      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

      opt_loss = scalar_loss
      #gradient_op = optimizer.compute_gradients(loss=opt_loss)
      #gradient_op_list = [gradient_op] * FLAGS.N
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
      train_op_list = [train_op] * FLAGS.N


