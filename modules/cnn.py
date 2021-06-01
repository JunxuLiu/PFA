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
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D

from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger

#from models.dnn_cifar10 import conv_net
from modules.models import Model

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = {'mnist':(28,28,1), 'fmnist':(28,28,1), 'cifar10':(32,32,3) }


class CNN(Model):

    def __init__(self, dataset, batch_size, lr, lr_decay):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay

        self.dpsgd = False

    
    def set_dpsgd_params(self, l2_norm_clip, num_microbatches, noise_multipliers):
        self.dpsgd = True
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches
        self.noise_multipliers = noise_multipliers


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


    def placeholder_inputs(self, batch_size, IMAGE_PIXELS):
        """Generate placeholder variables to represent the input tensors.
        These placeholders are used as inputs by the rest of the model building
        code and will be fed from the downloaded data in the .run() loop, below.
        Args:
        batch_size: The batch size will be baked into both placeholders.
        Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
        """
        # Note that the shapes of the placeholders match the shapes of the full
        # image and label tensors, except the first dimension is now batch_size
        # rather than the full size of the train or test data sets.
        images_placeholder = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS), name='images_placeholder')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')
        return images_placeholder, labels_placeholder

    
    def __cnn_mnist(self, features):
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

    def __cnn_cifar10(self, features):
        '''Given input features, returns the logits from a simple CNN model.'''
        input_layer = tf.reshape(features, [-1, 32, 32, 3])
        y = tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding='same', activation='relu').apply(input_layer)
        y = tf.keras.layers.MaxPool2D((2,2)).apply(y)
        y = tf.keras.layers.Dropout(0.25).apply(y)
        y = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=1, padding='same', activation='relu').apply(y)
        y = tf.keras.layers.MaxPool2D((2,2)).apply(y)
        y = tf.keras.layers.Dropout(0.25).apply(y)
        y = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=1, padding='same', activation='relu').apply(y)
        y = tf.keras.layers.MaxPool2D((2,2)).apply(y)
        y = tf.keras.layers.Dropout(0.25).apply(y)

        y = tf.keras.layers.Flatten().apply(y)
        y = tf.keras.layers.Dropout(0.25).apply(y)
        y = tf.keras.layers.Dense(1024, activation='relu').apply(y)
        y = tf.keras.layers.Dropout(0.25).apply(y)
        logits = tf.keras.layers.Dense(10, activation='softmax').apply(y)
        return logits
        
    
    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        '''
        帮助创建一个权重衰减的初始化变量
        
        请注意，变量是用截断的正态分布初始化的
        只有在指定了权重衰减时才会添加权重衰减
        
        Args:
        name: 变量的名称
        shape: 整数列表
        stddev: 截断高斯的标准差
        wd: 加L2Loss权重衰减乘以这个浮点数.如果没有，此变量不会添加权重衰减.
        
        Returns:
        变量张量
        '''
        var = self._variable_on_cpu(name, shape,
                            tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var


    def _variable_on_cpu(self, name, shape, initializer):
        '''
        帮助创建存储在CPU内存上的变量
        ARGS：
        name：变量的名称
        shape：整数列表
        initializer：变量的初始化操作
        返回：
        变量张量
        '''
        with tf.device('/cpu:0'): #用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _activation_summary(self, x):
        '''
        为激活创建summary
        
        添加一个激活直方图的summary
        添加一个测量激活稀疏度的summary
        
        ARGS：
        x：张量
        返回：
        没有
        '''
        # 如果这是多GPU训练，请从名称中删除'tower_ [0-9] /'.这有助于张量板上显示的清晰度.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

    def inference(self, images):
        """
        构建CIFAR-10模型
        ARGS：
        images：从distorted_inputs（）或inputs（）返回的图像
        返回：
        Logits
        """
        # 我们使用tf.get_variable（）而不是tf.Variable（）来实例化所有变量，以便跨多个GPU训练时能共享变量
        # 如果我们只在单个GPU上运行此模型，我们可以通过用tf.Variable（）替换tf.get_variable（）的所有实例来简化此功能
        
        # conv1-第一层卷积
        with tf.variable_scope('conv1') as scope: #每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
            # 5*5 的卷积核，64个
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                            stddev=1e-4, wd=0.0)
            # 卷积操作，步长为1，0padding SAME，不改变宽高，通道数变为64
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            # 在CPU上创建第一层卷积操作的偏置变量
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            # 加上偏置
            bias = tf.nn.bias_add(conv, biases)
            # relu非线性激活
            conv1 = tf.nn.relu(bias, name=scope.name)
            # 创建激活显示图的summary
            self._activation_summary(conv1)
            
        # pool1-第一层pooling
        # 3*3 最大池化，步长为2
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool1')
        # norm1-局部响应归一化
        # LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

        # conv2-第二层卷积
        with tf.variable_scope('conv2') as scope:
            # 卷积核：5*5 ,64个
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                            stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv2)

        # norm2-局部响应归一化
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
        # pool2-第二层最大池化
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3-全连接层，384个节点
        with tf.variable_scope('local3') as scope:
            # 把单个样本的特征拼成一个大的列向量，以便我们可以执行单个矩阵乘法
            dim = 1
            for d in pool2.get_shape()[1:].as_list():
                dim *= d
            reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
            
            # 权重
            weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                            stddev=0.04, wd=0.004)
            # 偏置
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            # relu激活
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            #生成summary
            _activation_summary(local3)

        # local4-全连接层，192个节点
        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                            stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            _activation_summary(local4)

        # softmax, i.e. softmax(WX + b)
        # 输出层
        with tf.variable_scope('softmax_linear') as scope:
            # 权重
            weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                            stddev=1/192.0, wd=0.0)
            # 偏置
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
            # 输出层的线性操作
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            # 生成summary
            _activation_summary(softmax_linear)
        return softmax_linear
        


    def build_model(self, features):

        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            return self.__cnn_mnist(features)

        elif self.dataset == 'cifar10':
            return self.__cnn_cifar10(features)

        else:
            raise ValueError('No model matches the required dataset.')


    def get_model(self, num_clients):
        # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
        img_size = IMAGE_SIZE[self.dataset]
        img_pixels = img_size[0] * img_size[1] * img_size[2]
        data_placeholder, labels_placeholder = self.placeholder_inputs(self.batch_size, img_pixels)

        # Define FCNN architecture
        # - logits : output of the [fully connected neural network] when fed with images.
        logits = self.build_model(data_placeholder)

        # - loss : when comparing logits to the true labels.
        # Calculate loss as a vector (to support microbatches in DP-SGD).
        labels_placeholder = tf.cast(labels_placeholder, dtype=tf.int64)
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=(logits))

        # Define mean of loss across minibatch (for reporting through tf.Estimator).
        scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

        # - eval_correct : when run, returns the amount of labels that were predicted correctly.
        eval_op = self.evaluation(logits, labels_placeholder)

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', scalar_loss)

        # - global_step : A Variable, which tracks the amount of steps taken by the clients:
        global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

        # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
        if self.lr_decay:
            learning_rate = tf.train.exponential_decay(learning_rate=self.lr, 
                                                        global_step=global_step,
                                                        decay_steps=5000, 
                                                        decay_rate=0.5,
                                                        staircase=True, 
                                                        name='learning_rate')
            print('decay lr: {}'.format(self.lr))

        else:
            learning_rate = self.lr
            print('constant lr: {}'.format(learning_rate))

        # Create the gradient descent optimizer with the given learning rate.
        if self.dpsgd:
            train_op_list = []
            for cid in range(num_clients):
                optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
                    l2_norm_clip=self.l2_norm_clip,
                    noise_multiplier=self.noise_multipliers[cid],
                    num_microbatches=self.num_microbatches,
                    learning_rate=learning_rate)
                opt_loss = vector_loss
                train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
                train_op_list.append(train_op)

        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
            opt_loss = scalar_loss
            train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
            train_op_list = [train_op] * num_clients

        return train_op_list, eval_op, scalar_loss, global_step, data_placeholder, labels_placeholder



