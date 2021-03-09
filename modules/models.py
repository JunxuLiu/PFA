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

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = {'mnist':(28,28,1), 'fmnist':(28,28,1), 'cifar10':(32,32,3) }


class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluation():
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

        Returns:
            A scalar int32 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
        """
    
    @abc.abstractmethod
    def init_placeholder():
        """Generate placeholder variables to represent the input tensors.
        These placeholders are used as inputs by the rest of the model building
        code and will be fed from the downloaded data in the .run() loop, below.

        Args:
            batch_size: The batch size will be baked into both placeholders.
        Returns:
            data_placeholder: Images placeholder.
            labels_placeholder: Labels placeholder.
        """

    @abc.abstractmethod
    def build_model():
         """Given input features, returns the logits from a ML model."""

    @abc.abstractmethod
    def train_model():
        pass

    @abc.abstractmethod
    def eval_model():
        pass



