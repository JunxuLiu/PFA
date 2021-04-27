"""
client update
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import pickle
import math
import copy
import tensorflow.compat.v1 as tf
import numpy as np
import scipy
import time

from modules.budgets_accountant import BudgetsAccountant

from common_utils import main_utils
from common_utils.tf_utils import Vname_to_FeedPname, Vname_to_Pname
np.random.seed(10)


class Client(object):

    def __init__(self, x_train, y_train, batch_size, loc_steps):
        self.x_train = x_train
        self.y_train = y_train
        self.dataset_size = len(x_train)
        self.batch_size = batch_size
        self.loc_steps = loc_steps
        
        self.ba = None
        self.Vk = None
        self.mean = None
        self.global_steps = 0

    def set_ba(self, ba):
        '''set client's budget accountant'''
        self.ba = ba

    def set_ops(self, train_op, eval_op, scalar_loss,
                data_placeholder, labels_placeholder):

        self.train_op = train_op
        self.eval_op = eval_op
        self.scalar_loss = scalar_loss
        self.data_ph = data_placeholder
        self.labels_ph = labels_placeholder
        
    def precheck(self):
        if self.ba is None:
            return True
        else:
            return self.ba.precheck(self.dataset_size, self.batch_size, self.loc_steps)

    def download_model(self, sess, assignments, set_global_step, model):

        sess.run(assignments, feed_dict=model)
        sess.run(set_global_step, feed_dict={'global_step_placeholder:0':self.global_steps})
        
    def set_projection(self, Vk=None, mean=None, is_private=False):

        self.Vk = Vk
        self.mean = mean
        self.is_private = is_private

    def local_update(self, sess, model, global_steps):

        # local SGD then get the model updates
        for it in range(self.loc_steps):
        
            # batch_ind holds the indices of the current batch
            batch_ind = np.random.permutation(self.dataset_size)[0:self.batch_size]
            x_batch = self.x_train[[int(j) for j in batch_ind]]
            y_batch = self.y_train[[int(j) for j in batch_ind]]

            # Fill a feed dictionary with the actual set of data and labels using the data and labels associated
            # to the indices stored in batch_ind:
            feed_dict = {str(self.data_ph.name): x_batch,
                         str(self.labels_ph.name): y_batch}

            # Run one optimization step.
            _ = sess.run(self.train_op, feed_dict = feed_dict)

        self.global_steps = sess.run(global_steps)
        
        updates = [model[Vname_to_FeedPname(var)] - sess.run(var) for var in tf.trainable_variables()]

        if (self.Vk is not None) and self.is_private:
            update_1d = [u.flatten() for u in updates]
            updates = [ np.dot(self.Vk[i].T, update_1d[i]-self.mean[i]) for i in range(len(update_1d)) ]
        print('update[0].shape: {}'.format(updates[0].shape))

        # update the budget accountant
        accum_bgts = self.ba.update(self.loc_steps) if self.ba is not None else None

        return updates, accum_bgts
