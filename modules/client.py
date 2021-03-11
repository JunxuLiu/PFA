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

from utils import *
from modules.budgets_accountant import BudgetsAccountant
from utils.tf_frame import Vname_to_FeedPname, Vname_to_Pname
np.random.seed(10)


class Client(object):

    def __init__(self, x_train, y_train, batch_size, loc_steps):
        self.x_train = x_train
        self.y_train = y_train
        self.dataset_size = len(x_train)
        self.batch_size = batch_size
        self.loc_steps = loc_steps
        
        self.ba = None

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

    def local_update(self, sess, assignments, glob_model):

        # local SGD then get the model updates
        sess.run(assignments, feed_dict=glob_model)
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
            _ = sess.run(self.train_op, feed_dict=feed_dict)

        updates = [glob_model[Vname_to_FeedPname(var)] - sess.run(var) for var in tf.trainable_variables()]

        # update the budget accountant
        accum_bgts = self.ba.update(self.loc_steps) if self.ba is not None else None

        return updates, accum_bgts
        

class LocalUpdate(object):

    def __init__(self, x_train, y_train, client_set, batch_size, data_placeholder, labels_placeholder):
        self.x_train = x_train
        self.y_train = y_train
        self.client_set = client_set
        self.batch_size = batch_size
        self.data_placeholder = data_placeholder
        self.labels_placeholder = labels_placeholder

    def update(self, sess, assignments, cid, glob_model, iters, train_op):
        sess.run(assignments, feed_dict=glob_model)
        
        for it in range(iters):
            #print('it: {}'.format(it))
            t1 = time.time()

            loc_dataset = np.random.permutation(self.client_set[cid])
            # batch_ind holds the indices of the current batch
            batch_ind = loc_dataset[0:self.batch_size]

            # Fill a feed dictionary with the actual set of data and labels using the data and labels associated
            # to the indices stored in batch_ind:
            feed_dict = {str(self.data_placeholder.name): self.x_train[[int(j) for j in batch_ind]],
                         str(self.labels_placeholder.name): self.y_train[[int(j) for j in batch_ind]]}
            #t2 = time.time()
            # Run one optimization step.
            _ = sess.run(train_op, feed_dict=feed_dict)
            #t3 = time.time()
            #print((t3-t1)/60, (t3-t2)/60)

        updates = [glob_model[Vname_to_FeedPname(var)] - sess.run(var) for var in tf.trainable_variables()]

        return updates

