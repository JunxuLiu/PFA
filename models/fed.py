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

from utils import Vname_to_FeedPname
np.random.seed(10)


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


def Lanczos( mat, m=128 ):

    print('lanczos iteration:', m)

    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    n = mat[0].shape[0]
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0,v0))
    
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    V[0, :] = v0
    
    # step 2.1 - 2.3 in https://en.wikipedia.org/wiki/Lanczos_algorithm
    #print(mat)
    w = np.sum([np.dot(col, np.dot(col.T, V[0,:])) for col in mat], 0)
    alfa = np.dot(w, V[0,:])
    w = w - alfa * V[0,:]
    T[0,0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m-1):
        
        beta = np.sqrt( np.dot( w, w ) )
        #print('w:{}, beta:{}'.format(w, beta))
        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(np.conj(V[j,:]), V[i, :])*V[i,:]
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])

        w = np.sum([np.dot(col, np.dot(col.T, V[j,:])) for col in mat], 0)
        alfa = np.dot(w, V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1,j] = beta
        T[j,j-1] = beta
    
    return T, V


class ServerAggregation(object):

    def __init__(self, glob_model, dpsgd=True, projection=True, proj_dims=10, lanczos_iter=128, wavg=False, weights=None):

        self.keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]
        self.shapes = [glob_model[k].shape for k in self.keys]
        self.num_vars = len(self.keys)
        
        self.projection = projection
        self.dpsgd = dpsgd
        self.wavg = wavg
        self.w = weights

        if not projection:
            self.updates = []

        else:
            if dpsgd:
                self.num_pub = 0
                self.num_priv = 0
                self.priv_updates = []
                self.pub_updates = []
            else:
                self.updates = []

            self.proj_dims = proj_dims
            self.lanczos_iter = lanczos_iter

    def aggregate(self, cid, update, is_public=False):

        def _aggregate(_update, _agg_updates):

            return [np.expand_dims(_update[i],0) for i in range(self.num_vars)] if not len(_agg_updates) else \
                   [np.append(_agg_updates[i], np.expand_dims(_update[i],0), 0) for i in range(self.num_vars)]

        aggregate_fn = lambda var1, var2 : _aggregate(var1, var2)

        if not self.projection:
            self.updates = aggregate_fn(update, self.updates)
        else:
            update_1d = [u.flatten() for u in update]
            
            if not self.dpsgd:
                self.updates = aggregate_fn(update_1d, self.updates)
                
            elif is_public:
                #print('is_public')
                self.num_pub += 1   
                self.pub_updates = aggregate_fn(update_1d, self.pub_updates)
             
            else:
                #print('is_private')
                self.num_priv += 1
                self.priv_updates = aggregate_fn(update_1d, self.priv_updates)


    def _standardize(self, M):
        '''Compute the mean of every dimension of the whole dataset'''
        [n, m] = M.shape
        if m == 1:
            return M, np.zeros(n)

        # calculate the mean 
        mean = np.dot(M,np.ones((m,1), dtype=np.float32)) / m
        #mean = [np.mean(M[i]) / m for i in range(n)]
                
        return M - mean, mean.flatten()

    
    def _eigen_by_lanczos(self, mat):
        #mat = self._standardize(mat)
        
        #mat = [col.reshape(1,-1) for col in mat]
        T, V = Lanczos(mat, self.lanczos_iter)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(self.proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


    def _project_priv_updates(self, errors):

        if len(self.priv_updates):
            if errors is not None:
                MeanPrivUps = [np.mean(self.priv_updates[i], 0) + errors[i].flatten() for i in range(self.num_vars)]
            else:
                MeanPrivUps = [np.mean(self.priv_updates[i], 0) for i in range(self.num_vars)]

            MeanPubUps = [np.mean(self.pub_updates[i], 0) for i in range(self.num_vars)]
            MeanProjPrivUps = [0] * self.num_vars
            MeanUpdates = [0] * self.num_vars
            new_errors = [0] * self.num_vars
            
            for i in range(self.num_vars):
                #start_time = time.time()
                #print(self.pub_updates[i].T)
                pub_updates, mean = self._standardize(self.pub_updates[i].T)
                #print('pub_updates.shape:{}, mean.shape:{}'.format(pub_updates.shape, mean.shape))
                Vk = self._eigen_by_lanczos(pub_updates.T)
                #print('Vk.shape:{}, (MeanPrivUps[{}] - mean).shape:{}'.format(Vk.shape, i, (MeanPrivUps[i] - mean).shape))
                MeanProjPrivUps[i] = np.dot(Vk, np.dot(Vk.T, (MeanPrivUps[i] - mean))) + mean
                #print('MeanProjPrivUps[i].shape:{}, MeanPubUps[i].shape:{}'.format(MeanProjPrivUps[i].shape, MeanPubUps[i].shape))
                #print('projection time:', time.time() - start_time)
                MeanUpdates[i] = ((self.num_priv * MeanProjPrivUps[i] + self.num_pub * MeanPubUps[i]) /
                                  (self.num_pub + self.num_priv)).reshape(self.shapes[i])
                if errors is not None:
                    new_errors[i] = (MeanPrivUps[i] - MeanProjPrivUps[i]).reshape(self.shapes[i])
             
            return MeanUpdates, new_errors

        elif len(self.pub_updates) and not len(self.priv_updates):

            MeanUpdates = [np.mean(self.pub_updates[i], 0).reshape(self.shapes[i]) for i in range(self.num_vars)]
            return MeanUpdates, errors

        else:
            raise ValueError('Cannot process the projection without private local updates.')


    def _pca_reconstruction(self, errors):
        if errors is not None:
            MeanUps = [np.mean(self.updates[i], 0) + errors[i].reshape(-1,1) for i in range(self.num_vars)]

        else:
            MeanUps = [np.mean(self.updates[i], 0) for i in range(self.num_vars)]

        MeanProjUps = [0] * self.num_vars
        new_errors = [0] * self.num_vars
        for i in range(self.num_vars):
            Vk = self._update_by_lanczos(self.updates[i])
            MeanProjUps[i] = np.dot(Vk, np.dot(Vk.T, MeanUps[i]))            
            if errors is not None:
                new_errors[i] = (MeanUps[i] - MeanProjUps[i]).reshape(self.shapes[i])
            MeanProjUps[i] = MeanProjUps[i].reshape(self.shapes[i])

        return MeanProjUps, new_errors

    def fedavg(self, glob_model, errors=None, weights=None):
        #MeanUpdates /= 0.1 #learning rate = 0.1
        if self.projection:
            #print('self.projection:{}'.format(self.projection))
            if self.dpsgd:
                #print('self.dpsgd:{}'.format(self.dpsgd))
                MeanUpdates, new_errors = self._project_priv_updates(errors)
                self.num_pub = 0
                self.num_priv = 0
                self.priv_updates = []
                self.pub_updates = []

            else:
                MeanUpdates, new_errors = self._pca_reconstruction(errors)
                self.updates = []

        elif self.wavg:
            assert( weights is None,  'weights is None while you want to apply weighted averaging.')
            MeanUpdates = [np.average(self.updates[i], 0, weights).reshape(self.shapes[i]) for i in range(self.num_vars)]
            self.updates = []
         
        else: #without both projection and weighte averaging
            MeanUpdates = [np.average(self.updates[i], 0).reshape(self.shapes[i]) for i in range(self.num_vars)]
            self.updates = []

        # average all collected updates
        new_weights = [glob_model[self.keys[i]] - MeanUpdates[i] for i in range(self.num_vars)]
        new_model = dict(zip(self.keys, new_weights))
        '''
        bias = [glob_model[self.keys[i]] - new_model[self.keys[i]] for i in range(self.num_vars)]
        print(bias)
        '''
        if errors is not None:
            return new_model, new_errors

        return new_model
