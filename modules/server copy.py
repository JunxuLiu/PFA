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

from lanczos import Lanczos
from utils import *
np.random.seed(10)

# def Lanczos( mat, m=128 ):

#     print('lanczos iteration:', m)

#     # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
#     n = mat[0].shape[0]
#     v0 = np.random.rand(n)
#     v0 /= np.sqrt(np.dot(v0,v0))
    
#     V = np.zeros( (m,n) )
#     T = np.zeros( (m,m) )
#     V[0, :] = v0
    
#     # step 2.1 - 2.3 in https://en.wikipedia.org/wiki/Lanczos_algorithm
#     #print(mat)
#     w = np.sum([np.dot(col, np.dot(col.T, V[0,:])) for col in mat], 0)
#     alfa = np.dot(w, V[0,:])
#     w = w - alfa * V[0,:]
#     T[0,0] = alfa

#     # needs to start the iterations from indices 1
#     for j in range(1, m-1):
        
#         beta = np.sqrt( np.dot( w, w ) )
#         #print('w:{}, beta:{}'.format(w, beta))
#         V[j,:] = w/beta

#         # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
#         for i in range(j-1):
#             V[j, :] = V[j,:] - np.dot(np.conj(V[j,:]), V[i, :])*V[i,:]
#         V[j, :] = V[j, :]/np.linalg.norm(V[j, :])

#         w = np.sum([np.dot(col, np.dot(col.T, V[j,:])) for col in mat], 0)
#         alfa = np.dot(w, V[j, :])
#         w = w - alfa * V[j, :] - beta*V[j-1, :]

#         T[j,j  ] = alfa
#         T[j-1,j] = beta
#         T[j,j-1] = beta
    
#     return T, V


class Server(Aggregation):

    def __init__(self, num_clients, sample_mode, sample_ratio):

        self.num_clients = num_clients
        self.samole_mode = sample_mode
        self.sample_ratio = sample_ratio

        self.__aggregate_updates = ServerAggregation()


    def __a_res(items, weights, m):
        """
        :samples: [(item, weight), ...]
        :k: number of selected items
        :returns: [(item, weight), ...]
        """
        weights = np.array(weights) / sum(weights)
        heap = [] # [(new_weight, item), ...]
        for i in items:
            wi = weights[i]
            ui = np.random.random()
            ki = ui ** (1/wi)

            if len(heap) < m:
                heapq.heappush(heap, (ki, i))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, i))

            if len(heap) > m:
                heapq.heappop(heap)

        return [item[1] for item in heap]


    def __naive_weighted_sampling(items, weights, m):
        weights = np.array(weights) / max(weights)
        samples = [ item for item in items if np.random.random() <= weights[item] ][0:min(m, len(items))]
        return samples


    def __top_k(items, weights, m):
        heap = [] # [(new_weight, item), ...]
        for i in items:
            wi = weights[i]

            if len(heap) < m:
                heapq.heappush(heap, (wi, i))
            elif wi > heap[0][0]:
                heapq.heappush(heap, (wi, i))
                if len(heap) > m:
                    heapq.heappop(heap)

        return [item[1] for item in heap]


    def sample_clients(client_set, client_batch_size, mode=None, ba=None):
        # Randomly choose a total of m (out of n) client-indices that participate in this round
        # randomly permute a range-list of length n: [1,2,3...n] --> [5,2,7..3]
        m = int(self.num_clients * self.sample_ratio)
        update = lambda x: ba.update(x)

        s = ba.precheck(N, client_set, client_batch_size)
        if len(s) < m:
            return []

        if mode == 'None':
            print('Full client participation.')
            return s

        elif mode == 'W' and len(s) > m:
            print('Partial client participation with weighted(adaptive) sampling.')
            remainder = ba.get_remainder()
            print('remainder:', remainder)
            #s = _naive_weighted_sampling(s, remainder, m)
            #s = _a_res(s, remainder, m)
            s_ = _top_k(s, remainder, m) if mode == 'W1' else a_res(s, remainder, m)
            update(s_)

        else:
            print('Partial client participation with ramdom sampling.')
            s_ = list(np.random.permutation(s))[0:m]
            # Only when we are running Pfizer method, `ba._public` is not None.
            # For FedAvg or WAVG or MIN/MAX, public clients are not necessary while sampling.
            if ba._public is None:
                update(s_)
                return s_
            
            # For Pfizer, we require the subset contains at least 1 public and 1 private client.
            check = 50
            while check and len(set(s_).intersection(set(ba._public))) == 0:
                check -= 1
                print('There are no public clients be sampled in this round.')
                s_ = list(np.random.permutation(s))[0:m]

            if check == 0:
                return []

            update(s_)

        return s_


class Aggregation(object):

    def __init__(self, glob_model, dpsgd=True, projection=True, proj_dims=10, lanczos_iter=128, wavg=False, weights=None):

        self.__keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]
        self.__shapes = [glob_model[k].shape for k in self.keys]
        self.__num_vars = len(self.keys)
        
        self.projection = projection
        self.dpsgd = dpsgd
        self.wavg = wavg
        self.w = weights

        if not projection:
            self.__updates = []

        else:
            if dpsgd:
                self.__num_pub = 0
                self.__num_priv = 0
                self.__priv_updates = []
                self.__pub_updates = []
            else:
                self.__updates = []

            self.proj_dims = proj_dims
            self.lanczos_iter = lanczos_iter


    def fedavg():
        """
        Federated Averaging
        """

    def aggregate(self, cid, update, is_public=False):

        def _aggregate(_update, _agg_updates):

            return [np.expand_dims(_update[i],0) for i in range(self.num_vars)] if not len(_agg_updates) else \
                   [np.append(_agg_updates[i], np.expand_dims(_update[i],0), 0) for i in range(self.num_vars)]

        aggregate_fn = lambda var1, var2 : _aggregate(var1, var2)

        if not self.projection:
            self.__updates = aggregate_fn(update, self.updates)
        else:
            update_1d = [u.flatten() for u in update]
            
            if not self.dpsgd:
                self.__updates = aggregate_fn(update_1d, self.updates)
                
            elif is_public:
                #print('is_public')
                self.__num_pub += 1   
                self.__pub_updates = aggregate_fn(update_1d, self.pub_updates)
             
            else:
                #print('is_private')
                self.__num_priv += 1
                self.__priv_updates = aggregate_fn(update_1d, self.priv_updates)


    def __standardize(self, M):
        '''Compute the mean of every dimension of the whole dataset'''
        [n, m] = M.shape
        if m == 1:
            return M, np.zeros(n)

        # calculate the mean 
        mean = np.dot(M,np.ones((m,1), dtype=np.float32)) / m
        #mean = [np.mean(M[i]) / m for i in range(n)]
                
        return M - mean, mean.flatten()

    
    def __eigen_by_lanczos(self, mat):
        #mat = self._standardize(mat)
        
        #mat = [col.reshape(1,-1) for col in mat]
        T, V = Lanczos(mat, self.lanczos_iter)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(self.proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


    def __project_priv_updates(self, errors):

        if len(self.priv_updates):
            if errors is not None:
                MeanPrivUps = [np.mean(self.__priv_updates[i], 0) + errors[i].flatten() for i in range(self.num_vars)]
            else:
                MeanPrivUps = [np.mean(self.__priv_updates[i], 0) for i in range(self.num_vars)]

            MeanPubUps = [np.mean(self.__pub_updates[i], 0) for i in range(self.num_vars)]
            MeanProjPrivUps = [0] * self.num_vars
            MeanUpdates = [0] * self.num_vars
            new_errors = [0] * self.num_vars
            
            for i in range(self.num_vars):
                #start_time = time.time()
                #print(self.pub_updates[i].T)
                pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                #print('pub_updates.shape:{}, mean.shape:{}'.format(pub_updates.shape, mean.shape))
                Vk = self.__eigen_by_lanczos(pub_updates.T)
                #print('Vk.shape:{}, (MeanPrivUps[{}] - mean).shape:{}'.format(Vk.shape, i, (MeanPrivUps[i] - mean).shape))
                MeanProjPrivUps[i] = np.dot(Vk, np.dot(Vk.T, (MeanPrivUps[i] - mean))) + mean
                #print('MeanProjPrivUps[i].shape:{}, MeanPubUps[i].shape:{}'.format(MeanProjPrivUps[i].shape, MeanPubUps[i].shape))
                #print('projection time:', time.time() - start_time)
                MeanUpdates[i] = ((self.__num_priv * MeanProjPrivUps[i] + self.__num_pub * MeanPubUps[i]) /
                                  (self.__num_pub + self.__num_priv)).reshape(self.shapes[i])
                if errors is not None:
                    new_errors[i] = (MeanPrivUps[i] - MeanProjPrivUps[i]).reshape(self.shapes[i])
             
            return MeanUpdates, new_errors

        elif len(self.pub_updates) and not len(self.priv_updates):

            MeanUpdates = [np.mean(self.__pub_updates[i], 0).reshape(self.shapes[i]) for i in range(self.__num_vars)]
            return MeanUpdates, errors

        else:
            raise ValueError('Cannot process the projection without private local updates.')


    def __pca_reconstruction(self, errors):
        if errors is not None:
            MeanUps = [np.mean(self.__updates[i], 0) + errors[i].reshape(-1,1) for i in range(self.num_vars)]

        else:
            MeanUps = [np.mean(self.__updates[i], 0) for i in range(self.num_vars)]

        MeanProjUps = [0] * self.__num_vars
        new_errors = [0] * self.__num_vars
        for i in range(self.num_vars):
            Vk = self._update_by_lanczos(self.__updates[i])
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
                self.__num_pub = 0
                self.__num_priv = 0
                self.__priv_updates = []
                self.__pub_updates = []

            else:
                MeanUpdates, new_errors = self._pca_reconstruction(errors)
                self.__updates = []

        elif self.wavg:
            assert( weights is None,  'weights is None while you want to apply weighted averaging.')
            MeanUpdates = [np.average(self.__updates[i], 0, weights).reshape(self.shapes[i]) for i in range(self.__num_vars)]
            self.updates = []
         
        else: #without both projection and weighte averaging
            MeanUpdates = [np.average(self.__updates[i], 0).reshape(self.shapes[i]) for i in range(self.num_vars)]
            self.updates = []

        # average all collected updates
        new_weights = [glob_model[self.keys[i]] - MeanUpdates[i] for i in range(self.__num_vars)]
        new_model = dict(zip(self.keys, new_weights))
        '''
        bias = [glob_model[self.keys[i]] - new_model[self.keys[i]] for i in range(self.num_vars)]
        print(bias)
        '''
        if errors is not None:
            return new_model, new_errors

        return new_model
