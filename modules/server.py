from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import abc
import os
import math
import copy
import tensorflow.compat.v1 as tf
import numpy as np
import scipy
import time

from modules.lanczos import Lanczos

from utils.tf_frame import Vname_to_FeedPname

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



class ServerOperation(metaclass=abc.ABCMeta):

    def _add_one(self, num_vars, _update, _agg_updates):

        return [np.expand_dims(_update[i],0) for i in range(num_vars)] if not len(_agg_updates) else \
                [np.append(_agg_updates[i], np.expand_dims(_update[i],0), 0) for i in range(num_vars)]


    @abc.abstractmethod
    def aggregate(self, update, is_public=None):
        """
        Aggregation
        """

    @abc.abstractmethod
    def average(self):
        """
        Federated Averaging: average all collected updates
        """

    def update(self, glob_model, eps_list=None):
        """
        return the updated global model
        """
        
        keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]
        num_vars = len(keys)
        shape_vars = [glob_model[k].shape for k in keys ]
        mean_updates = self.average(num_vars, shape_vars, eps_list)

        new_weights = [glob_model[keys[i]] - mean_updates[i] for i in range(num_vars)]
        new_model = dict(zip(keys, new_weights))

        return new_model


class FedAvg(ServerOperation):
    
    def __init__(self):
        
        print('Using naive FedAvg algorithm...')
        self.__updates = []

    def aggregate(self, update, is_public=None):
        num_vars = len(update)
        aggregate_fn = lambda var1, var2 : self._add_one(num_vars, var1, var2)
        self.__updates = aggregate_fn(update, self.__updates)

    def average(self, num_vars=None, shape_vars=None, eps_subset=None):
        mean_updates = [np.average(self.__updates[i], 0).reshape(shape_vars[i]) \
                        for i in range(num_vars)]
        self.updates = []
        return mean_updates


class WeiAvg(ServerOperation):

    def __init__(self):

        print('Using weighted averaging algorithm...')
        self.__updates = []

    def aggregate(self, update, is_public=None):
        num_vars = len(update)
        aggregate_fn = lambda var1, var2 : self._add_one(num_vars, var1, var2)
        self.__updates = aggregate_fn(update, self.__updates)

    def average(self, num_vars=None, shape_vars=None, eps_subset=None):
        print(eps_subset)
        eps_sum = sum(eps_subset)
        weights = np.array([eps/eps_sum for eps in eps_subset])

        mean_updates = [np.average(self.__updates[i], 0, weights).reshape(shape_vars[i]) \
                        for i in range(num_vars)]
        self.__updates = []
        return mean_updates


class Pfizer(ServerOperation):

    def __init__(self, proj_dims, lanczos_iter):

        print('Using projected averaging (Pfizer) algorithm...')
        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []

        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        

    def aggregate(self, update, is_public=False):
        num_vars = len(update)
        update_1d = [u.flatten() for u in update]
        aggregate_fn = lambda var1, var2 : self._add_one(num_vars, var1, var2)
        if is_public:
            #print('is_public')
            self.__num_pub += 1   
            self.__pub_updates = aggregate_fn(update_1d, self.__pub_updates)
            
        else:
            #print('is_private')
            self.__num_priv += 1
            self.__priv_updates = aggregate_fn(update_1d, self.__priv_updates)


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
        T, V = Lanczos(mat, self.lanczos_iter)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(self.proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


    def __project_priv_updates(self, num_vars, shape_vars):
        if len(self.__priv_updates):
            mean_priv_updates = [np.mean(self.__priv_updates[i], 0) for i in range(num_vars)]
            mean_pub_updates = [np.mean(self.__pub_updates[i], 0) for i in range(num_vars)]
            mean_proj_priv_updates = [0] * num_vars
            mean_updates = [0] * num_vars
            
            for i in range(num_vars):
                #start_time = time.time()
                #print(self.pub_updates[i].T)
                pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                #print('pub_updates.shape:{}, mean.shape:{}'.format(pub_updates.shape, mean.shape))
                Vk = self.__eigen_by_lanczos(pub_updates.T)
                #print('Vk.shape:{}, (mean_priv_updates[{}] - mean).shape:{}'.format(Vk.shape, i, (mean_priv_updates[i] - mean).shape))
                mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                #print('mean_proj_priv_updates[i].shape:{}, mean_pub_updates[i].shape:{}'.format(mean_proj_priv_updates[i].shape, mean_pub_updates[i].shape))
                #print('projection time:', time.time() - start_time)
                mean_updates[i] = ((self.__num_priv * mean_proj_priv_updates[i] + self.__num_pub * mean_pub_updates[i]) /
                                  (self.__num_pub + self.__num_priv)).reshape(shape_vars[i])

            return mean_updates

        elif len(self.__pub_updates) and not len(self.__priv_updates):

            mean_updates = [np.mean(self.__pub_updates[i], 0).reshape(shape_vars[i]) for i in range(num_vars)]
            return mean_updates

        else:
            raise ValueError('Cannot process the projection without private local updates.')


    def average(self, num_vars, shape_vars, eps_list=None):
        mean_updates = self.__project_priv_updates(num_vars, shape_vars)
        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []
        return mean_updates


class Server(object):

    def __init__(self, num_clients, sample_mode, sample_ratio):

        self.num_clients = num_clients
        self.sample_mode = sample_mode
        self.sample_ratio = sample_ratio

        self.__public = None

    def set_public_clients(self, epsilons):
        sorted_eps = np.sort(epsilons)    
        percent = 0.1
        threshold = sorted_eps[-int(percent * self.num_clients)]

        self.__public = list(np.where(np.array(epsilons) >= threshold)[0])
        

    def init_global_model(self, sess):

        keys = [Vname_to_FeedPname(var) for var in tf.trainable_variables()]
        global_model = dict(zip(keys, [sess.run(var) for var in tf.trainable_variables()]))
        global_model['global_step_placeholder:0'] = 0
    
        return global_model


    def init_alg(self, fedavg=False, weiavg=False, projection=True,\
                dp=True, proj_dims=None, lanczos_iter=None):

        if fedavg:
            self.__alg = FedAvg()

        elif weiavg:
            assert( dp==False,  'Detected DP components were not applied so that the WeiAvg algorithm was denied.')
            self.__alg = WeiAvg()

        elif projection:
            assert( dp==False,  'Detected DP components were not applied so that the Pfizer algorithm was denied.')
            self.__alg = Pfizer(proj_dims, lanczos_iter)

        else:
            raise ValueError('Choose an algorithm (FedAvg/WeiAvg/Pfizer) to get the aggregated model.')

    def aggregate(self, cid, update):

        if self.__public:
            self.__alg.aggregate(update, is_public=True if (cid in self.__public) else False)
        else:
            self.__alg.aggregate(update)

    def update(self, global_model, eps_list=None):

        return self.__alg.update(global_model, eps_list)


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

    def sample_clients(self, candidates):
        # Randomly choose a total of m (out of n) client-indices that participate in this round
        # randomly permute a range-list of length n: [1,2,3...n] --> [5,2,7..3]
        m = int(self.num_clients * self.sample_ratio)
        if len(candidates) < m:
            return []

        if self.sample_mode == 'None':
            print('Full client participation.')
            return candidates

        else:
            print('Partial client participation with ramdom client sampling.')
            participants = list(np.random.permutation(candidates))[0:m]

            # Only when we are running Pfizer method, `ba._public` is not None.
            # For FedAvg or WAVG or MIN/MAX, public clients are not necessary while sampling.
            if self.__public is None:
                return participants
                
            # For Pfizer, we require the subset contains at least 1 public and 1 private client.
            check = 50
            while check and len(set(participants).intersection(set(self.__public))) == 0:
                check -= 1
                print('There are no public clients be sampled in this round.')
                participants = list(np.random.permutation(candidates))[0:m]

            return participants if check else []
        
