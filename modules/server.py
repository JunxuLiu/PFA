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
from common_utils.tf_utils import Vname_to_FeedPname
np.random.seed(10)

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
        self.__updates = []
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
        
        eps_sum = sum(eps_subset)
        weights = np.array([eps/eps_sum for eps in eps_subset])
        print('weights: {}'.format(weights))
        
        mean_updates = [np.average(self.__updates[i], 0, weights).reshape(shape_vars[i]) \
                        for i in range(num_vars)]
        self.__updates = []
        return mean_updates


class PFA(ServerOperation):

    def __init__(self, proj_dims, lanczos_iter, delay):

        print('Using projected averaging (Pfizer) algorithm...')
        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []
        
        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.delay = delay
        self.Vk = None
        self.mean = None

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
            print(m==1)
            return M, np.zeros(n)
        # calculate the mean 
        mean = np.dot(M,np.ones((m,1), dtype=np.float32)) / m
        return M - mean, mean.flatten()


    def __eigen_by_lanczos(self, mat):
        T, V = Lanczos(mat, self.lanczos_iter)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(self.proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk

    def __projection(self, num_vars, shape_vars):

        if len(self.__priv_updates):
            mean_priv_updates = [np.mean(self.__priv_updates[i], 0) for i in range(num_vars)]
            mean_pub_updates = [np.mean(self.__pub_updates[i], 0) for i in range(num_vars)]
            mean_proj_priv_updates = [0] * num_vars
            mean_updates = [0] * num_vars
            
            for i in range(num_vars):
                pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                Vk = self.__eigen_by_lanczos(pub_updates.T)
                mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                mean_updates[i] = ((self.__num_priv * mean_proj_priv_updates[i] + self.__num_pub * mean_pub_updates[i]) /
                                  (self.__num_pub + self.__num_priv)).reshape(shape_vars[i])

            return mean_updates

        elif len(self.__pub_updates) and not len(self.__priv_updates):

            mean_updates = [np.mean(self.__pub_updates[i], 0).reshape(shape_vars[i]) for i in range(num_vars)]
            return mean_updates

        else:
            raise ValueError('Cannot process the projection without private local updates.')


    def __delayed_projection(self, num_vars, shape_vars, warmup=False):

        if len(self.__priv_updates):
            mean_pub_updates = [np.mean(self.__pub_updates[i], 0) for i in range(num_vars)]
            mean_priv_updates = [np.mean(self.__priv_updates[i], 0) for i in range(num_vars)]
            mean_proj_priv_updates = [0] * num_vars
            mean_updates = [0] * num_vars

            Vks = []
            means = []
            if warmup:
                for i in range(num_vars):

                    pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                    Vk = self.__eigen_by_lanczos(pub_updates.T)
                    Vks.append(Vk)
                    means.append(mean)
                    
                    mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                    mean_updates[i] = ((self.__num_priv * mean_proj_priv_updates[i] + self.__num_pub * mean_pub_updates[i]) /
                                    (self.__num_pub + self.__num_priv)).reshape(shape_vars[i])
                    

            else:
                for i in range(num_vars):

                    mean_proj_priv_updates[i] = np.dot(self.Vk[i], mean_priv_updates[i]) + self.mean[i]
                    mean_updates[i] = ((self.__num_priv * mean_proj_priv_updates[i] + self.__num_pub * mean_pub_updates[i]) /
                                        (self.__num_pub + self.__num_priv)).reshape(shape_vars[i])

                    pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                    Vk = self.__eigen_by_lanczos(pub_updates.T)
                    Vks.append(Vk)
                    means.append(mean)
            
            self.Vk = Vks
            self.mean = means
            return mean_updates


        elif len(self.__pub_updates) and not len(self.__priv_updates):

            mean_updates = [np.mean(self.__pub_updates[i], 0).reshape(shape_vars[i]) for i in range(num_vars)]
            return mean_updates

        else:
            raise ValueError('Cannot process the projection without private local updates.')


    def average(self, num_vars, shape_vars, eps_list=None):

        if self.delay:
            mean_updates = self.__delayed_projection(num_vars, shape_vars, warmup=(self.Vk is None))
        else:
            mean_updates = self.__projection(num_vars, shape_vars)

        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []
        return mean_updates


class WeiPFA(ServerOperation):
    def __init__(self, proj_dims, lanczos_iter, delay):
        print('Using projected averaging (Pfizer) algorithm...')

        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []
        self.__priv_eps = []
        self.__pub_eps = []

        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.delay = delay
        self.Vk = None
        self.mean = None

    def aggregate(self, eps, update, is_public=False):
        num_vars = len(update)
        update_1d = [u.flatten() for u in update]

        aggregate_fn = lambda var1, var2 : self._add_one(num_vars, var1, var2)
        if is_public:
            #print('is_public')
            self.__num_pub += 1
            self.__pub_eps.append(eps)
            self.__pub_updates = aggregate_fn(update_1d, self.__pub_updates)
            
        else:
            #print('is_private')
            self.__num_priv += 1
            self.__priv_eps.append(eps)
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
        #sorted_vec = sorted(T_evals, reverse=True)
        #print(sorted_vec)
        return Vk

    
    def __weighted_project_priv_updates(self, num_vars, shape_vars):

        if len(self.__priv_updates):
            
            priv_weights = np.array(self.__priv_eps) / sum(self.__priv_eps)
            pub_weights = np.array(self.__pub_eps) / sum(self.__pub_eps)
            #print(priv_weights, pub_weights)
            mean_priv_updates = [np.average(self.__priv_updates[i], 0, priv_weights) \
                                for i in range(num_vars)]
            mean_pub_updates = [np.average(self.__pub_updates[i], 0, pub_weights) \
                                for i in range(num_vars)]
            mean_proj_priv_updates = [0] * num_vars
            mean_updates = [0] * num_vars
            
            print(num_vars)
            for i in range(num_vars):
                print('__pub_updates[{}].shape: {}'.format(i, self.__pub_updates[i].shape))
                pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                print('pub_updates[{}].shape: {}'.format(i, pub_updates[i].shape))
                Vk = self.__eigen_by_lanczos(pub_updates.T)
                print('Vk.shape: {}'.format(Vk.shape))
                mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                mean_updates[i] = ((mean_proj_priv_updates[i] * sum(self.__priv_eps) + mean_pub_updates[i] * sum(self.__pub_eps)) 
                                    / sum(self.__priv_eps + self.__pub_eps)).reshape(shape_vars[i])

            return mean_updates

        elif len(self.__pub_updates) and not len(self.__priv_updates):

            mean_updates = [np.mean(self.__pub_updates[i], 0).reshape(shape_vars[i]) for i in range(num_vars)]
            return mean_updates

        else:
            raise ValueError('Cannot process the projection without private local updates.')
    
    
    def __delayed_weighted_project_priv_updates(self, num_vars, shape_vars, warmup=False):

        if len(self.__priv_updates):
            priv_weights = np.array(self.__priv_eps) / sum(self.__priv_eps)
            pub_weights = np.array(self.__pub_eps) / sum(self.__pub_eps)

            mean_pub_updates = [np.average(self.__pub_updates[i], 0, pub_weights) \
                                for i in range(num_vars)]
            mean_priv_updates = [np.average(self.__priv_updates[i], 0, priv_weights) \
                                for i in range(num_vars)]
            mean_proj_priv_updates = [0] * num_vars
            mean_updates = [0] * num_vars

            Vks = []
            means = []

            if warmup:
                for i in range(num_vars):

                    pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                    Vk = self.__eigen_by_lanczos(pub_updates.T)
                    Vks.append(Vk)
                    means.append(mean)
                    
                    mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                    mean_updates[i] = ((sum(self.__priv_eps) * mean_proj_priv_updates[i] + sum(self.__pub_eps) * mean_pub_updates[i]) /
                                    (sum(self.__priv_eps) + sum(self.__pub_eps))).reshape(shape_vars[i])
            else:
                for i in range(num_vars):
                
                    mean_proj_priv_updates[i] = np.dot(self.Vk[i], mean_priv_updates[i]) + self.mean[i]
                    
                    mean_updates[i] = ((sum(self.__priv_eps) * mean_proj_priv_updates[i] + sum(self.__pub_eps) * mean_pub_updates[i]) /
                                    (sum(self.__priv_eps + self.__pub_eps))).reshape(shape_vars[i])

                    pub_updates, mean = self.__standardize(self.__pub_updates[i].T)
                    Vk = self.__eigen_by_lanczos(pub_updates.T)
                    Vks.append(Vk)
                    means.append(mean)
            
            self.Vk = Vks
            self.mean = means
            return mean_updates


        elif len(self.__pub_updates) and not len(self.__priv_updates):

            mean_updates = [np.mean(self.__pub_updates[i], 0).reshape(shape_vars[i]) for i in range(num_vars)]
            return mean_updates

        else:
            raise ValueError('Cannot process the projection without private local updates.')

    def average(self, num_vars, shape_vars, eps_list=None):
        if not self.delay:
            mean_updates = self.__weighted_project_priv_updates(num_vars, shape_vars)
        else:
            mean_updates = self.__delayed_weighted_project_priv_updates(num_vars, shape_vars, warmup=(self.Vk is None))

        self.__num_pub = 0
        self.__num_priv = 0
        self.__priv_updates = []
        self.__pub_updates = []
        self.__priv_eps = []
        self.__pub_eps = []
        return mean_updates


class Server(object):

    def __init__(self, num_clients, sample_mode, sample_ratio):

        self.num_clients = num_clients
        self.sample_mode = sample_mode
        self.sample_ratio = sample_ratio

        self.public = None
        self.__epsilons = None

    '''clustering'''
    def set_public_clients(self, epsilons):
        self.__epsilons = epsilons

        sorted_eps = np.sort(epsilons) 
        percent = 0.1
        threshold = sorted_eps[-int(percent * self.num_clients)]
        
        self.public = list(np.where(np.array(epsilons) >= threshold)[0])
        

    def init_global_model(self, sess):

        keys = [Vname_to_FeedPname(var) for var in tf.trainable_variables()]
        global_model = dict(zip(keys, [sess.run(var) for var in tf.trainable_variables()]))

        return global_model


    def init_alg(self, dp=True, fedavg=False, weiavg=False, \
                projection=False, proj_wavg=True, delay=True, proj_dims=None, lanczos_iter=None):
    
        if fedavg or (not dp):
            self.__alg = FedAvg()

        elif weiavg:
            assert( dp==False,  'Detected DP components were not applied so that the WeiAvg algorithm was denied.')
            self.__alg = WeiAvg()

        elif projection:
            assert( dp==False,  'Detected DP components were not applied so that the Pfizer algorithm was denied.')
            self.__alg = PFA(proj_dims, lanczos_iter, delay)

        elif proj_wavg:
            assert( dp==False,  'Detected DP components were not applied so that the Pfizer algorithm was denied.')
            self.__alg = WeiPFA(proj_dims, lanczos_iter, delay)

        else:
            raise ValueError('Choose an algorithm (FedAvg/WeiAvg/Pfizer) to get the aggregated model.')

    def get_proj_info(self):
        return self.__alg.Vk, self.__alg.mean

    def aggregate(self, cid, update, projection=False, proj_wavg=False):
        if projection:
            self.__alg.aggregate(update, is_public=True if (cid in self.public) else False)
        elif proj_wavg:
            self.__alg.aggregate(self.__epsilons[cid], update, is_public=True if (cid in self.public) else False)
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
            if self.public is None:
                return participants
                
            # For Pfizer, we require the subset contains at least 1 public and 1 private client.
            check = 50
            while check and len(set(participants).intersection(set(self.public))) == 0:
                check -= 1
                print('There are no public clients be sampled in this round.')
                participants = list(np.random.permutation(candidates))[0:m]

            return participants if check else []
        
