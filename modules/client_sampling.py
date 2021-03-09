from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import pickle
import math
import heapq
import csv
import re
#import copy
import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(10)

def _a_res(items, weights, m):
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


def _naive_weighted_sampling(items, weights, m):
    weights = np.array(weights) / max(weights)
    samples = [ item for item in items if np.random.random() <= weights[item] ][0:min(m, len(items))]
    return samples


def _top_k(items, weights, m):
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


def sampling(N, client_set, client_batch_size, ratio, mode=None, ba=None):
    # Randomly choose a total of m (out of n) client-indices that participate in this round
    # randomly permute a range-list of length n: [1,2,3...n] --> [5,2,7..3]
    m = int(N * ratio)
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


