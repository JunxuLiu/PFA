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
"""
def set_epsilons(filename, N, is_distributions = True):
    print('=========Epsilons Info========')
    with open('epsfiles/{}.txt'.format(filename), 'r') as rfile:
        lines = rfile.readlines()
        num_lines = len(lines)

        
        if re.search('mixgauss', filename):
            print('{} is a mix gaussian distribution.'.format(filename))
            dists = []
            for i in range(num_lines-2):
                print(lines[i])
                values = lines[i].split()
                dist = {'mean':float(values[1]), 'std':float(values[2])}
                dists.append(dist)

            threshold = float(lines[num_lines-2].split()[1])
            pr_dist = [ float(x) for x in lines[num_lines-1].split()[1:] ]
            print('pr_list:{}, threshold:{}'.format(pr_dist, threshold))
            
            while(True):
                epsilons = []
                for i in range(N):
                    dist_idx = np.argmax(np.random.multinomial(1, pr_dist))
                    eps = np.random.normal(dists[dist_idx]['mean'], dists[dist_idx]['std'])
                    #print(dist_idx,': ', eps)
                    epsilons.append(eps)

                epsilons = np.array(epsilons)
                if (len( epsilons [epsilons > threshold] ) > 0) :
                    break

        elif re.search('gauss', filename):
            print('{} is a gaussian distribution.'.format(filename))
            values = lines[0].split()
            dist = {'mean':float(values[1]), 'std':float(values[2])}
            epsilons = np.random.normal(dist['mean'], dist['std'], N)
            

        elif re.search('uniform', filename):
            print('{} is a uniform distribution.'.format(filename))
            values = lines[0].split()[1:]
            _min, _max = float(values[0]), float(values[1])
            epsilons = np.random.uniform(_min, _max, N)
            threshold = float(lines[1].split()[1])

            while len( epsilons [epsilons > threshold] ) == 0:
                epsilons = np.random.uniform(_min, _max, N)
                if len( epsilons [epsilons > threshold] ) > 0:
                    break

        elif re.search('pareto', filename):
            print('{} is a pareto distribution.'.format(filename))
            x_m, alpha = float(lines[0].split()[1]), float(lines[0].split()[2])
            print(x_m, alpha)
            epsilons = (np.random.pareto(alpha, N) + 1) * x_m
            #threshold = np.sort(epsilons)[::-1][int(N*0.2)-1]
            threshold = 2 if N == 10 else 5
        
        elif re.search('min', filename):
            print('{} take the minimum value over all clients\' preferences.'.format(filename))
            x_min = float(lines[0].split()[1])
            print(x_min)
            epsilons = [x_min] * N
            threshold = None

        elif re.search('max', filename):
            print('{} take the maximum value over all clients\' preferences.'.format(filename))
            x_max = float(lines[0].split()[1])
            epsilons = [x_max] * N
            threshold = None

        else:
            '''or you can directly provide the exact epsilons of each clients. Note that the total number
             of epsilons should be equal to the number of clients N.

             #format:
             epsilons 0.5 0.5 0.5 0.5 ... (total N values)
             threshold 1.0
            '''
            print('{} is not a distribution.'.format(filename))
            values = lines[0].split()[1:]
            epsilons = [float(v) for v in values]
            threshold = float(lines[1][1]) 

    print('epsilons:{}, total {} values.'.format(epsilons, len(epsilons)))
    return epsilons, threshold    
"""

def set_epsilons(filename, N, is_distributions = True):

    print('=========Epsilons Info========')
    with open('epsfiles/{}.txt'.format(filename), 'r') as rfile:
        lines = rfile.readlines()
        num_lines = len(lines)

        if re.search('mixgauss', filename):
            print('{} is a mix gaussian distribution.'.format(filename))
            dists = []
            for i in range(num_lines-2):
                print(lines[i])
                values = lines[i].split()
                dist = {'mean':float(values[1]), 'std':float(values[2])}
                dists.append(dist)

            threshold = float(lines[-1].split()[1])
            pr_dist = [ float(x) for x in lines[-2].split()[1:] ]
            print('pr_list:{}, threshold:{}'.format(pr_dist, threshold))
            
            while(True):
                epsilons = []
                for i in range(N):
                    dist_idx = np.argmax(np.random.multinomial(1, pr_dist))
                    eps = np.random.normal(dists[dist_idx]['mean'], dists[dist_idx]['std'])
                    epsilons.append(eps)

                epsilons = np.array(epsilons)
                if (len( epsilons [epsilons > threshold] ) > 0) :
                    break


        elif re.search('gauss', filename):
            print('{} is a gaussian distribution.'.format(filename))
            values = lines[0].split()
            dist = {'mean':float(values[1]), 'std':float(values[2])}
            epsilons = np.random.normal(dist['mean'], dist['std'], N)

            threshold = float(lines[-1].split()[1])

        elif re.search('uniform', filename):
            print('{} is a uniform distribution.'.format(filename))
            values = lines[0].split()[1:]
            _min, _max = float(values[0]), float(values[1])
            epsilons = np.random.uniform(_min, _max, N)
            threshold = float(lines[-1].split()[1])

            while len( epsilons [epsilons > threshold] ) == 0:
                epsilons = np.random.uniform(_min, _max, N)
                if len( epsilons [epsilons > threshold] ) > 0:
                    break

        elif re.search('pareto', filename):
            print('{} is a pareto distribution.'.format(filename))
            x_m, alpha = float(lines[0].split()[1]), float(lines[0].split()[2])
            print(x_m, alpha)
            epsilons = (np.random.pareto(alpha, N) + 1) * x_m
            #threshold = np.sort(epsilons)[::-1][int(N*0.2)-1]
            threshold = 2 if N == 10 else 5
        
        elif re.search('min', filename):
            print('{} take the minimum value over all clients\' preferences.'.format(filename))
            x_min = float(lines[0].split()[1])
            print(x_min)
            epsilons = [x_min] * N
            threshold = None

        elif re.search('max', filename):
            print('{} take the maximum value over all clients\' preferences.'.format(filename))
            x_max = float(lines[0].split()[1])
            epsilons = [x_max] * N
            threshold = None

        else:
            '''or you can directly provide the exact epsilons of each clients. Note that the total number
             of epsilons should be equal to the number of clients N.

             #format:
             epsilons 0.5 0.5 0.5 0.5 ... (total N values)
             threshold 1.0
            '''
            print('{} is not a distribution.'.format(filename))
            values = lines[0].split()[1:]
            epsilons = [float(v) for v in values]
            threshold = float(lines[1][1]) 

    print('epsilons:{}, total {} values.'.format(epsilons, len(epsilons)))
    return epsilons    



def compute_noise_multiplier( N, L, T, epsilon, delta):
    q = L / N
    nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm
