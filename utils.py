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
            sorted_eps = np.sort(epsilons)

            percent = 0.1
            threshold = sorted_eps[-int(percent * N)]

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
            print(x_max)
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
    

def global_step_creator():
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='global_step_placeholder')
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return increase_global_step, set_global_step

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

def Assignements(dic):
    return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]


def Vname_to_Pname(var):
    return var.name[:var.name.find(':')] + '_placeholder'


def Vname_to_FeedPname(var):
    return var.name[:var.name.find(':')] + '_placeholder:0'


def Vname_to_Vname(var):
    return var.name[:var.name.find(':')]


def create_save_dir(FLAGS):
    '''
    :return: Returns a path that is used to store training progress; the path also identifies the chosen setup uniquely.
    '''
    raw_directory = FLAGS.save_dir + '/'
    if FLAGS.gm: gm_str = 'Dp/'
    else: gm_str = 'non_Dp/'
    if FLAGS.priv_agent:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Epochs_' + str(
            int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.batch_size))
        return raw_directory + str(model) + '/' + FLAGS.PrivAgentName
    else:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Sigma_' + str(FLAGS.Sigma) + '_C_'+str(FLAGS.m)+'/Epochs_' + str(
            int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.batch_size))
        return raw_directory + str(model)

def initialize(eps, N):
    Accuracy_accountant = []
    Eps_accountant = [0]
    Budgets_accountant = BudgetsAccountant(N, eps)
    model = []

    return model, Eps_accountant, Accuracy_accountant, Budgets_accountant


def save_progress(FLAGS, model, Accuracy_accountant, Budgets_accountant=None):
    '''
    This function saves our progress either in an existing file structure or writes a new file.
    :param save_dir: STRING: The directory where to save the progress.
    :param model: DICTIONARY: The model that we wish to save.
    :param Delta_accountant: LIST: The list of deltas that we allocared so far.
    :param Accuracy_accountant: LIST: The list of accuracies that we allocated so far.
    :param PrivacyAgent: CLASS INSTANCE: The privacy agent that we used (specifically the m's that we used for Federated training.)
    :param FLAGS: CLASS INSTANCE: The FLAGS passed to the learning procedure.
    :return: nothing
    '''
    save_dir = os.path.join(os.getcwd(), 'res_{}'.format(FLAGS.version), FLAGS.dataset, FLAGS.model, ('noniid' if FLAGS.noniid else 'iid'), (FLAGS.eps if FLAGS.dpsgd else 'nodp'))
    '''
    if os.path.exists(dir + '/'+str(num)+'_clients.pkl'):
        print('Client exists at: '+ dir + '/'+str(num)+'_clients.pkl')
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}{}{}{}".format( FLAGS.N, ('-fedavg' if FLAGS.fedavg else ''), ('-wavg' if FLAGS.wei_avg else ''), ('-pro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.projection else ''))
    #nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0131_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0131_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0131_v6 2>&1 &
    '''
    filehandler = open(filename + '.pkl', "wb")
    pickle.dump(model, filehandler)
    filehandler.close()
    '''
    with open(os.path.join(save_dir, filename + '.csv'), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if FLAGS.dpsgd:
            writer.writerow(Budgets_accountant)
        writer.writerow(Accuracy_accountant)


def print_loss_and_accuracy(global_loss, accuracy, stage='validation'):
    print(' - Current Model has a loss of:           %s' % global_loss)
    print(' - The Accuracy on the ' + stage + ' set is: %s' % accuracy)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')


def print_new_comm_round(real_round):
    print('--------------------------------------------------------------------------------------')
    print('------------------------ Communication round %s ---------------------------------------' % str(real_round))
    print('--------------------------------------------------------------------------------------')


