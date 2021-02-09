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
#import copy
import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(10)

def set_epsilons(filename, N, is_distributions = True):

    print('=========Epsilons Info========')
    epsilons = []

    if is_distributions:
        '''you can assume the personal epsilons follows a mixture of several guassian distributions.

        #format:
        dist1 0.5 0.01
        dist2 10 0.1
        ...(or more)
        threshold 1.0
        prob 0.9 0.1 (num_dists values)

        '''

        with open('epsfiles/{}.txt'.format(filename), 'r') as rfile:
            lines = rfile.readlines()
            num_lines = len(lines)
            dists = []
            pr_dist = []
            for i in range(num_lines-2):
                print(lines[i])
                values = lines[i].split()
                dist = {'mean':float(values[1]), 'std':float(values[2])}
                dists.append(dist)

            threshold = float(lines[num_lines-2].split()[1])
            pr_dist = [ float(x) for x in lines[num_lines-1].split()[1:] ]
            print('pr_list:{}, threshold:{}'.format(pr_dist, threshold))

        for i in range(N):
            dist_idx = np.argmax(np.random.multinomial(1, pr_dist)) 
            eps = np.random.normal(dists[dist_idx]['mean'], dists[dist_idx]['std'])
            epsilons.append(eps)

    elif filename == 'epsilons':
        '''or you can directly provide the exact epsilons of each clients. Note that the total number
        of epsilons should be equal to the number of clients N.

        #format:
        epsilons 0.5 0.5 0.5 0.5 ... (total N values)
        threshold 1.0

        '''
        with open('epsilons/{}.txt'.format(filename), 'r') as rfile:
            lines = rfile.readlines()
            epsilons = lines[0].split()[1:]
            threshold = float(lines[1][1])

    else:
        '''sample uniformly'''

        epsilons = np.random.uniform(1.0, 10.0, N)
        threshold = 9.0
        while len( epsilons [epsilons > threshold] ) == 0:
            epsilons = np.random.uniform(1.0, 10.0, N)


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


def sampling(N, m, client_set, client_batch_size, mode=None, ba=None):
    # Randomly choose a total of m (out of n) client-indices that participate in this round
    # randomly permute a range-list of length n: [1,2,3...n] --> [5,2,7..3]
    update = lambda x: ba.update(x)

    s = ba.precheck(N, client_set, client_batch_size)
    if len(s) < m:
        return []

    if mode == 'W' and len(s) > m:
        print('mode:',mode)
        remainder = ba.get_remainder()
        print('remainder:', remainder)
        #s = _naive_weighted_sampling(s, remainder, m)
        #s = _a_res(s, remainder, m)
        s_ = _top_k(s, remainder, m) if mode == 'W1' else a_res(s, remainder, m)
        update(s_)

    else:
        print('mode:',mode)
        s_ = list(np.random.permutation(s))[0:m]

        # just resample a vaild subset of clients no more than 5 times
        # we require the subset contains at least 1 public client and 1 private client.
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


