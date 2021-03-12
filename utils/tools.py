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

def check_labels(N, client_set, y_train):
    labels_set = []
    for cid in range(N):
        idx = [int(val) for val in client_set[cid]]
        labels_set.append(set(np.array(y_train)[idx]))

        labels_count = [0]*10
        for label in np.array(y_train)[idx]:
            labels_count[int(label)] += 1
        print('cid: {}, number of labels: {}/10.'.format(cid, len(labels_set[cid])))
        print(labels_count)


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
    print('------------------>',FLAGS.noniid_level)
    save_dir = os.path.join(os.getcwd(), 'res_{}'.format(FLAGS.version), FLAGS.dataset, FLAGS.model, ('noniid{}'.format(FLAGS.noniid_level) if FLAGS.noniid else 'iid'), (FLAGS.eps if FLAGS.dpsgd else 'nodp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}{}{}{}{}{}".format( FLAGS.N, 
                                    ('-fedavg' if FLAGS.fedavg else ''), 
                                    ('-wavg' if FLAGS.weiavg else ''), 
                                    ('-pro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.projection else ''),
                                    '-bs{}'.format(FLAGS.client_batch_size), 
                                    ('-decaylr' if FLAGS.lr_decay else '-constlr') )

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


