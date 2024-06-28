from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import numpy as np
np.random.seed(10)

def save_progress(FLAGS, model, Accuracy_accountant, Budgets_accountant=None, nbytes1=None, nbytes2=None):
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
    save_dir = os.path.join(os.getcwd(), FLAGS.save_dir, 'res_{}'.format(FLAGS.version), FLAGS.dataset, FLAGS.model, ('noniid{}'.format(FLAGS.noniid_level) if FLAGS.noniid else 'iid'), (FLAGS.eps if FLAGS.dpsgd else 'nodp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}{}{}{}{}{}{}{}".format( FLAGS.N, 
                                    ('-fedavg' if FLAGS.fedavg else ''), 
                                    ('-wavg' if FLAGS.weiavg else ''), 
                                    ('-pro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.projection else ''),
                                    ('-wpro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.proj_wavg else ''),
                                    ('-plus' if FLAGS.delay else ''),
                                    '-{}-bs{}'.format(FLAGS.local_steps, FLAGS.client_batch_size), 
                                    ('-decaylr{}'.format(FLAGS.lr) if FLAGS.lr_decay else '-constlr{}'.format(FLAGS.lr)) )
                                    
    with open(os.path.join(save_dir, filename + '.csv'), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if FLAGS.dpsgd:
            writer.writerow(Budgets_accountant)
        if FLAGS.delay:
            writer.writerow(nbytes1)
            writer.writerow(nbytes2)
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


