
"""
Non projection component.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import pickle
import math
import heapq
#import copy
import tensorflow.compat.v1 as tf
from tensorflow.keras import datasets

import numpy as np
from utils import global_step_creator, sampling, Vname_to_FeedPname, Vname_to_Pname, print_new_comm_round, save_progress, \
    print_loss_and_accuracy, print_new_comm_round
from data_reader import load_dataset
from create_clients import create_iid_clients
from budgets_accountant import BudgetsAccountant
from models.nets_keras import get_model
from models.fed_keras import LocalUpdate, ServerAggregation

np.random.seed(10)

flags.DEFINE_enum('dataset', 'cifar10', ['mnist', 'cifar10'], 'Which dataset to use.')
flags.DEFINE_enum('model', 'cnn', ['lr', 'cnn', '2nn', 'resnet'], 'Which model to use.')

# fl
flags.DEFINE_integer('N', 1, 'Total number of clients.')
flags.DEFINE_boolean('noniid', False, 'If True, train with noniid data distribution.')
flags.DEFINE_integer('loc_steps', 20, 'Number of local training steps of each client.')
flags.DEFINE_integer('batch_size', 250,
                   'Batch size used on the client.')
flags.DEFINE_integer('num_microbatches', None, 'Number of microbatches '
                           '(must evenly divide batch_size)')

# dpsgd
flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD.')
flags.DEFINE_float('eps', 1.0, 'If None, eps is the default value.')
flags.DEFINE_float('nm', 1.1, 'Noise multiplier.')

# projection
flags.DEFINE_boolean('projection', False, 'If True, use projection.')
flags.DEFINE_integer('proj_dims', 5,
                   'The dimensions of subspace.')
flags.DEFINE_boolean('error_feedback', False, 'If True, use error feedback.')


# learning rate
flags.DEFINE_enum('lr_mode', 'const', ['const', 'decay'], 'learning rate mode.')
flags.DEFINE_float('lr', 0.1, 'Learning rate for local update procedure.')

FLAGS = flags.FLAGS


def _test_dpsgd_cnn():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    noise_multiplier = [ FLAGS.nm ] * FLAGS.N
    model, _ = get_model(FLAGS, noise_multiplier)
    w = model.get_weights()
    #print('old model structure: ', len(old_model), old_model[0].shape)
    #print(old_model[0])   
    history = model.fit(x_train, y_train, epochs=1, batch_size=200)
    w = model.get_weights()
    #print('new model structure: ', len(new_model), new_model[0].shape)
    #print(new_model[0])
    results = model.evaluate(x_test, y_test, batch_size=200)
    print('test loss/acc: ', results)

    history = model.fit(x_train, y_train, epochs=1, batch_size=200)
    results = model.evaluate(x_test, y_test, batch_size=200)
    print('test loss/acc: ', results)


def _test_dpfl_cnn():

    project_path = os.getcwd()
    # load dataset
    x_train, y_train, x_test, y_test = load_dataset(FLAGS.dataset, project_path)
    print(len(x_train), x_train[0].shape, len(x_test), x_test[0].shape)

    # split data
    client_set_path = os.path.join(project_path, 'dataset', FLAGS.dataset, 'clients', ('noniid' if FLAGS.noniid else 'iid'))
    client_set = create_iid_clients(FLAGS.N, len(x_train), 10, client_set_path)

    noise_multiplier = [FLAGS.nm] * FLAGS.N
    model, glob_cls, loc_cls, loss = get_model(FLAGS, noise_multiplier)
    model.compile(optimizer=glob_cls, loss=loss, metrics=['accuracy'])
    glob_w = glob_cls.get_weights()

    # initial server aggregation
    server = ServerAggregation(glob_w, FLAGS.dpsgd, FLAGS.projection, FLAGS.proj_dims)
    # initial local update
    local = LocalUpdate(x_train, y_train, client_set, FLAGS.batch_size)

    COMM_ROUND = 10
    for r in range(COMM_ROUND):
        print_new_comm_round(r)

        for c in range(FLAGS.N):

            update = local.update(c, model, loc_cls[c], loss, FLAGS.loc_steps, glob_w)
            server.aggregate(update, is_public = (epsilons[c] >= 10 if FLAGS.dpsgd else True))         

        # average and update the global model
        glob_w = server.fedavg(glob_w)
        glob_cls.set_weights(glob_w)
        results = glob_cls.evaluate(x_vali, y_vali)
        #results = server.evaluate(x_vali, y_vali, glob_cls)
        print('client {}\'s validate loss/acc: {}'.format(c, results))

    glob_cls.set_weights(glob_w)
    results = glob_cls.evaluate(x_test, y_test, batch_size=FLAGS.batch_size)
    #results = server.evaluate(x_test, y_test, glob_cls)
    print('test loss/acc: {}'.format(results))


def main(unused_argv):
    _test_dpsgd_cnn()
    #_test_dpfl_cnn()


if __name__ == '__main__':
    app.run(main)


