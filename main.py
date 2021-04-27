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
import time
import re
import tensorflow.compat.v1 as tf
import numpy as np

from modules.cnn import CNN
from modules.logistic_reg import LogisticRegression
from modules.client import Client
from modules.server import Server
from modules.budgets_accountant import BudgetsAccountant

from simulation.datasets import data_reader
from simulation.clients import create_clients

from common_utils import dpsgd_utils, main_utils
from common_utils.tf_utils import global_step_creator, Vname_to_FeedPname, Vname_to_Pname
from modules.hparams import HParams
np.random.seed(10)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

# Experiment hyperparameters
flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'fmnist', 'cifar10'], 
                  'Which dataset to use.')
flags.DEFINE_enum('model', 'cnn', ['lr', 'cnn', '2nn'], 
                'Which model to use. This can be a convolutional model (cnn)'
                'or a two hidden-layer densely connected network (2nn).')
flags.DEFINE_boolean('noniid', False, 'If True, train with noniid data.')
flags.DEFINE_integer('noniid_level', 10, 'Level of noniid.')
flags.DEFINE_integer('N', 10,
                   'Total number of clients.')
flags.DEFINE_integer('max_steps', 10000,
                   'Total number of communication round.')
flags.DEFINE_integer('local_steps', 100,
                   'The round gap between two consecutive communications.')
flags.DEFINE_integer('client_dataset_size', None,
                   'If None, set the default value.')
flags.DEFINE_integer('client_batch_size', 4,
                   'Batch size used on the client.')
flags.DEFINE_integer('num_microbatches', 4, 'Number of microbatches '
                           '(must evenly divide batch_size)')

# learning rate
flags.DEFINE_boolean('lr_decay', False, 'If True, learning rate decays.')
flags.DEFINE_float('lr', 0.1, 'Learning rate for local update procedure.')

# Differential privacy flags
flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. '
                   'If False, train with vanilla SGD.')
flags.DEFINE_string('eps', None, 'epsilon file name.')
flags.DEFINE_float('delta', 1e-5, 'DP parameter Delta.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')

# Personalized privacy flags
flags.DEFINE_enum('sample_mode', 'R', ['R','W1','W2'],
                  'R for random sample, W for weighted sample and '
                  'None for full participation.')
flags.DEFINE_float('sample_ratio', 0.8, 'Sample ratio.')

# minimum epsilon
flags.DEFINE_boolean('min', False, 'If True, train eps_min dp.')
# weighted average
flags.DEFINE_boolean('weiavg', False, 'If True, train with weighted averaging.')
# fedavg
flags.DEFINE_boolean('fedavg', False, 'If True, train with fedavg.')
# Projection flags
flags.DEFINE_boolean('projection', False, 'If True, use projection.')
flags.DEFINE_boolean('proj_wavg', False, 'If True, use the weighted projection.')
flags.DEFINE_integer('proj_dims', 1, 'The dimensions of subspace.')
flags.DEFINE_integer('lanczos_iter', 256, 'Projection method.')
# save dir flags
flags.DEFINE_integer('version', 1, 'version of dataset.')
flags.DEFINE_string('save_dir', 'res', 'Model directory')
flags.DEFINE_string('log', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                  'tensorflow/mnist/logs'), 'Log data directory')
FLAGS = flags.FLAGS


def prepare_local_data(project_path, dataset, nclients, noniid, version):

    # universal set
    x_train, y_train, x_test, y_test = data_reader.load_dataset(project_path, dataset)
    print('x_train:{} y_train:{} / x_test:{}, y_test:{}'.format(\
          len(x_train), len(y_train), len(x_test), len(y_test)))

    # split the universal
    client_set_path = os.path.join(project_path, 'dataset', dataset, 'clients', 
                                  ('noniid' if noniid else 'iid'), 
                                  'v{}'.format(version))

    client_dataset_size = len(x_train) // nclients if FLAGS.client_dataset_size is None \
                          else FLAGS.client_dataset_size
    if not noniid:
        client_set = create_clients.create_iid_clients(nclients, len(x_train), 10,
                                      client_dataset_size, client_set_path)
    else:
        client_set = create_clients.create_noniid_clients(nclients, len(x_train), 10, 
                                      client_dataset_size, FLAGS.noniid_level, client_set_path)

    labels = [0]*10
    for i in y_train:
        labels[int(i)] += 1

    return x_train, y_train, x_test, y_test, client_set


def prepare_priv_preferences(epsfile, num_clients):

    epsilons = None
    if FLAGS.dpsgd:
        epsilons = dpsgd_utils.set_epsilons(epsfile, num_clients)
    return epsilons

def main(unused_argv):

    hp = HParams(loc_batch_size=FLAGS.client_batch_size, 
                loc_num_microbatches=FLAGS.num_microbatches, 
                loc_lr=FLAGS.lr,
                glob_steps=FLAGS.max_steps,
                loc_steps=FLAGS.local_steps,
                loc_l2_norm=FLAGS.l2_norm_clip)

    project_path = os.getcwd()
    # prepare the local dataset all clients
    x_train, y_train, x_test, y_test, client_set = \
            prepare_local_data(project_path, FLAGS.dataset, FLAGS.N, FLAGS.noniid, FLAGS.version)
  
    create_clients.check_labels(FLAGS.N, client_set, y_train) # print and check
    print('client dataset size: {}'.format(len(client_set[0])))

    # Prepare all clients (simulation)
    # simulate a list of the personal privacy preferences of all clients
    # If FLAGS.dpsgd is False, `prepare_priv_preferences` return None 
    # otherwise return a list of epsilon with size FLAGS.N
    priv_preferences = prepare_priv_preferences(FLAGS.eps, FLAGS.N)
    print('priv_preferences: {}'.format(priv_preferences))

    clients = []
    for cid in range(FLAGS.N):
        print(client_set[cid])
        idx = [int(val) for val in client_set[cid]]
        client = Client(x_train=x_train[idx],
                        y_train=y_train[idx],
                        batch_size=hp.bs, # batch_size
                        loc_steps=hp.loc_steps) # learning_rate
    
        if FLAGS.dpsgd:
            # prepare the dpsgd params for client #c
            # `noise_multiplier` is a parameter in tf_privacy package, which is also the gaussian distribution parameter for random noise.
            epsilon = priv_preferences[cid]
            delta = FLAGS.delta
            noise_multiplier = dpsgd_utils.compute_noise_multiplier(N=client.dataset_size,
                                                        L=hp.bs,
                                                        T=hp.glob_steps * FLAGS.sample_ratio,
                                                        epsilon=epsilon,
                                                        delta=delta)
            
            ba = BudgetsAccountant(epsilon, delta, noise_multiplier)
            client.set_ba(ba)

        clients.append(client)
  
    # Prepare server (simulation)
    server = Server(FLAGS.N, FLAGS.sample_mode, FLAGS.sample_ratio)
    if FLAGS.projection or FLAGS.proj_wavg:
        server.set_public_clients(priv_preferences) 

    # pre-define the number of server-clients communication rounds
    COMM_ROUND = hp.glob_steps // hp.loc_steps
    print('communication rounds:{}'.format(COMM_ROUND))

    # record the test accuracy of the training process.
    accuracy_accountant = []
    privacy_accountant = []
    start_time = time.time()

    # define tensors and operators in the graph 'g_c'
    with tf.Graph().as_default():
        # build model
        if FLAGS.model == 'lr':
            model = LogisticRegression(FLAGS.dataset, FLAGS.client_batch_size, FLAGS.lr, FLAGS.lr_decay)
        elif FLAGS.model =='cnn':
            model = CNN(FLAGS.dataset, FLAGS.client_batch_size, FLAGS.lr, FLAGS.lr_decay)
        else:
            raise ValueError('No avaliable class in `./modules` matches the required model.')

        if FLAGS.dpsgd:
            model.set_dpsgd_params(l2_norm_clip = FLAGS.l2_norm_clip,
                                num_microbatches = FLAGS.num_microbatches,
                                noise_multipliers = [clients[cid].ba.noise_multiplier for cid in range(FLAGS.N)])
        
        # build the model on the server side
        train_op_list, eval_op, loss, global_steps, data_placeholder, labels_placeholder = model.get_model(FLAGS.N)
        # clients download the model from server
        for cid in range(FLAGS.N):
            clients[cid].set_ops( train_op_list[cid], eval_op, loss, 
                                data_placeholder, labels_placeholder )

        # increase and set global step
        real_global_steps = 0
        set_global_step = global_step_creator()

        # dict, each key-value pair corresponds to the placeholder_name of each tf.trainable_variables
        # and its placeholder.
        # trainable_variables: the placeholder name corresponding to each tf.trainable variable.
        model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                            [tf.placeholder(name=Vname_to_Pname(var),
                                            shape=var.get_shape(),
                                            dtype=tf.float32)
                            for var in tf.trainable_variables()]))

        # all trainable variables are set to the value specified through
        # the placeholders in 'model_placeholder'.
        assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)])\
                       for var in tf.trainable_variables()]

        with tf.Session(config = tf.ConfigProto(log_device_placement=False,
                                                allow_soft_placement=True,
                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            #sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_all_variables())

            # initial global model and errors
            model = server.init_global_model(sess)
            alg = server.init_alg(FLAGS.dpsgd,
                                FLAGS.fedavg, 
                                FLAGS.weiavg, 
                                FLAGS.projection, 
                                FLAGS.proj_wavg,
                                FLAGS.proj_dims, 
                                FLAGS.lanczos_iter)

            # initial local update
            #local = LocalUpdate(x_train, y_train, client_set, hp.bs, data_placeholder, labels_placeholder)

            for r in range(COMM_ROUND):
                main_utils.print_new_comm_round(r)
                comm_start_time = time.time()
                # precheck and pick up the candidates who can take the next commiunication round.
                candidates = [ cid for cid in range(FLAGS.N) if clients[cid].precheck() ]
                # select the participating clients
                participants = server.sample_clients(candidates)
                # if the condition of training cannot be satisfied. 
                # (no public clients or no sufficient candidates.
                if len(participants) == 0:
                    print("the condition of training cannot be satisfied. (no public clients or no sufficient candidates.")
                    print('Done! The procedure time:', time.time() - start_time)
                    break
                
                print('==== participants in round {} includes: ====\n {} '.format(r, participants))
                max_accum_bgts = 0
                #####################################################
                # For each client c (out of the m chosen ones):
                for cid in participants:
                    #####################################################
                    # Start local update
                    # 1. Simulate that clients download the global model from server.
                    # in here, we set the trainable Variables in the graph to the values stored in feed_dict 'model'
                    clients[cid].download_model(sess, assignments, set_global_step, model)

                    # 2. clients update the model locally
                    update, accum_bgts = clients[cid].local_update(sess, model, global_steps)

                    if accum_bgts is not None:
                        max_accum_bgts = max(max_accum_bgts, accum_bgts)
                    
                    server.aggregate(cid, update, FLAGS.projection, FLAGS.proj_wavg)

                    if FLAGS.dpsgd:
                        print('For client %d and delta=%f, the budget is %f and the left budget is: %f' %
                            (cid, delta, clients[cid].ba.epsilon, clients[cid].ba.accum_bgts))

                    # End of the local update
                    #####################################################

                # average and update the global model
                model = server.update( model, eps_list=(priv_preferences[participants] if FLAGS.weiavg else None) )
                if FLAGS.projection and FLAGS.delayed:
                    Vk, mean = server.get_proj_info()

                # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
                sess.run(assignments, feed_dict=model)

                # validate the (current) global model using validation set.
                # create a feed-dict holding the validation set.
                feed_dict = {str(data_placeholder.name): x_test,
                            str(labels_placeholder.name): y_test}

                # compute the loss on the validation set.
                global_loss = sess.run(loss, feed_dict=feed_dict)
                count = sess.run(eval_op, feed_dict=feed_dict)
                accuracy = float(count) / float(len(y_test))
                accuracy_accountant.append(accuracy)

                if FLAGS.dpsgd:
                    privacy_accountant.append(max_accum_bgts)
                    main_utils.save_progress(FLAGS, model, accuracy_accountant, privacy_accountant)
                else:
                    main_utils.save_progress(FLAGS, model, accuracy_accountant)

                main_utils.print_loss_and_accuracy(global_loss, accuracy, stage='test')
                print('time of one communication:', time.time() - comm_start_time)
              
    print('Done! The procedure time:', time.time() - start_time)

if __name__ == '__main__':

    app.run(main)
