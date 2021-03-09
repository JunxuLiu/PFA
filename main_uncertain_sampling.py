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
#import copy
import tensorflow.compat.v1 as tf
import numpy as np

import utils.check as check
import utils.dpsgd as dpsgd
import utils.client_sampling as csample

from data_reader import load_dataset
from create_clients import create_iid_clients, create_noniid_clients
from budgets_accountant import BudgetsAccountant
from models import nets
from models.fed import LocalUpdate, ServerAggregation
np.random.seed(10)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

# Experiment hyperparameters
flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'fmnist', 'cifar10'], 'Which dataset to use.')
flags.DEFINE_enum('model', 'cnn', ['lr', 'cnn', '2nn'], 'Which model to use. This '
                'can be a convolutional model (cnn) or a two hidden-layer '
                'densely connected network (2nn).')
flags.DEFINE_boolean('noniid', False, 'If True, train with noniid data distribution.')
flags.DEFINE_integer('noniid_level', 2, 'Classes per client.')
flags.DEFINE_integer('N', 10,
                   'Total number of clients.')
flags.DEFINE_integer('max_steps', 10000,
                   'Total number of communication round.')
flags.DEFINE_integer('client_dataset_size', None,
                   'If None, set the default value.')
flags.DEFINE_integer('client_batch_size', 128,
                   'Batch size used on the client.')
flags.DEFINE_integer('num_microbatches', 64, 'Number of microbatches '
                           '(must evenly divide batch_size)')

# learning rate
flags.DEFINE_enum('lr_mode', 'const', ['const', 'decay'], 'learning rate mode.')
flags.DEFINE_float('lr', 0.1, 'Learning rate for local update procedure.')

# Differential privacy flags
flags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_string('eps', None, 'epsilon file name.')
flags.DEFINE_float(
    'delta', 1e-5, 'Privacy parameter delta'
    'Delta.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('local_steps', 50,
                 'The round gap between two consecutive communications.')

# Personalized privacy flags
flags.DEFINE_enum('sample_mode', None, ['R','W1','W2'], 'Samping mechanism: '
                  'R for random sample, W for weighted sample and None for full participation.')
flags.DEFINE_float('sample_ratio', 0.1, 'Sample ratio.')

# weighted average
flags.DEFINE_boolean('weiavg', False, 'If True, train with weighted averaging.')

# fedavg
flags.DEFINE_boolean('fedavg', False, 'If True, train with fedavg.')

# Projection flags
#flags.DEFINE_enum('projection', 'False', ['True','False','Mixture'], 'Projection mode: '
#                  'Mixture for without projection at formal period and with projection for later period.')
flags.DEFINE_boolean('projection', False, 'If True, use projection.')
flags.DEFINE_integer('proj_dims', 5, 'The dimensions of subspace.')
flags.DEFINE_integer('lanczos_iter', 128, 'Projection method.')
flags.DEFINE_boolean('error_feedback', False, 'If True, use error feedback.')

# save dir flags
flags.DEFINE_integer('version', 1, 'version of dataset.')
#flags.DEFINE_string('save_dir', 'res', 'Model directory')
flags.DEFINE_string('log', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                  'tensorflow/mnist/logs'), 'Log data directory')
FLAGS = flags.FLAGS


def main(unused_argv):

  #print('hello world.')
  #print(flags.fedavg, (re.search('min', flags.eps) or re.search('max', flags.eps)))
  if re.search('min', FLAGS.eps) or re.search('max', FLAGS.eps):
    assert FLAGS.fedavg, 'min or max setting are only applicable for fedavg case.'

  print(FLAGS.model)
  project_path = os.getcwd()

  # load dataset
  x_train, y_train, x_test, y_test = load_dataset(FLAGS.dataset, project_path)
  print('x_train:{} y_train:{} / x_test:{}, y_test:{}'.format(len(x_train), len(y_train), len(x_test), len(y_test)))

  # split data
  client_set_path = os.path.join(project_path, 
                                 'dataset', FLAGS.dataset, \
                                 'clients', \
                                 ('noniid' if FLAGS.noniid else 'iid'), \
                                 'v{}'.format(FLAGS.version))

  #client_set_path = project_path + '/dataset/' + FLAGS.dataset + '/clients/' + ('noniid' if FLAGS.noniid else 'iid')
  client_dataset_size = len(x_train) // FLAGS.N if FLAGS.client_dataset_size is None else FLAGS.client_dataset_size
  if not FLAGS.noniid:
    client_set = create_iid_clients(FLAGS.N, len(x_train), 10, client_dataset_size, client_set_path)
  else:
    client_set = create_noniid_clients(FLAGS.N, len(x_train), 10, client_dataset_size, FLAGS.noniid_level, client_set_path)
  check.check_labels(FLAGS.N, client_set, y_train)
  print('client dataset size: {}'.format(len(client_set[0])))

  COMM_ROUND = int(FLAGS.max_steps / FLAGS.local_steps)
  print('communication rounds:{}'.format(COMM_ROUND))

  # set personalized privacy budgets  
  if FLAGS.dpsgd:
    # simulate the privacy perferences for all clients. `epsilons` is a list with length N.
    # `threshold` is a real value. If a client's privacy preference is larger than threshold, 
    # then this client is seen as a public client to construct the projection matrix.
    epsilons, threshold = dpsgd.set_epsilons(FLAGS.eps, FLAGS.N)
    print('epsilons:{}, \nthreshold:{}'.format(epsilons, threshold))

    # `noise_multiplier` is a parameter in tf_privacy package, which is also the gaussian distribution
    # parameter for random noise.
    noise_multiplier = dpsgd.compute_noise_multipliers(num_clients = FLAGS.N, \
                                                 client_data = client_set,\
                                                 L = FLAGS.client_batch_size,\
                                                 epsilons = epsilons,\
                                                 T = FLAGS.max_steps * FLAGS.sample_ratio,\
                                                 delta = FLAGS.delta)
    print('noise_multiplier:', noise_multiplier)

    # simulate the budget accountant and assign one for each client if dpsgd
    budgets_accountant = BudgetsAccountant(FLAGS.N, epsilons, FLAGS.delta, \
                                           noise_multiplier, \
                                           FLAGS.client_batch_size,
                                           FLAGS.local_steps, \
                                           threshold)
 
  start_time = time.time()

  accuracy_accountant = []
  # define tensors and operators in the graph 'g_c'
  with tf.Graph().as_default():
    # build model
    if FLAGS.dpsgd:
      train_op_list, eval_op, loss, data_placeholder, labels_placeholder = nets.mnist_model(FLAGS, \
                epsilons, noise_multiplier)
    else:
      train_op_list, eval_op, loss, data_placeholder, labels_placeholder = nets.mnist_model(FLAGS)

    # increase and set global step
    increase_global_step, set_global_step = global_step_creator()

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
    assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
                   tf.trainable_variables()]

    #init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

      #sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_variables())

      # initial the server
      if FLAGS.dpsgd:
        server = Server(FLAGS.N, FlAGS.sample_mode, FLAGS.sample_ratio, epsilons)
      else:
        server = Server(FLAGS.N, FlAGS.sample_mode, FLAGS.sample_ratio)

      # initial global model and errors
      model = server.init_global_model()
      server.init_alg(FLAGS.fedavg, FLAGS.weiavg, FLAGS.projection, FLAGS.proj_dims, FLAGS.lanczos_iter)

      # initial local update
      local = LocalUpdate(x_train, y_train, client_set, FLAGS.client_batch_size, data_placeholder, labels_placeholder)

      for r in range(COMM_ROUND):
        print_new_comm_round(r)
        comm_start_time = time.time()
        candidates = budgets_accountant.precheck(N, client_set, FLAGS.client_batch_size)

        # if the condition of training cannot be satisfied. (no public clients or no sufficient candidates.
        if not len(participating_clients):
          print("the condition of training cannot be satisfied. (no public clients or no sufficient candidates.")
          print('Done! The procedure time:', time.time() - start_time)
          break
        print(participating_clients)

        ############################################################################################################
        # For each client c (out of the m chosen ones):
        for c in range(FLAGS.N):
          
          if (budgets_accountant.precheck(c, len(client_set[c])) is False) or \
              (np.ramdom.rand() > FLAGS.sample_ratio):
            continue

          #########################################################################################################
          # Start local update
          # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
          #sess.run(assignments, feed_dict=model)
          update = local.update(sess, assignments, c, model, FLAGS.local_steps, train_op_list[c])
          server.aggregate(c, update, is_public = (c in budgets_accountant._public if FLAGS.projection else True))

          if FLAGS.dpsgd:
            print('For client %d and delta=%f, the budget is %f and the used budget is: %f' %
               (c, float(FLAGS.delta), epsilons[c], budgets_accountant.get_accumulation(c)))
          #print('local update procedure time:', time.time() - start_time)
          # End of the local update
          ############################################################################################################

        # average and update the global model, apply_gradients(grads_and_vars, global_step)
        model = server.fedavg(model, None, w)

        # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
        sess.run(assignments + [increase_global_step], feed_dict=model)

        # validate the (current) global model using validation set.
        # create a feed-dict holding the validation set.
        feed_dict = {str(data_placeholder.name): x_test,
                     str(labels_placeholder.name): y_test}

        # compute the loss on the validation set.
        global_loss = sess.run(loss, feed_dict=feed_dict)
        count = sess.run(eval_op, feed_dict=feed_dict)
        accuracy = float(count) / float(len(y_test))
        accuracy_accountant.append(accuracy)
        print_loss_and_accuracy(global_loss, accuracy, stage='test')
        print('time of one communication:', time.time() - comm_start_time)        
        if FLAGS.dpsgd:
          save_progress(FLAGS, model, accuracy_accountant, budgets_accountant.get_global_budget())
        else:
          save_progress(FLAGS, model, accuracy_accountant)

    print('Done! The procedure time:', time.time() - start_time)

if __name__ == '__main__':

    app.run(main)
