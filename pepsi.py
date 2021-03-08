
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
#import copy
import tensorflow.compat.v1 as tf
import numpy as np
from utils import global_step_creator, sampling, Vname_to_FeedPname, Vname_to_Pname, print_new_comm_round, save_progress, \
    print_loss_and_accuracy, print_new_comm_round
from data_reader import load_dataset
from create_clients import create_iid_clients, create_noniid_clients
from budgets_accountant import BudgetsAccountant
from models import nets
from models.fed import LocalUpdate, ServerAggregation

np.random.seed(10)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)


def run_pepsi():

    accuracy_accountant = []
    # define tensors and operators in the graph 'g_c'
    with tf.Graph().as_default():
        # build model
        if FLAGS.dpsgd:
            gradient_op_list, train_op_list, eval_op, loss, data_placeholder, labels_placeholder = nets.mnist_model(FLAGS, \
                epsilons, noise_multiplier)
        else:
          gradient_op_list, train_op_list, eval_op, loss, data_placeholder, labels_placeholder = nets.mnist_model(FLAGS)

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
        assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) 
                       for var in tf.trainable_variables()]

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            # initial global model and errors
            model = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                            [sess.run(var) for var in tf.trainable_variables()]))
            model['global_step_placeholder:0'] = 0
            errors = list(model.values()) if FLAGS.error_feedback else [0]*len(tf.trainable_variables())
            #server.set_global_model(model)   

            # initial server aggregation
            #w = weights if FLAGS.wei_avg else None
            server = ServerAggregation(model, FLAGS.dpsgd, FLAGS.projection, FLAGS.proj_dims, FLAGS.wei_avg)

            # initial local update
            local = LocalUpdate(x_train, y_train, client_set, FLAGS.client_batch_size, data_placeholder, labels_placeholder)

            for r in range(COMM_ROUND):
                print_new_comm_round(r)

            # select the participating clients
            if FLAGS.dpsgd:
                participating_clients = sampling(FLAGS.N, m, client_set, FLAGS.client_batch_size, \
                                                 FLAGS.sample_mode, budgets_accountant)
            else:
                participating_clients = range(FLAGS.N) # temporary

            # if the condition of training cannot be satisfied. (no public clients or no sufficient candidates.
            if not len(participating_clients):
                print("the condition of training cannot be satisfied. (no public clients or no sufficient candidates.")
                break

            print(participating_clients)
            ############################################################################################################
            # For each client c (out of the m chosen ones):
            for c in participating_clients:
                start_time = time.time()
                #########################################################################################################
                # Start local update
                # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
                #sess.run(assignments, feed_dict=model)
                update = local.update(sess, assignments, c, model, FLAGS.local_steps, train_op_list[c])
                server.aggregate(c, update, is_public = (c in budgets_accountant._public if FLAGS.dpsgd else True))

                if FLAGS.dpsgd:
                    print('For client %d and delta=%f, the budget is %f and the used budget is: %f' %
                          (c, float(FLAGS.delta), epsilons[c], budgets_accountant.get_accumulation(c)))
                    #print('local update procedure time:', time.time() - start_time)
                    # End of the local update
                ############################################################################################################

            # average and update the global model, apply_gradients(grads_and_vars, global_step)
            e = errors if FLAGS.error_feedback else None
            if FLAGS.fedavg:
                n_clients = len(participating_clients)
                w = np.array([1/n_clients] * n_clients)
                print(w)
            elif FLAGS.wei_avg:
                epsSubset = np.array(epsilons)[participating_clients]
                eps_sum = sum(epsSubset)
                w = np.array([eps/eps_sum for eps in epsSubset])
                print(epsSubset, w)
            else:
                w = None

            model = server.fedavg(model, e, w)

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
            '''
            if FLAGS.dpsgd:
                save_progress(FLAGS, model, accuracy_accountant, budgets_accountant.get_global_budget())
            else:
                save_progress(FLAGS, model, accuracy_accountant)
            '''

if __name__ == '__main__':
    app.run(main)

