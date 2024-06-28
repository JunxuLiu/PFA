import argparse
import importlib
import numpy as np
import os
import pandas as pd
import time
import warnings # ignore warnings for clarity
warnings.simplefilter("ignore")

import torch
from torch.utils.data import DataLoader

from configs.config_utils import read_config, get_config_file_path

# from modules.cnn import CNN
# from modules.logistic_reg import LogisticRegression
# from modules.client import Client
# from modules.server import Server
# from modules.budgets_accountant import BudgetsAccountant

# from simulation.datasets import data_reader
# from simulation.clients import create_clients

# from common_utils import dpsgd_utils, main_utils
# from common_utils.tf_utils import global_step_creator, Vname_to_FeedPname, Vname_to_Pname
# from modules.hparams import HParams
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='mnist') # ['mnist', 'fmnist', 'cifar10']
parser.add_argument("--gpuid", type=int, default=7,
                    help="Index of the GPU device.")
parser.add_argument("--seed", type=int, default=41, 
                    help="random seed")
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

module_name = f"datasets.fed_{args.dataset}"
try:
    dataset_modules = importlib.import_module(module_name)
    FedClass = dataset_modules.FedClass
    RawClass = dataset_modules.RawClass
    BaselineModel = dataset_modules.BaselineModel
    BaselineLoss = dataset_modules.BaselineLoss
    Optimizer = dataset_modules.Optimizer
    metric = dataset_modules.metric
    
except ModuleNotFoundError as e:
    print(f'{module_name} import failed: {e}')

project_abspath = os.path.abspath(os.path.join(os.getcwd(),".."))
dict = read_config(get_config_file_path(dataset_name=f"fed_{args.dataset}", debug=False))
# save_dir
save_dir = os.path.join(project_abspath, dict["save_dir"])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_filename = os.path.join(save_dir, f"results_fedavg_rpdp_{args.dataset}_{args.seed}.csv")

NUM_CLIENTS = dict["fedavg"]["num_clients"]
NUM_STEPS = dict["fedavg"]["num_steps"]
NUM_ROUNDS = dict["fedavg"]["num_rounds"]
CLIENT_RATE = dict["fedavg"]["client_rate"]
BATCH_SIZE = dict["fedavg"]["batch_size"]
LR = dict["fedavg"]["learning_rate"]

""" Prepare local datasets """
# data_dir
data_path = os.path.join(project_abspath, dict["dataset_dir"][f"iid_{NUM_CLIENTS}"])
rawdata = RawClass(data_path=data_path)
test_dls, training_dls = [], []
for i in range(NUM_CLIENTS): # NUM_CLIENTS
    train_data = FedClass(rawdata=rawdata, center=i, train=True)
    train_dl = DataLoader(train_data, batch_size=len(train_data))
    training_dls.append(train_dl)

    test_data = FedClass(rawdata=rawdata, center=i, train=False)
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)
    test_dls.append(test_dl)

# def prepare_local_data(project_path, dataset, nclients, noniid, version):
#     data_path = os.path.abspath(os.path.join(project_path,"..","PFA_res","dataset"))
#     print(data_path)
#     # universal set
#     x_train, y_train, x_test, y_test = data_reader.load_dataset(data_path, dataset)
#     print('x_train:{} y_train:{} / x_test:{}, y_test:{}'.format(\
#           len(x_train), len(y_train), len(x_test), len(y_test)))

#     # split the universal
#     client_set_path = os.path.join(data_path, dataset, 'clients', 
#                                   ('noniid' if noniid else 'iid'), 
#                                   'v{}'.format(version))

#     client_dataset_size = len(x_train) // nclients if FLAGS.client_dataset_size is None \
#                           else FLAGS.client_dataset_size
#     if not noniid:
#         client_set = create_clients.create_iid_clients(nclients, len(x_train), 10,
#                                     client_dataset_size, client_set_path)
#     else:
#         client_set = create_clients.create_noniid_clients(nclients, len(x_train), 10, 
#                                     client_dataset_size, FLAGS.noniid_level, client_set_path)

#     labels = [0]*10
#     for i in y_train:
#         labels[int(i)] += 1
#     return x_train, y_train, x_test, y_test, client_set

# hp = HParams(loc_batch_size=FLAGS.client_batch_size, 
#             loc_num_microbatches=FLAGS.num_microbatches, 
#             loc_lr=FLAGS.lr,
#             glob_steps=FLAGS.max_steps,
#             loc_steps=FLAGS.local_steps,
#             loc_l2_norm=FLAGS.l2_norm_clip)

# project_path = os.getcwd()
# print(project_path)
# prepare the local dataset all clients
# x_train, y_train, x_test, y_test, client_set = \
#         prepare_local_data(project_path, FLAGS.dataset, FLAGS.N, FLAGS.noniid, FLAGS.version)

create_clients.check_labels(FLAGS.N, client_set, y_train) # print and check
print('client dataset size: {}'.format(len(client_set[0])))
# Prepare all clients (simulation)
# simulate a list of the personal privacy preferences of all clients
# If FLAGS.dpsgd is False, `prepare_priv_preferences` return None 
# otherwise return a list of epsilon with size FLAGS.N
epsilons = None
if FLAGS.dpsgd:
    epsilons = dpsgd_utils.set_epsilons(epsfile, num_clients)
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
                            noise_multipliers = [ clients[cid].ba.noise_multiplier for cid in range(FLAGS.N) ] )
    
    # build the model on the server side
    train_op_list, eval_op, loss, global_steps, data_placeholder, labels_placeholder = model.get_model(FLAGS.N)
    # clients download the model from server
    for cid in range(FLAGS.N):
        clients[cid].set_ops( train_op_list[cid], eval_op, loss, data_placeholder, labels_placeholder )

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
                            FLAGS.delay,
                            FLAGS.proj_dims, 
                            FLAGS.lanczos_iter)
        
        Vk, mean = None, None
        accum_nbytes1 = 0 # before pfaplus
        accum_nbytes2 = 0 # after pfaplus
        accum_nbytes_list1 = []
        accum_nbytes_list2 = []

        # initial local update
        #local = LocalUpdate(x_train, y_train, client_set, hp.bs, data_placeholder, labels_placeholder)

        for r in range(COMM_ROUND):
            main_utils.print_new_comm_round(r)
            comm_start_time = time.time()

            if FLAGS.N == 1:
                for it in range(FLAGS.local_steps):
                    # batch_ind holds the indices of the current batch
                    batch_ind = np.random.permutation(FLAGS.client_dataset_size)[0:FLAGS.client_batch_size]
                    x_batch = clients[0].x_train[[int(j) for j in batch_ind]]
                    y_batch = clients[0].y_train[[int(j) for j in batch_ind]]

                    # Fill a feed dictionary with the actual set of data and labels using the data and labels associated
                    # to the indices stored in batch_ind:
                    feed_dict = {str(data_placeholder.name): x_batch,
                                str(labels_placeholder.name): y_batch}
                    # Run one optimization step.
                    _ = sess.run(train_op_list[0], feed_dict = feed_dict)

                #self.global_steps = sess.run(global_steps)
                weights = [sess.run(var) for var in tf.trainable_variables()]
                keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]
                model = dict(zip(keys, weights))

            else:
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
                    if Vk is not None:
                        clients[cid].set_projection(Vk, mean, is_private=(cid not in server.public))

                    #print(model['dense_1/bias_placeholder:0'])
                    # 2. clients update the model locally
                    update, accum_bgts, bytes1, bytes2 = clients[cid].local_update(sess, model, global_steps)
                    accum_nbytes1 += (bytes1)/(1024*1024)
                    accum_nbytes2 += (bytes2)/(1024*1024)

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
                if (FLAGS.projection or FLAGS.proj_wavg) and FLAGS.delay:
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
                if FLAGS.delay:
                    accum_nbytes_list1.append(accum_nbytes1)
                    accum_nbytes_list2.append(accum_nbytes2)
                    main_utils.save_progress(FLAGS, model, accuracy_accountant, privacy_accountant, accum_nbytes_list1, accum_nbytes_list2)
                else:
                    main_utils.save_progress(FLAGS, model, accuracy_accountant, privacy_accountant)

            else:
                main_utils.save_progress(FLAGS, model, accuracy_accountant)

            main_utils.print_loss_and_accuracy(global_loss, accuracy, stage='test')
            print('time of one communication:', time.time() - comm_start_time)
            
print('Done! The procedure time:', time.time() - start_time)