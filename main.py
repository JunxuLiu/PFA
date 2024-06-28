import argparse
import copy
import datetime
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
from myopacus import PrivacyEngine
from myopacus.accountants.pdp_utils import GENERATE_EPSILONS_FUNC
from strategies import FedAvg, PFA

# from modules.cnn import CNN
# from modules.logistic_reg import LogisticRegression
# from modules.client import Client
# from modules.server import Server
# from modules.budgets_accountant import BudgetsAccountant

# from simulation.datasets import data_reader
# from simulation.clients import create_clients

from common_utils import dpsgd_utils, main_utils
# from common_utils.tf_utils import global_step_creator, Vname_to_FeedPname, Vname_to_Pname
# from modules.hparams import HParams
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='mnist') # ['mnist', 'fmnist', 'cifar10']
parser.add_argument("--method", type=str, default='pfa') # ['pfa', 'pfa+', 'weiavg']
parser.add_argument("--gpuid", type=int, default=7, help="Index of the GPU device.")
parser.add_argument("--seed", type=int, default=41, help="random seed")
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

LR = dict["dpfedavg"]["learning_rate"]
NOISE_MULTIPLIER = dict["dpfedavg"]["noise_multiplier"]
MAX_GRAD_NORM = dict["dpfedavg"]["max_grad_norm"]
TARGET_DELTA = dict["dpfedavg"]["target_delta"]
MAX_PHYSICAL_BATCH_SIZE = dict["dpfedavg"]["max_physical_batch_size"]

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

""" Prepare model and loss """
# We set model and dataloaders to be the same for each rep
global_init = BaselineModel.to(device)
criterion = BaselineLoss()

training_args = {
    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "loss": criterion,
    "optimizer_class": Optimizer,
    "learning_rate": LR,
    "batch_size": SAMPLE_RATE,
    "num_steps": NUM_STEPS,
    "num_rounds": NUM_ROUNDS,
    "client_rate": CLIENT_RATE,
    "device": device,
    "metric": metric,
    "seed": args.seed
}

""" Prepare privacy budgets """
# simulate a list of the personal privacy preferences of all clients
# different distributions & different settings
SETTINGS = dict["rpdpfedavg"]["settings"]
MIN_EPSILON, MAX_EPSILON = dict["rpdpfedavg"]["min_epsilon"], dict["rpdpfedavg"]["max_epsilon"]
BoundedFunc = lambda values: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in values])
name, p_id = "BoundedMixGauss", 0
target_epsilons = np.array(
    BoundedFunc(GENERATE_EPSILONS_FUNC[name](NUM_CLIENTS, SETTINGS[name][int(p_id)]))
)
print('target epsilons: {}'.format(target_epsilons))

# Run PFA with cPDP
privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
privacy_engine.prepare_cpdpfl(
    num_steps = NUM_STEPS,
    num_rounds = NUM_ROUNDS,
    client_rate = CLIENT_RATE,
    target_epsilons = target_epsilons,
    target_delta = TARGET_DELTA,
    noise_multiplier = NOISE_MULTIPLIER,
    max_grad_norm = MAX_GRAD_NORM,
    max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
)
current_args = copy.deepcopy(training_args)
current_args["model"] = copy.deepcopy(global_init)
current_args["privacy_engine"] = privacy_engine

s = PFA(**current_args, log=False)
cm, perf = s.run()
mean_perf = np.mean(perf[-3:])
expected_batch_size = [int(sum(acct.sample_rate)) for acct in s.privacy_engine.accountant.accountants]

print(f"Mean performance, min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}, seed={args.seed}")
results_dict = [{
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "mean_perf": round(mean_perf,4), "perf": perf, 
    "e": f"Ours", 
    "d": TARGET_DELTA, 
    "nm": round(s.privacy_engine.default_noise_multiplier, 2), 
    "norm": MAX_GRAD_NORM, 
    "bs": expected_batch_size, 
    "lr": LR,
    "num_clients": NUM_CLIENTS,
    "client_rate": CLIENT_RATE}]
results = pd.DataFrame.from_dict(results_dict)

results.to_csv(save_filename, mode='a', index=False)

# create_clients.check_labels(NUM_CLIENTS, client_set, y_train) # print and check
# print('client dataset size: {}'.format(len(client_set[0])))

# clients = []
# for cid in range(NUM_CLIENTS):
#     idx = [int(val) for val in client_set[cid]]
#     client = Client(x_train=x_train[idx],
#                     y_train=y_train[idx],
#                     batch_size=hp.bs, # batch_size
#                     loc_steps=hp.loc_steps) # learning_rate

#     # prepare the dpsgd params for client #c
#     # `noise_multiplier` is a parameter in tf_privacy package, which is also the gaussian distribution parameter for random noise.
#     epsilon = priv_preferences[cid]
#     delta = FLAGS.delta
#     noise_multiplier = dpsgd_utils.compute_noise_multiplier(N=client.dataset_size,
#                                                         L=hp.bs,
#                                                         T=hp.glob_steps * FLAGS.sample_ratio,
#                                                         epsilon=epsilon,
#                                                         delta=delta)
#     ba = BudgetsAccountant(epsilon, delta, noise_multiplier)
#     client.set_ba(ba)
#     clients.append(client)

# # Prepare server (simulation)
# server = Server(NUM_CLIENTS, FLAGS.sample_mode, FLAGS.sample_ratio)
# if FLAGS.projection or FLAGS.proj_wavg:
#     server.set_public_clients(priv_preferences)

# # record the test accuracy of the training process.
# accuracy_accountant = []
# privacy_accountant = []
# start_time = time.time()

# model.set_dpsgd_params(l2_norm_clip = MAX_GRAD_NORM,
#                     num_microbatches = FLAGS.num_microbatches,
#                     noise_multipliers = [ clients[cid].ba.noise_multiplier for cid in range(NUM_CLIENTS) ] )

# # build the model on the server side
# train_op_list, eval_op, loss, global_steps, data_placeholder, labels_placeholder = model.get_model(NUM_CLIENTS)
# # clients download the model from server
# for cid in range(NUM_CLIENTS):
#     clients[cid].set_ops( train_op_list[cid], eval_op, loss, data_placeholder, labels_placeholder )

# # dict, each key-value pair corresponds to the placeholder_name of each tf.trainable_variables
# # and its placeholder.
# # trainable_variables: the placeholder name corresponding to each tf.trainable variable.
# model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
#                     [tf.placeholder(name=Vname_to_Pname(var),
#                                     shape=var.get_shape(),
#                                     dtype=tf.float32)
#                     for var in tf.trainable_variables()]))

# # all trainable variables are set to the value specified through
# # the placeholders in 'model_placeholder'.
# assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)])\
#                 for var in tf.trainable_variables()]

# # initial global model and errors
# alg = server.init_alg(FLAGS.dpsgd,
#                     FLAGS.fedavg, 
#                     FLAGS.weiavg, 
#                     FLAGS.projection, 
#                     FLAGS.proj_wavg,
#                     FLAGS.delay,
#                     FLAGS.proj_dims, 
#                     FLAGS.lanczos_iter)

# Vk, mean = None, None
# accum_nbytes1 = 0 # before pfaplus
# accum_nbytes2 = 0 # after pfaplus
# accum_nbytes_list1 = []
# accum_nbytes_list2 = []

# # initial local update
# #local = LocalUpdate(x_train, y_train, client_set, hp.bs, data_placeholder, labels_placeholder)

# for r in range(NUM_ROUNDS):
#     main_utils.print_new_comm_round(r)
#     comm_start_time = time.time()
#     # precheck and pick up the candidates who can take the next commiunication round.
#     candidates = [ cid for cid in range(NUM_CLIENTS) if clients[cid].precheck() ]
#     # select the participating clients
#     participants = server.sample_clients(candidates)
#     # if the condition of training cannot be satisfied. 
#     # (no public clients or no sufficient candidates.
#     if len(participants) == 0:
#         print("the condition of training cannot be satisfied. (no public clients or no sufficient candidates.")
#         print('Done! The procedure time:', time.time() - start_time)
#         break
    
#     print('==== participants in round {} includes: ====\n {} '.format(r, participants))
#     max_accum_bgts = 0
#     #####################################################
#     # For each client c (out of the m chosen ones):
#     for cid in participants:
#         #####################################################
#         # Start local update
#         # 1. Simulate that clients download the global model from server.
#         # in here, we set the trainable Variables in the graph to the values stored in feed_dict 'model'
#         clients[cid].download_model(assignments, set_global_step, model)
#         if Vk is not None:
#             clients[cid].set_projection(Vk, mean, is_private=(cid not in server.public))

#         #print(model['dense_1/bias_placeholder:0'])
#         # 2. clients update the model locally
#         update, accum_bgts, bytes1, bytes2 = clients[cid].local_update(model, global_steps)
#         accum_nbytes1 += (bytes1)/(1024*1024)
#         accum_nbytes2 += (bytes2)/(1024*1024)

#         if accum_bgts is not None:
#             max_accum_bgts = max(max_accum_bgts, accum_bgts)
        
#         server.aggregate(cid, update, FLAGS.projection, FLAGS.proj_wavg)

#         print('For client %d and delta=%f, the budget is %f and the left budget is: %f' %
#             (cid, delta, clients[cid].ba.epsilon, clients[cid].ba.accum_bgts))

#         # End of the local update
#         #####################################################

#     # average and update the global model
#     model = server.update( model, eps_list=(priv_preferences[participants] if FLAGS.weiavg else None) )
#     if args.method in ['pfa', 'pfa+']:
#         Vk, mean = server.get_proj_info()

#     # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
#     sess.run(assignments, feed_dict=model)

#     # validate the (current) global model using validation set.
#     # create a feed-dict holding the validation set.
#     feed_dict = {str(data_placeholder.name): x_test,
#                 str(labels_placeholder.name): y_test}

#     # compute the loss on the validation set.
#     global_loss = sess.run(loss, feed_dict=feed_dict)
#     count = sess.run(eval_op, feed_dict=feed_dict)
#     accuracy = float(count) / float(len(y_test))
#     accuracy_accountant.append(accuracy)
    

#     privacy_accountant.append(max_accum_bgts)
#     if args.method == 'pfa+':
#         accum_nbytes_list1.append(accum_nbytes1)
#         accum_nbytes_list2.append(accum_nbytes2)
#     #     main_utils.save_progress(FLAGS, model, accuracy_accountant, privacy_accountant, accum_nbytes_list1, accum_nbytes_list2)
#     # else:
#     #     main_utils.save_progress(FLAGS, model, accuracy_accountant, privacy_accountant)
#     main_utils.print_loss_and_accuracy(global_loss, accuracy, stage='test')
#     print('time of one communication:', time.time() - comm_start_time)
        
# print('Done! The procedure time:', time.time() - start_time)