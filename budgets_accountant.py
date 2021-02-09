
import numpy as np
import copy
import math

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

class BudgetsAccountant:
  def __init__(self, N, eps_list, delta, noise_multiplier, comm_gap, priv_threshold, accumulation=None):
    self._init = copy.deepcopy(eps_list)
    self._public = list(np.where(np.array(self._init) >= priv_threshold)[0])
    self._private = list(set(range(N)).difference(set(self._public)))
    self._remainder = copy.deepcopy(eps_list)

    if accumulation is None:
        self._accumulation = [0] * N
    else:
        self._accumulation = accumulation

    self._step_accum = [0] * N
    self._tmp_accum = [0] * N
    self._delta = delta
    self._tmp_delta = [0] * N
    self._round = [0] * N
    self._comm_gap = comm_gap
    self._finished = [False] * N
    self._noise_multiplier = noise_multiplier
    self._global_budgets = []

  def set_finished(self, client_id):
    self._finished[client_id] = True

  def get_finished(self):
    return self._finished

  def precheck(self, N, client_set, batch_size):
    idx = np.where(np.array(self._finished) == False)[0].tolist()
    s = []

    for c in idx:

      tmp_round = self._round[c] + self._comm_gap
#     if self._step_accum[c] == 0:
      '''
      tmp_delta, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            len(client_set[c]), FLAGS.client_batch_size, self._noise_multiplier[c],
            tmp_round * int(FLAGS.client_epochs_per_round), eps=self._init[c]/FLAGS.max_comm_round)
      '''
      '''
      tmp_accum, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            len(client_set[c]), batch_size, self._noise_multiplier[c],
            tmp_round, float(self._delta))
      '''
      q = batch_size*1.0 / len(client_set[c])
#      print(q, tmp_round)
      tmp_accum = 10 * q * math.sqrt(tmp_round*(-math.log10(self._delta))) / self._noise_multiplier[c]
      #tmp_accum = tmp_round * self._step_accum[c]
#      print(c, tmp_accum)
      if tmp_accum > self._init[c]:
        self.set_finished(c)
      else:
        s.append(c)
        self._tmp_accum[c] = tmp_accum
        #self._tmp_accum[c] = (self._init[c]/FLAGS.max_comm_round)*tmp_round
        #self._tmp_delta[c] = self._delta
      #print(s)
    return s

  def get_remainder(self):
    return self._remainder

  def get_accumulation(self, client_id):
    return self._accumulation[client_id]

  def set_global_budget(self):
    self._global_budgets.append(min(self._accumulation))

  def get_global_budget(self):
    return self._global_budgets

  def update(self, clients_id):
    #print('update: ', clients_id) 
    for c in clients_id:
      self._round[c] += self._comm_gap
      self._remainder[c] = self._init[c]-self._tmp_accum[c]
      self._accumulation[c] = self._tmp_accum[c]
      self._tmp_accum[c] = 0


    self.set_global_budget()
    #print('global_budgets:', self._global_budgets)
       
