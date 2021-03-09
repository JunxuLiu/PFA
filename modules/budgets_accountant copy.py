import numpy as np
import copy
import math

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

class BudgetsAccountant:
  def __init__(self, N, \
               epsilons, delta, \
               noise_multipliers, \
               dataset_size,\
               batch_size,\
               loc_steps, priv_threshold, \
               accumulation=None):

    self._init = copy.deepcopy(epsilons)
    self._public = None if priv_threshold is None else list(np.where(np.array(self._init) >= priv_threshold)[0])
    self._private = None if self._public is None else list(set(range(N)).difference(set(self._public)))
    self._remainder = copy.deepcopy(epsilons)

    if accumulation is None:
        self._accumulation = [0] * N
    else:
        self._accumulation = accumulation

    self._tmp_accum_bgts = [0] * N
    self._delta = delta
    self._tmp_delta = [0] * N
    self._round = [0] * N
    self.dataset_size = dataset_size
    self._batch_size = batch_size
    self._loc_steps = loc_steps
    self._finished = [False] * N
    self._noise_multipliers = noise_multipliers
    self._global_budgets = []

  def set_finished(self, client_id):
    self._finished[client_id] = True

  def get_finished(self):
    return self._finished

  def precheck(self):
    '''check '''
    # `idx` refers to the clients which have already finished.
    idx = np.where(np.array(self._finished) == False)[0].tolist()
    s = []

    # Then we need to find the clients that will exhaust their budgets in the following round.
    # These clients will also set as 'finished' and are not allowed to participate the rest training.
    # Other clients will be added to `s` and seen as the candidates of the following round.
    for c in idx:
      tmp_steps = self._round[c] + self._loc_steps
      q = self._batch_size * 1.0 / self.dataset_size[c]
      tmp_accum_bgts = 10 * q * math.sqrt(tmp_steps*(-math.log10(self._delta))) / self._noise_multipliers[c]

      if tmp_accum_bgts > self._init[c]:
        self.set_finished(c)

      else:
        s.append(c)
        self._tmp_accum_bgts[c] = tmp_accum_bgts

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
      self._round[c] += self._loc_steps
      self._remainder[c] = self._init[c]-self._tmp_accum[c]
      self._accumulation[c] = self._tmp_accum[c]
      self._tmp_accum[c] = 0


    self.set_global_budget()
    #print('global_budgets:', self._global_budgets)
       
