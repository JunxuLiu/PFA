import numpy as np
import copy
import math

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

class BudgetsAccountant:
  def __init__(self, epsilon, delta, noise_multiplier,
               accumulation=0):

    #self._public = None if priv_threshold is None else list(np.where(np.array(self._init) >= priv_threshold)[0])
    #self._private = None if self._public is None else list(set(range(N)).difference(set(self._public)))
    self.epsilon = epsilon
    self.delta = delta
    self.noise_multiplier = noise_multiplier
    self.accum_bgts = 0
    self.finished = False
    
    self.__curr_steps = 0

  def precheck(self, dataset_size, batch_size, loc_steps):
    '''Pre-check if the current client could participate in next round'''

    if self.finished: 
      return False

    # Then we need to check if client will exhaust her budget in the following round, i.e., temp_accum_bgts > epsilon.
    tmp_steps = self.__curr_steps + loc_steps
    q = batch_size * 1.0 / dataset_size
    tmp_accum_bgts = 10 * q * math.sqrt(tmp_steps*(-math.log10(self.delta))) / self.noise_multiplier

    # If so, set the status as 'finished' and will not participate the rest training anymore; else, return True
    if self.epsilon - tmp_accum_bgts < 0:
        self.finished = True
        return False
    else:
        self.tmp_accum_bgts = tmp_accum_bgts
        return True
      
  def update(self, loc_steps):
    #print('update: ', clients_id) 
    self.__curr_steps += loc_steps
    self.accum_bgts = self.tmp_accum_bgts
    self.tmp_accum_bgts = 0
    return self.accum_bgts

