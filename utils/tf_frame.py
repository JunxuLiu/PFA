from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import pickle
import math
import heapq
import csv
import re
#import copy
import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(10)

def global_step_creator():
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='global_step_placeholder')
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return increase_global_step, set_global_step


def Assignements(dic):
    return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]


def Vname_to_Pname(var):
    return var.name[:var.name.find(':')] + '_placeholder'


def Vname_to_FeedPname(var):
    return var.name[:var.name.find(':')] + '_placeholder:0'


def Vname_to_Vname(var):
    return var.name[:var.name.find(':')]
