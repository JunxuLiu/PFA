
"""
Non projection component.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class HParams(object):

    def __init__(self, loc_batch_size, 
                loc_num_microbatches, 
                loc_lr,
                glob_steps,
                loc_steps,
                loc_l2_norm):

        self.bs = loc_batch_size
        self.num_mbs = loc_num_microbatches
        self.lr = loc_lr
        self.glob_steps = glob_steps
        self.loc_steps = loc_steps
        self.l2_norm_clip = loc_l2_norm
    