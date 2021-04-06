from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import math
import tensorflow.compat.v1 as tf
import numpy as np

from utils import *
np.random.seed(10)

def Lanczos( mat, m=128 ):

    # reference: https://en.wikipedia.org/wiki/Lanczos_algorithm
    n = mat[0].shape[0]
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0,v0))
    
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    V[0, :] = v0
    
    # step 2.1 - 2.3
    #print(mat)
    w = np.sum([np.dot(col, np.dot(col.T, V[0,:])) for col in mat], 0)
    alfa = np.dot(w, V[0,:])
    w = w - alfa * V[0,:]
    T[0,0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m-1):
        
        beta = np.sqrt( np.dot( w, w ) )
        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors
        # are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(np.conj(V[j,:]), V[i, :])*V[i,:]
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])

        w = np.sum([np.dot(col, np.dot(col.T, V[j,:])) for col in mat], 0)
        alfa = np.dot(w, V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1,j] = beta
        T[j,j-1] = beta
    
    return T, V