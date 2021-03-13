import os
import struct
import numpy as np
import pickle
import tensorflow.compat.v1 as tf
#import cifar10
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read_mnist(dataset = "training", data_path = "."):

    if dataset is "training":
        fname_img = os.path.join(data_path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(data_path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_path, 't10k-labels-idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    # Reshape and normalize
    print('shape of img:', img.shape)
    img = np.reshape(img, [img.shape[0], img.shape[1] * img.shape[2]])*1.0/255.0

    return img, lbl

def read_cifar10(dataset = "training", data_path = "."):
    if dataset == "training":
        img, lbl = cifar10.load_training_data(data_path)
    
    elif dataset == "testing":
        img, lbl = cifar10.load_test_data(data_path)

    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Reshape and normalize
    print('shape of img:', img.shape)
    img = img / 255.0
    #img = np.reshape(img, [img.shape[0], img.shape[1] * img.shape[2] * img.shape[3]])*1.0/255.0

    return img, lbl

def load_dataset(path, dataset):
    # load the data
    if dataset == 'mnist' or 'fmnist':
        data_path = os.path.join(path, 'dataset', dataset)
        x_train, y_train = read_mnist('training', data_path)
        x_test, y_test = read_mnist('testing', data_path)
       
    elif dataset == 'cifar10':
        data_path = os.path.join(path, 'dataset', dataset)
        x_train, y_train = read_cifar10('training', data_path)
        x_test, y_test = read_cifar10('testing', data_path)
    
    print('shape of data: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # create the validation set
    #x_vali = x_train[TRAIN_SIZE:].astype(float)
    #y_vali = y_train[TRAIN_SIZE:].astype(float)

    # create the train set
    x_train = x_train.astype(float)
    y_train = y_train.astype(float)

    # sort train set (to make federated learning non i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = x_train[indices_train]
    sorted_y_train = y_train[indices_train]

    # create a test set
    x_test = x_test.astype(float)
    y_test = y_test.astype(float)

    return np.array(sorted_x_train), np.array(sorted_y_train), np.array(x_test), np.array(y_test)
