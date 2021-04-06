import pickle
import numpy as np
import math
import os

def create_clients(num, num_examples, dir):
    '''
    This function creates clients that hold non-iid MNIST data accroding to the experiments in 
    https://research.google.com/pubs/pub44822.html. (it actually just creates indices that point 
    to data. but the way these indices are grouped, they create a non-iid client.)

    we first sort the data by digit label, divide it into 200 shards of size 250, and assign each 
    of 100 clients 2 shards. Thus, as most clients will only have examples of two digits.
    :param num: number of clients
    :param dir: where to store
    :return: _
    '''
    examples_per_client = num_examples//num
    num_classes = 10
    clients = os.path.join(dir, str(num)+'_clients.pkl')
    if os.path.exists(clients):
        print('Client exists at: {}'.format(clients))
        return
    if not os.path.exists(dir):
        os.makedirs(dir)

    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(int(num / 10)):
            temp = np.hstack((temp, k * int(num_examples/10) + np.random.permutation(int(num_examples/10))))
        print('temp.len: ', len(temp))
        buckets = np.hstack((buckets, temp))
        print('buckets.len: ', len(buckets))

    shards = 2 * num # 20
    print('buckets.shape:', buckets.shape, 'shards', shards) # buckets.shape: (10, 5000*(N/10))
    perm = np.random.permutation(shards)

    # z will be of length N*5000 and each element represents a client.
    z = []
    ind_list = np.split(buckets, shards) # 50000/20 = 2500
    print('ind_list.len:', len(ind_list))
    for j in range(0, shards, 2):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 2500*2 datapoints.
        z.append(np.hstack((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        # shuffle the data in each element of z, so that each client doesn't have all digits stuck together.
        perm_2 = np.random.permutation(int(2 * len(buckets) / shards))
        z[-1] = z[-1][perm_2]

    filehandler = open(clients, "wb")
    pickle.dump(z, filehandler)
    filehandler.close()
    print('client created at: {}'.format(clients))

def create_iid_clients(num_clients, num_examples, num_classes, num_examples_per_client, path):
    '''
    This function creates clients that hold iid MNIST data.

    we first sort the data by digit label, divide it into 200 shards of size 250, and assign each 
    of 100 clients 2 shards. Thus, as most clients will only have examples of two digits.
    :param num: number of clients
    :param dir: where to store
    :return: _
    '''
    #assert num_examples % examples_per_client == 0, "Number of examples per client must devide the total number of examples."

    file_path = os.path.join(path, '{}_{}_clients.pkl'.format(num_clients, num_examples_per_client))
    if os.path.exists(os.path.join(file_path)):
        print('Client exists at: {}'.format(file_path))
        client_set = pickle.load(open(file_path, 'rb'))
        return client_set

    if not os.path.exists(path):
        os.makedirs(path)

    # client_set will be of length N*5000 and each element represents a client.
    client_set = []
    rounds = math.ceil(num_clients * num_examples_per_client / num_examples)
    client_per_round = int(num_examples / num_examples_per_client)
    client_count = 0
    for i in range(rounds):
        # shuffle the data
        perm = np.random.permutation(num_examples)
        for j in range(client_per_round):
            if client_count == num_clients:
                break
            client_count += 1
            #each entry of z is associated to 'examples_per_client' examples.
            client_set.append(np.array(perm[j * num_examples_per_client : (j+1) * num_examples_per_client]))

    filehandler = open(file_path, "wb")
    pickle.dump(client_set, filehandler)
    filehandler.close()
    print('client created at: {}'.format(file_path))
    return client_set

def create_noniid_clients(num_clients, num_examples, num_classes, \
                        num_examples_per_client, num_classes_per_client, path):
    '''
    This function creates clients that hold non-iid MNIST data accroding to the experiments in 
    https://research.google.com/pubs/pub44822.html. (it actually just creates indices that point 
    to data. but the way these indices are grouped, they create a non-iid client.)

    we first sort the data by digit label, divide it into 200 shards of size 250, and assign each 
    of 100 clients 2 shards. Thus, as most clients will only have examples of two digits.
    :param num: number of clients
    :param dir: where to store
    :return: _
    '''
    print('Number of classes per client {}'.format(num_classes_per_client))
    classes_per_client = num_classes_per_client
    examples_per_client = num_examples_per_client
    file_path = os.path.join(path, '{}_{}_{}_clients.pkl'.format(num_clients, examples_per_client, classes_per_client))
    if os.path.exists(os.path.join(file_path)):
        print('Client exists at: {}'.format(file_path))
        client_set = pickle.load(open(file_path, 'rb'))
        return client_set

    if not os.path.exists(path):
        os.makedirs(path)

    buckets = [] # 60000 = 10 * 6000
    for k in range(num_classes):        
        temp = np.array(k * int(num_examples / num_classes) + np.random.permutation(int(num_examples / num_classes)))
        print('temp:{}'.format(temp))
        '''
        for j in range(int(num_clients / 10)):
            temp = np.hstack((temp, k * int(num_examples / num_classes) + np.random.permutation(int(num_examples / num_classes))))
        print('temp.len: ', len(temp))
        '''
        buckets = np.hstack((buckets, temp))
        print('buckets.len: ', len(buckets))

    shards = classes_per_client * num_clients # 20
    print('buckets.shape:', buckets.shape, 'shards', shards) # buckets.shape: (10 * 6000)
    perm = np.random.permutation(shards) # 20

    # client_set will be of length num_examples/N and each element represents a client.
    client_set = []
    extra = len(buckets) % shards
    if extra:
        buckets = buckets[:-extra]
    ind_list = np.split(buckets, shards) # 60000/20 = 3000
    print('ind_list.len:', len(ind_list))

    for j in range(0, shards, classes_per_client):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 2500*2 datapoints.
        temp = []
        for k in range(classes_per_client):
            temp = np.hstack((temp, ind_list[int(perm[j+k])]))
        client_set.append(temp)
        # shuffle the data in each element of z, so that each client doesn't have all digits stuck together.
        perm_2 = np.random.permutation(len(temp))
        client_set[-1] = client_set[-1][perm_2]
    
    filehandler = open(file_path, "wb")
    pickle.dump(client_set, filehandler)
    filehandler.close()
    print('client created at: {}'.format(file_path))
    return client_set
    #filehandler = open(dir + '/' + str(num_clients) + '_clients.pkl', "wb")
    #pickle.dump(z, filehandler)
    #filehandler.close()
    #print('client created at: '+ dir + '/' + str(num_clients) + '_clients.pkl')

def check_labels(N, client_set, y_train):
    labels_set = []
    for cid in range(N):
        idx = [int(val) for val in client_set[cid]]
        labels_set.append(set(np.array(y_train)[idx]))

        labels_count = [0]*10
        for label in np.array(y_train)[idx]:
            labels_count[int(label)] += 1
        print('cid: {}, number of labels: {}/10.'.format(cid, len(labels_set[cid])))
        print(labels_count)
