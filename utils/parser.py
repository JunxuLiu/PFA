import os
import csv
#from utils import set_epsilons


rfpath = 'log/log_2/mnist/cnn/iid/mixgauss1/'
wfpath = 'res/res_final/mnist/cnn/iid/mixgauss1/'

#wfilename = 'log_test/log_lr_iid_20_bs128_nm128_10000_100_R8_mediandp_pro5_256_constlr_0121_v6'
#file4 = 'parser_res/log_lr_iid_20_bs128_nm128_10000_100_R8_mediandp_pro5_256_constlr_0121_v6'

settings = [10,20,30,40,50]
#settings = [10,50,100]
#settings = [(128,128),(128,16),(64,64),(64,16),(32,32),(32,16)]
#settings = [1,2,5,10]

for i in settings:
    if not os.path.exists(wfpath):
        os.makedirs(wfpath)

    fname = '10-pro{}_256'.format(i)
    #fname = 'log_lr_noniid_{}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0125_v6'.format(i)
    #fname = '10_bs{}_nm{}_5000_100_R8_pro1_128_constlr'.format(i[0],i[1])
    rfname = os.path.join(rfpath,fname)
    #wfname = os.path.join(wfpath,fname)    
    wfname = os.path.join(wfpath,'30_bs128_nm128_10000_100_R8_pro{}_256_constlr'.format(i))+'.csv'

    if os.path.isfile(rfname):
        print(rfname)

        Accuracy_accountant = []
        with open(rfname, 'r') as rfile:
            line = rfile.readline()
            while(line):
                if(line[:6]==' - The'):
                    #print(line)
                    acc = line.split(':')[1]
                    acc = float(acc)
                    Accuracy_accountant.append(acc)
                line = rfile.readline()

        print(wfname)
        with open(wfname, "w") as csvfile:
            wfile = csv.writer(csvfile, delimiter=',')
            wfile.writerow(Accuracy_accountant)

    else:
        print('file {} not exists.'.format(rfname))

        
'''
if not os.path.exists(wfpath):
    os.makedirs(wfpath)

fname = '10_bs16_nm16_5000_100_R8_pro1_128_constlr'
rfname = os.path.join(rfpath,fname)
wfname = os.path.join(wfpath,fname)+'.csv'

if os.path.isfile(rfname):
    print(rfname)

    Accuracy_accountant = []
    with open(rfname, 'r') as rfile:
        line = rfile.readline()
        while(line):
            if(line[:6]==' - The'):
                #print(line)
                acc = line.split(':')[1]
                acc = float(acc)
                Accuracy_accountant.append(acc)
            line = rfile.readline()

    print(wfname)
    with open(wfname, "w") as csvfile:
        wfile = csv.writer(csvfile, delimiter=',')
        wfile.writerow(Accuracy_accountant)

else:
    print('file {} not exists.'.format(rfname))
'''
'''
filename = 'mixgauss1'
epsilons, threshold = set_epsilons(filename, 10, is_distributions = True)
epsilons, threshold = set_epsilons(filename, 20, is_distributions = True)
epsilons, threshold = set_epsilons(filename, 30, is_distributions = True)
epsilons, threshold = set_epsilons(filename, 40, is_distributions = True)
epsilons, threshold = set_epsilons(filename, 50, is_distributions = True)
print(epsilons)
'''
