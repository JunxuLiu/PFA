import os
import csv
#from utils import set_epsilons


rfpath = 'log_wproj_1200/log_2/mnist/cnn/iid/mixgauss2/'
wfpath = 'res_wproj_1200/log_2/mnist/cnn/iid/mixgauss2/'

#wfilename = 'log_test/log_lr_iid_20_bs128_nm128_10000_100_R8_mediandp_pro5_256_constlr_0121_v6'
#file4 = 'parser_res/log_lr_iid_20_bs128_nm128_10000_100_R8_mediandp_pro5_256_constlr_0121_v6'

settings = [1,2,3,5,10,20,30,50,100]
#settings = [10,50,100]
#settings = [(128,128),(128,16),(64,64),(64,16),(32,32),(32,16)]
#settings = [1,2,5,10]

for i in settings:
    if not os.path.exists(wfpath):
        os.makedirs(wfpath)

    fname = '30_bs4_nm4_10000_100_R8_wpro{}_256_constlr0.01'.format(i)
    rfname = os.path.join(rfpath,fname)

    wfname = os.path.join(wfpath,'30-wpro{}_256-100-bs4-constlr0.01'.format(i))+'.csv'

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
