import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

import os
import math

#style.use('ggplot')

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.unicode_minus'] = False
#doc.generate_pdf(clean_tex=False, compiler='pdfLaTeX')


def read_data(filename):

	data = []
	with open(filename, 'r') as rfile:
		lines = rfile.readlines()
		for idx, val in enumerate(lines[2::2]):
			values = val.strip().split(",")
			data.append([float(v) for v in values])

	return data

def TestAcc(wfile, epsilons, ax=None, x=None, y=None):

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(num=1, clear=True)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	#for i in range(len(x))
	ax.plot(x, res[0], color="lightcoral", linewidth=1.5, linestyle="-", marker='h', label=r"Projection(dim=5)")
	ax.plot(x, res[1], color="burlywood", linewidth=1.5, linestyle="-", marker='p', label=r"Weighted Average")
	ax.plot(x, res[2], color="mediumturquoise", linewidth=1.5, linestyle="-", marker='o', label=r"FedAvg")
	#ax.plot(x, res[3], color="mediumpurple", linewidth=1.5, linestyle="-", marker='s', label=r"No Additive Noise(Baseline)")

	ax.legend(loc='lower left')
	ax.set(	xlabel='Number of Clients (Local dataset size)', \
			ylabel=r'Test Accuracy')
	#title='Num_clients={}, epsilons={}'.format(num_clients, epsilons))
	ax.axis((0, None, 0, None)) # (x_min, x_max, y_min, y_max)
	ax.grid(True)
	'''
	# add annotations on the figure
	ax.annotate("End Point",
                xy=(0.6, 0.5),
                fontsize=12,
                xycoords="data")
	'''
	if not subplot:
		# plot and use space most effectively
		fig.tight_layout()
		fig.savefig(wfile)
		plt.show()

def LowVSHigh(rfile, wfile):

	# x
	x = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	# y
	res = read_data(rfile)
	print(res)

	#plt.figure(figsize=(7,5))
	plt.subplot(211)

	plt.plot(x, res[0], color="lightcoral", linewidth=1.5, linestyle="-", label=r"Projection(dim=5)")
	plt.plot(x, res[1], color="burlywood", linewidth=1.5, linestyle="-", label=r"Weighted Average")
	plt.plot(x, res[2], color="mediumturquoise", linewidth=1.5, linestyle="-", label=r"FedAvg")
	plt.plot(x, res[6], color="mediumpurple", linewidth=1.5, linestyle="-", label=r"No Additive Noise(Baseline)")
	plt.legend(loc="best",fontsize=8)
	#plt.xlabel('Number of Clients (Local dataset size)', fontsize=12)
	plt.ylabel('Test Accuracy', fontsize=12)
	plt.grid(True)
	plt.title('Low-level Privacy Protection')
	
	plt.subplot(212)
	plt.plot(x, res[3], color="lightcoral", linewidth=1.5, linestyle="--", label=r"Projection(dim=5)")
	plt.plot(x, res[4], color="burlywood", linewidth=1.5, linestyle="--", label=r"Weighted Average")
	plt.plot(x, res[5], color="mediumturquoise", linewidth=1.5, linestyle="--", label=r"FedAvg")
	plt.plot(x, res[6], color="mediumpurple", linewidth=1.5, linestyle="-", label=r"No Additive Noise(Baseline)")

	plt.legend(loc="best",fontsize=8)
	plt.xlabel('Number of Clients (Local dataset size)', fontsize=12)
	plt.ylabel('Test Accuracy', fontsize=12)
	plt.title('High-level Privacy Protection')
	plt.grid(True)
	plt.savefig("figures/{}.png".format(wfile))
	plt.show()


def IIDVSNonIID(rfile, wfile):
	# x
	x = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	# y
	res = read_data('IIDVSNonIID')

	plt.plot(x, res[0], color="lightcoral", linewidth=1.5, linestyle="-", label=r"Projection(IID, dim=5)")
	plt.plot(x, res[1], color="burlywood", linewidth=1.5, linestyle="-", label=r"Weighted Average(IID)")

	plt.plot(x, res[2], color="lightcoral", linewidth=1.5, linestyle="--", label=r"Projection(NonIID, dim=5)")
	plt.plot(x, res[3], color="burlywood", linewidth=1.5, linestyle="--", label=r"Weighted Average(NonIID)")

	plt.legend(loc="best", fontsize=8)
	plt.xlabel('Number of Clients (Local dataset size)')
	plt.ylabel('Test Accuracy')
	plt.title('IID VS. NonIID')
	plt.grid(True)
	plt.savefig("figures/{}.png".format(wfile)) # IIDVSNonIID.png
	plt.show()

def DSize1500(rfile, wfile):
	# x
	x = ['10(1500)', '20(1500)', '30(1500)', '40(1500)']
	# y
	res = read_data('DSize1500')
	print(res)

	plt.plot(x, res[0], color="lightcoral", linewidth=1.5, linestyle="-", label=r"Projection(Low, dim=5)")
	plt.plot(x, res[1], color="burlywood", linewidth=1.5, linestyle="-", label=r"Weighted Average(Low)")

	plt.plot(x, res[2], color="lightcoral", linewidth=1.5, linestyle="--", label=r"Projection(High, dim=5)")
	plt.plot(x, res[3], color="burlywood", linewidth=1.5, linestyle="--", label=r"Weighted Average(High)")

	plt.legend(loc="best", fontsize=18)
	plt.xlabel('Number of Clients', fontsize=18)
	plt.ylabel('Test Accuracy', fontsize=18)
	plt.title('Dataset Size = 1500')
	plt.grid(True)
	plt.savefig("DSize1500.png")
	plt.show()


def Convergence(wfile, epsilons, num_clients, ax=None, x=None, y=None):
	# y
	m = min(len(y[0]), len(y[1]), len(y[2]))
	# x
	if x is None:
		x = range(m)

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(num=1, clear=True)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	ax.plot(x, y[0][:m], color="red", linewidth=1.5, linestyle="-", label=r"Pfizer")
	ax.plot(x, y[1][:m], color="blue", linewidth=1.5, linestyle="-", label=r"WeiAvg")
	ax.plot(x, y[2][:m], color="yellow", linewidth=1.5, linestyle="-", label=r"FedAvg")

	ax.legend(loc='lower left')
	ax.set(	xlabel='Communication Rounds (N={})'.format(num_clients), \
			ylabel=r'Test Accuracy')
	#title='Num_clients={}, epsilons={}'.format(num_clients, epsilons))
	ax.axis((0, None, 0, None)) # (x_min, x_max, y_min, y_max)
	ax.grid(True)
	'''
	# add annotations on the figure
	ax.annotate("End Point",
                xy=(0.6, 0.5),
                fontsize=12,
                xycoords="data")
	'''
	if not subplot:
		# plot and use space most effectively
		fig.tight_layout()
		fig.savefig(wfile)
		plt.show()


def ConvergenceGroup(wfile, res_pro, res_wavg, res_fedavg, epsilons, x_axis):

	res = []
	for val in x_axis:
		res.append([res_pro[val//10-1]] + [res_wavg[val//10-1]] + [res_fedavg[val//10-1]])
	'''
	fig = plt.figure(num=1, clear=True)
	ax = fig.add_subplot(1, 1, 1)
	'''
	'''
	fig = plt.figure(num=4, figsize=(10, 10), dpi=100)
	ax = fig.add_subplot(2, 2, 4)  # subgraph
	'''
	fig, ax = plt.subplots(2, 2, figsize=(8,6))
	fig.subplots_adjust(wspace =0, hspace =0.3)
	Convergence(wfile, epsilons, num_clients=x_axis[0], ax=ax[0][0], y=res[0])
	Convergence(wfile, epsilons, num_clients=x_axis[1], ax=ax[0][1], y=res[1])
	Convergence(wfile, epsilons, num_clients=x_axis[2], ax=ax[1][0], y=res[2])
	Convergence(wfile, epsilons, num_clients=x_axis[3], ax=ax[1][1], y=res[3])

	plt.title('Easy as 1,2,3')
	fig.tight_layout()
	fig.savefig(wfile)
	plt.show()


rfilepath = os.path.join(os.getcwd(), 'res', 'epsilons1', 'v1')
wfilepath = os.path.join(os.getcwd(), 'figures', 'epsilons2')


rfile = os.path.join(rfilepath, 'pro1_256')
pro1_256 = read_data(rfile)

rfile = os.path.join(rfilepath, 'wavg')
wavg = read_data(rfile)

rfile = os.path.join(rfilepath, 'fedavg')
fedavg = read_data(rfile)

epsilons = '0.5,10(prob:0.9,0.1)'
'''
############### fig 1 #################
## Converegnce, single
N = 10
res = [pro1_256[N//10]] + [wavg[N//10]] + [fedavg[N//10]]
wfile = os.path.join(wfilepath, 'convergence_single_{}'.format(N))
Convergence(wfile, epsilons, num_clients=N, y=res)
#######################################
'''
'''
############### fig 2 #################
## Convergence, group
x_axis = [20, 30, 40, 50]
wfile = os.path.join(wfilepath, 'convergence_group.png')
ConvergenceGroup(wfile, pro1_256, wavg, fedavg, epsilons, x_axis)
#######################################
'''
#DSize1500Converge
#Dimension

############### fig 3 #################
## test acc, single
x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
res = []
for i in range(len(x_axis)):
	print(pro1_256[i][-10:])
	pro_avg10 = np.mean(pro1_256[i][-10:])
	wavg_avg10 = np.mean(wavg[i][-10:])
	fedavg_avg10 = np.mean(fedavg[i][-10:])
	
	res.append([pro_avg10, wavg_avg10, fedavg_avg10])
print(res)
res = np.array(res).T.tolist()
print(res)
wfile = os.path.join(wfilepath, 'acc_single.png')
TestAcc(wfile, epsilons, x=x_axis, y=res)
#######################################

#IIDVSNonIID()

#DimsConvergence()
