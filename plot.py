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

epsilons = {'epsilons1':'0.5(prob=0.9), 10(prob=0.1)',
			'epsilons2':'1.0(prob=0.9), 10(prob=0.1)'}
colors = ['red','blue','green','orange']
markers = ['s','x','o','s']
fillstyles = ['none','none','none','none']
labels = ['Pfizer', 'WeiAvg', 'Fedavg', 'NP-FedAvg']

def read_data(filename):

	data = []
	with open(filename, 'r') as rfile:
		lines = rfile.readlines()
		for idx, val in enumerate(lines[2::2]):
			values = val.strip().split(",")
			data.append([float(v) for v in values])
	return data

def read_results(rfilepath, filenames):

	files = []
	for fname in filenames:
		rfile = os.path.join(rfilepath, fname)
		data = read_data(rfile)
		files.append(data)

	return files

def plotTestAcc(wfile, epsilons, ax=None, x=None, xlabel='xlabel', y=None):

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(figsize=(8,6), num=1, clear=True, dpi=100)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	for i in range(len(y)):
		ax.plot(x, y[i], color=colors[i], linewidth=1.5, linestyle="-", marker=markers[i], fillstyle=fillstyles[i], label=labels[i])

	ax.tick_params(direction="in")
	ax.legend(loc='best')

	ax.set_title('epsilons: {}'.format(epsilons), fontsize=14)
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel('Test Accuracy', fontsize=14)
	ax.axis((None, None, 0, 1.0)) # (x_min, x_max, y_min, y_max)
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

def plotConvergence(wfile, epsilons, num_clients, ax=None, x=None, y=None):
	# y
	m = min(len(y[0]), len(y[1]), len(y[2]))
	# x
	if x is None:
		x = range(m)

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(figsize=(4,3), num=1, clear=True)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	ax.plot(x, y[0][:m], color="red", linewidth=1.5, linestyle="-", label=r"Pfizer")
	ax.plot(x, y[1][:m], color="blue", linewidth=1.5, linestyle="-", label=r"WeiAvg")
	ax.plot(x, y[2][:m], color="yellow", linewidth=1.5, linestyle="-", label=r"FedAvg")
	ax.tick_params(direction="in")
	ax.legend(loc='upper left')
	
	#title='Num_clients={}, epsilons={}'.format(num_clients, epsilons))
	ax.axis((0, None, 0, 1.0)) # (x_min, x_max, y_min, y_max)
	ax.grid(True)
	'''
	# add annotations on the figure
	ax.annotate("End Point",
                xy=(0.6, 0.5),
                fontsize=12,
                xycoords="data")
	'''
	# plot and use space most effectively
	ax.set_title('N={}'.format(num_clients), fontsize=20)
	if not subplot:
		# plot and use space most effectively
		ax.set(	xlabel='Communication Rounds', \
				ylabel=r'Test Accuracy')

		# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
		plt.setp([ax.get_xticklabels()], visible=False)
		plt.setp([ax.get_yticklabels()], visible=False)

		# Tight layout often produces nice results
		# but requires the title to be spaced accordingly
		fig.tight_layout()
		fig.savefig(wfile)
		plt.show()


def plotConvergenceGroup(wfile, epsilons, x_axis, res):
	
	fig, ax = plt.subplots(1, 5, figsize=(24,5), dpi=100)

	for i in range(len(x_axis)):
		print(res[i])
		plotConvergence(wfile, epsilons, num_clients=x_axis[i], ax=ax[i], y=res[i])

	fig.suptitle("epsilons: {}".format(epsilons), fontsize=24)
	fig.text(0.5, 0.02, 'Communication Rounds (total number of SGD iterations is 10000)', ha='center', fontsize=20)
	fig.text(0.005, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=20)

	# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
	#plt.setp([a.get_xticklabels() for a in ax[:]], visible=False)
	#plt.setp([a.get_yticklabels() for a in ax[1:]], visible=False)
	#plt.setp(labels, rotation=45)

	fig.tight_layout()
	fig.subplots_adjust(top=0.85, bottom=0.12, left=0.03, wspace = 0.1, hspace = 0)
	fig.savefig(wfile)
	plt.show()

def ConvergenceSingle(wfilepath, epsilons, N=10, *files ):
	res = []
	for i in range(len(files)):
		res = res + [files[i][N//10-1]]
	
	wfile = os.path.join(wfilepath, 'convergence_single_{}'.format(N))
	plotConvergence(wfile, epsilons, num_clients=N, y=res)


def ConvergenceGroup(wfilepath, epsilon, *files):
	x_axis = [10, 20, 30, 40, 50]
	wfile = os.path.join(wfilepath, 'convergence_group.png')

	res = []
	for x in x_axis:
		res_tmp = []
		for i in range(len(files)):
			res_tmp = res_tmp + [files[i][x//10-1]]

		res.append(res_tmp)

	plotConvergenceGroup(wfile, epsilon, x_axis, res)


def MethodComp(wfilepath, epsilon, *files):
	x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	xlabel = 'Number of Clients(Local dataset size)'

	res = []
	for x in range(len(x_axis)):
		res_tmp = []
		for i in range(len(files)):
			res_comb = []
			for j in range(len(files[i])):
				res_comb += files[i][j][x][-10:]
			avg10 = np.mean(res_comb)
			res_tmp.append(avg10)

		#pro_avg10 = np.mean(pro1_256[x][-10:] + pro1_256_v2[x][-10:])
		#wavg_avg10 = np.mean(wavg[x][-10:] + wavg_v2[x][-10:])
		#fedavg_avg10 = np.mean(fedavg[x][-10:] + fedavg_v2[x][-10:])
		res.append(res_tmp)

	res = np.array(res).T.tolist()
	wfile = os.path.join(wfilepath, 'acc_methods.png')
	plotTestAcc(wfile, epsilon, x=x_axis, xlabel=xlabel, y=res)


def DimensionComp(wfilepath, epsilon, files):
	
	x_axis = ['1', '2', '5', '10', '20', '50']
	xlabel = 'Dimension'
	res = []
	for x in range(len(x_axis)):
		avg10 = np.mean(files[x][-10:])
		res.append([avg10])

	res = np.array(res).T.tolist()
	wfile = os.path.join(wfilepath, 'acc_proj_dims.png')
	plotTestAcc(wfile, epsilon, x=x_axis, xlabel=xlabel, y=res)


if __name__ == "__main__":

	dataset = 'mnist'
	eid = 0
	wfilepath = os.path.join(os.getcwd(), 'figures', dataset, list(epsilons.keys())[eid])

	
	rfilepath = os.path.join(os.getcwd(), 'res', dataset, list(epsilons.keys())[eid], 'v1')
	filelist = ['pro1_256', 'wavg', 'fedavg']
	pro1_256, wavg, fedavg = read_results(rfilepath, filelist)
	'''
	## fig 1
	# Converegnce, single
	# ConvergenceSingle(wfilepath, epsilons, 20, pro1_256, wavg, fedavg)

	## fig 2
	# Convergence, group
	# ConvergenceGroup(wfilepath, list(epsilons.values())[eid], pro1_256, wavg, fedavg)
	'''
	
	#DSize1500Converge
	'''
	## fig 3
	# test acc, methods
	rfilepath = os.path.join(os.getcwd(), 'res', dataset, list(epsilons.keys())[eid], 'v2')
	filelist = ['pro1_256', 'wavg', 'fedavg']
	pro1_256_v2, wavg_v2, fedavg_v2 = read_results(rfilepath, filelist)
	
	MethodComp(wfilepath, list(epsilons.values())[eid], (pro1_256, pro1_256_v2), (wavg, wavg_v2), (fedavg, fedavg_v2))
	'''
	'''
	## fig 4
	# test acc, dimensions
	rfilepath = os.path.join(os.getcwd(), 'res', dataset, list(epsilons.keys())[eid], 'v1')
	filelist = ['proj_dims']
	[proj_dims] = read_results(rfilepath, filelist)
	DimensionComp(wfilepath, list(epsilons.values())[eid], proj_dims)
	#######################################
	'''
	#IIDVSNonIID()
	