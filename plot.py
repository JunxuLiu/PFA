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

epsilons = {'epsilons1':'MixGauss\n0.5(prob=0.9), 10(prob=0.1)',
			'epsilons2':'MixGauss\n1.0(prob=0.9), 10(prob=0.1)',
			'epsilons4':'MixGauss\n1.0(prob=0.9), 10(prob=0.1)',
			'epsilons5':'MixGauss\n1.0(prob=0.9), 10(prob=0.1)',
			'epsilonsu':'Uniform\n[1.0, 10.0]'}

colors = ['red','blue','green','orange']
markers = ['s','x','o','.']
fillstyles = ['none','none','none','none']
filelist = ['pro1_256', 'wavg', 'fedavg', 'nodp']
labels = [r'Pfizer', r'WeiAvg', r'Fedavg', r'NP-FedAvg']

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
	ax.legend(loc='lower left')

	ax.set_title('epsilons: {}'.format(epsilons), fontsize=14)
	
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
		ax.set_xlabel(xlabel, fontsize=14)
		ax.set_ylabel('Test Accuracy', fontsize=14)
		# plot and use space most effectively
		fig.tight_layout()
		fig.savefig(wfile)
		plt.show()


def plotTestAccGroup(wfile, x, xlabel, y):
	
	fig, ax = plt.subplots(1, 5, figsize=(24,5), dpi=100)
	for i in range(len(epsilons.keys())):
		plotTestAcc(wfile, list(epsilons.values())[i], ax=ax[i], x=x, xlabel=xlabel, y=y[i])

	fig.suptitle("Method Comparison", fontsize=20)
	fig.text(0.5, 0.02, 'Communication Rounds (total number of SGD iterations is 10000)', ha='center', fontsize=20)
	fig.text(0.005, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=20)

	# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
	#plt.setp([a.get_xticklabels() for a in ax[:]], visible=False)
	#plt.setp([a.get_yticklabels() for a in ax[1:]], visible=False)
	#plt.setp(labels, rotation=45)

	fig.tight_layout()
	fig.subplots_adjust(top=0.82, bottom=0.12, left=0.03, wspace = 0.1, hspace = 0)
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

	for i in range(len(labels)):
		ax.plot(x, y[i][:m], color=colors[i], linewidth=1.5, linestyle="-", label=labels[i])

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
		ax.set(	xlabel=r'Communication Rounds', \
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


def MethodRes(version):

	res = []
	for eid in range(len(epsilons.keys())):

		pros = []
		wavgs = []
		fedavgs = []
		nodps = []

		for i in range(version):
			rfilepath = os.path.join(os.getcwd(), 'res', dataset, model, isIID, list(epsilons.keys())[eid], 'v{}'.format(i+1))
			pro, wavg, fedavg, nodp = read_results(rfilepath, filelist)
			pros.append(np.mean([values[-10:] for values in pro], axis=1))

			wavgs.append(np.mean([values[-10:] for values in wavg], axis=1))
			fedavgs.append(np.mean([values[-10:] for values in fedavg], axis=1))
			nodps.append(np.mean([values[-10:] for values in nodp], axis=1))

		pro_avg = np.mean(pros, axis=0)
		wavg_avg = np.mean(wavgs, axis=0)
		fedavg_avg = np.mean(fedavgs, axis=0)
		nodp_avg = np.mean(nodps, axis=0)

		res.append([pro_avg, wavg_avg, fedavg_avg, nodp_avg])

	print(res)
	return res

'''
def MethodCompSingle(wfilepath, eid, res):
	x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	xlabel = 'Number of Clients(Local dataset size)'

	wfile = os.path.join(wfilepath, 'acc_methods_eps{}.png'.format(eid))
	plotTestAcc(wfile, epsilon, x=x_axis, xlabel=xlabel, y=res)
'''

def MethodCompGroup(wfilepath, res):
	x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	xlabel = 'Number of Clients(Local dataset size)'

	wfile = os.path.join(wfilepath, 'acc_methods.png')
	plotTestAccGroup(wfile, x=x_axis, xlabel=xlabel, y=res)



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
	model = 'lr'
	isIID = 'iid'
	eid = 0
	#wfilepath = os.path.join(os.getcwd(), 'figures', dataset, model, isIID, list(epsilons.keys())[eid])
	#rfilepath = os.path.join(os.getcwd(), 'res', dataset, model, isIID, list(epsilons.keys())[eid], 'v1')

	#pro1_256, wavg, fedavg, nodp = read_results(rfilepath, filelist)
	## fig 1
	# Converegnce, single
	#ConvergenceSingle(wfilepath, epsilons, 20, pro1_256, wavg, fedavg, nodp)

	## fig 2
	# Convergence, group
	#ConvergenceGroup(wfilepath, list(epsilons.values())[eid], pro1_256, wavg, fedavg, nodp)

	## fig 3
	# test acc, methods
	wfilepath = os.path.join(os.getcwd(), 'figures', dataset, model, isIID)
	res = MethodRes(version=1)
	MethodCompGroup(wfilepath, res)
	
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
	