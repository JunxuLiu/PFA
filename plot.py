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

epsilons = {
			'mixgauss1':'MixGauss1 (0.5, 10.0)',
			'mixgauss2':'MixGauss2 (1.0, 10.0)',
			#'mixgauss4':'MixGauss3 (0.5, 5.0)',
			#'mixgauss5':'MixGauss4 (1.0, 5.0)',
			#'gauss1':'Gaussian(3.0)',
			#'pareto1':'Pareto(0.5)',
			#'uniform1':'Uniform(1.0, 10.0)'
			}

colors = ['red','blue','green','orange','darkblue','cyan','magenta','black']
markers = ['s','x','o','.','v','*','D']
fillstyles = ['none','none','none','none','none','none','none']
algs = ['pro1_256', 'wavg', 'fedavg', 'nodp']
num_clients = [10, 20, 30, 40, 50]
#algs = ['pro1_256']
labels = [r'Pfizer', r'WeiAvg', r'Fedavg', r'NP-FedAvg']


def read_data(filename):

	data = []
	with open(filename, 'r') as rfile:
		lines = rfile.readlines()
		for idx, val in enumerate(lines[1::2]):
			values = val.strip().split(",")
			print(values)
			data.append([float(v) for v in values])

	return data

def parser_results(path, epsType, alg, wfilename):

	#print(wfilename)
	wfile = open(wfilename, 'w')

	alg_data = []

	for i in num_clients:
		if alg == 'nodp':
			rfilename = os.path.join(path, 'nodp', '{}.csv'.format(str(i)))
		else:
			rfilename = os.path.join(path, epsType, '{}-{}.csv'.format(str(i), alg))

		#print(rfilename)
		with open(rfilename, 'r') as rfile:
			lines = rfile.readlines()
			values = lines[-1].strip().split(",")
			#print(values)
			alg_data.append([float(v) for v in values])
			
			wfile.write('{}\n'.format(str(i)))
			wfile.write(lines[-1])

	wfile.close()

	return alg_data


def read_results(rfilepath, epsType, res_path):

	data = []
	files = os.listdir(rfilepath)

	# pro, wavg, fedavg
	for alg in algs[:-1]:
		alg_data = []
		wfilename = os.path.join(res_path, '{}.csv'.format(alg))
		#print(wfilename)
		#if os.path.exists(wfilename):
		#	alg_data = read_data(wfilename)
		#else:
		#	alg_data = parser_results(rfilepath, epsType, alg, wfilename)
		alg_data = parser_results(rfilepath, epsType, alg, wfilename)
		data.append(alg_data)

	# nodp
	nodp_data = []
	wfilename = os.path.join(res_path, 'nodp.csv')
	'''
	if os.path.exists(wfilename):
		nodp_data = read_data(wfilename)
	else:
		nodp_data = parser_results(rfilepath, epsType, 'nodp', wfilename)
	'''
	nodp_data = parser_results(rfilepath, epsType, 'nodp', wfilename)
	data.append(nodp_data)

	print(wfilename, len(data), len(data[0]))
	return data

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

	ax.set_title('{}'.format(epsilons), fontsize=14)
	
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
	
	num_subplots = len(epsilons.keys())
	fig, ax = plt.subplots(1, num_subplots, figsize=(4*num_subplots+2, 5), dpi=150)
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
	fig.subplots_adjust(top=0.82, bottom=0.12, left=0.05, wspace = 0.1, hspace = 0)
	fig.savefig(wfile)
	plt.show()

def plotConvergence(wfile, epsilons, num_clients, ax=None, x=None, y=None):
	# y
	lenOfRes = [len(element) for element in y]
	m = min(lenOfRes)

	# x
	if x is None:
		x = range(m)

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(figsize=(4,3), num=1, clear=True)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	for i in range(len(algs)):
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

	fig.suptitle("{}".format(epsilons), fontsize=24)
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

def ConvergenceSingle(wfile_path, epsilons, N=10, *files ):
	res = []
	for i in range(len(files)):
		res = res + [files[i][N//10-1]]
	
	wfile = os.path.join(wfile_path, 'convergence_single_{}'.format(N))
	plotConvergence(wfile, epsilons, num_clients=N, y=res)


def ConvergenceGroup(wfile_path, epsilon, *files):

	wfile = os.path.join(wfile_path, 'convergence_group.png')

	res = []
	for x in num_clients:
		res_tmp = []
		for i in range(len(files)):
			res_tmp = res_tmp + [files[i][x//10-1]]
		res.append(res_tmp)

	plotConvergenceGroup(wfile, epsilon, num_clients, res)


def MethodRes(vers, res_path):

	res = []
	for eid in range(len(epsilons.keys())):

		pros = []
		wavgs = []
		fedavgs = []
		nodps = []

		for i in vers:
			rfilepath = os.path.join(os.getcwd(), 'res_{}'.format(i), dataset, model, isIID)
			#print('******{}********'.format(res_path))
			wfilepath = os.path.join(res_path, list(epsilons.keys())[eid], 'v{}'.format(str(i)))
			#print('******{}********'.format(wfilepath))
			if not os.path.exists(wfilepath):
				os.makedirs(wfilepath)

			if len(algs) == 1:
				pro = read_results(rfilepath, list(epsilons.keys())[eid], wfilepath)[0]
				pros.append(np.mean([values[-10:] for values in pro], axis=1))

			else:
				pro, wavg, fedavg, nodp = read_results(rfilepath, list(epsilons.keys())[eid], wfilepath)
				pros.append(np.mean([values[-10:] for values in pro], axis=1))
				wavgs.append(np.mean([values[-10:] for values in wavg], axis=1))
				fedavgs.append(np.mean([values[-10:] for values in fedavg], axis=1))
				print(len(nodp))
				for values in nodp:
					print('***',values[-10:])

				nodps.append(np.mean([values[-10:] for values in nodp], axis=1))
		
		if len(algs) == 1:
			pro_avg = np.mean(pros, axis=0)
			res.append([pro_avg])

		else:
			pro_avg = np.mean(pros, axis=0)
			wavg_avg = np.mean(wavgs, axis=0)
			fedavg_avg = np.mean(fedavgs, axis=0)
			nodp_avg = np.mean(nodps, axis=0)
			res.append([pro_avg, wavg_avg, fedavg_avg, nodp_avg])
		
	print(res)
	return res

'''
def MethodCompSingle(wfile_path, eid, res):
	x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	xlabel = 'Number of Clients(Local dataset size)'

	wfile = os.path.join(wfile_path, 'acc_methods_eps{}.png'.format(eid))
	plotTestAcc(wfile, epsilon, x=x_axis, xlabel=xlabel, y=res)
'''

def MethodCompGroup(wfile_path, wfile_name, res):

	x_axis = ['10(6000)', '20(3000)', '30(2000)', '40(1500)', '50(1200)']
	xlabel = 'Number of Clients(Local dataset size)'

	wfile = os.path.join(wfile_path, '{}.png'.format(wfile_name))
	plotTestAccGroup(wfile, x=x_axis, xlabel=xlabel, y=res)



def DimensionComp(wfile_path, epsilon, files):
	
	x_axis = ['1', '2', '5', '10', '20', '50']
	xlabel = 'Dimensions'
	res = []
	for x in range(len(x_axis)):
		avg10 = np.mean(files[x][-10:])
		res.append([avg10])

	res = np.array(res).T.tolist()
	wfile = os.path.join(wfile_path, 'acc_proj_dims.png')

	#plotTestAcc(wfile, epsilon, x=x_axis, xlabel=xlabel, y=res)
	plotTestAccGroup(wfile, x=x_axis, xlabel=xlabel, y=res)
	#plotConvergence(wfile, epsilons, num_clients=x_axis[i], ax=ax[i], y=res[i])

def plotDistribution(wfile, ax=None, x=None, xlabel='xlabel', y=None):

	# Set up figure and axis
	subplot = True
	if ax is None:
		fig = plt.figure(figsize=(8,6), num=1, clear=True, dpi=100)
		ax = fig.add_subplot(1, 1, 1)
		subplot = False

	for i in range(len(epsilons.keys())):
		ax.plot(x, y[i], color=colors[i], linewidth=1.5, linestyle="-", marker=markers[i], markersize=3, fillstyle=fillstyles[i], label=list(epsilons.keys())[i])

	ax.tick_params(direction="in")
	ax.legend(loc='lower left')

	ax.set_title('N={}'.format(len(x)), fontsize=14)
	
	ax.axis((None, None, None, None)) # (x_min, x_max, y_min, y_max)
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

def plotDistributionGroup():
	'''
	budgetDic = {
			'mixgauss1':[],
			'mixgauss2':[],
			'mixgauss4':[],
			'mixgauss5':[],
			'gauss1':[],
			'pareto1':[],
			'uniform1':[]
			}
	'''
	budgetDic = {
			10:[],
			20:[],
			30:[],
			40:[],
			50:[]
			}


	with open('eps_distribution', 'r') as rfile:
		lines = rfile.readlines()
		num_lines = len(lines)
		for idx in range(num_lines)[::5]:
			for i in range(5):
				values = lines[idx+i].split()
				dist,N,points = values[0],int(values[1]),[float(e) for e in values[2:]]
				budgetDic[N].append( points )

	print(budgetDic[30])

	wfile_path = os.path.join(os.getcwd(), 'figures')
	wfile = os.path.join(wfile_path, 'eps_distribution.png')	

	fig, ax = plt.subplots(1, 5, figsize=(24,5), dpi=100)

	for i in range(5):
		num_clients = (i+1)*10
		y = []
		for lst in budgetDic[num_clients]:
			y.append(np.sort(lst))

		plotDistribution(wfile, ax=ax[i], x=range(num_clients+1)[1:], xlabel='Number of Clients', y=y)

	fig.suptitle('Privacy Perferences Distributions', fontsize=24)
	fig.text(0.5, 0.02, 'Number of Clients', ha='center', fontsize=20)
	fig.text(0.005, 0.5, 'Privacy Budgets', va='center', rotation='vertical', fontsize=20)

	fig.tight_layout()
	fig.subplots_adjust(top=0.85, bottom=0.12, left=0.03, wspace = 0.1, hspace = 0)
	fig.savefig(wfile)
	plt.show()

if __name__ == "__main__":

	dataset = 'fmnist'
	model = 'lr'
	isIID = r"noniid10"
	vers = 2
	eid = 0
	'''
	wfile_path = os.path.join(os.getcwd(), 'figures', dataset, model, isIID, list(epsilons.keys())[eid], vers)
	if not os.path.exists(wfile_path):
		os.makedirs(wfile_path)

	rfilepath = os.path.join(os.getcwd(), 'res', dataset, model, isIID, list(epsilons.keys())[eid], vers)
	if len(algs) == 1:
		pro = read_results(rfilepath, algs)[0]
		ConvergenceGroup(wfile_path, list(epsilons.values())[eid], pro)

	else:
		pro, wavg, fedavg, nodp = read_results(rfilepath, algs)
		ConvergenceGroup(wfile_path, list(epsilons.values())[eid], pro, wavg, fedavg, nodp)
	
	# Converegnce, single
	#ConvergenceSingle(wfile_path, epsilons, 20, pro, wavg, fedavg, nodp)
	'''

	## Privacy Perferences
	#plotDistributionGroup()
	
	# Test acc, diff. pri. distributions
	fig_path = os.path.join(os.getcwd(), 'figures', dataset, model, isIID)
	res_path = os.path.join(os.getcwd(), 'res', dataset, model, isIID)
	print(res_path)
	if not os.path.exists(fig_path):
		os.makedirs(fig_path)
	
	fig_name = 'acc_methods'
	res = MethodRes(vers=[2], res_path=res_path)
	MethodCompGroup(fig_path, fig_name, res)
	
	'''
	# test acc, dimensions
	wfile_path = os.path.join(os.getcwd(), 'figures', dataset, model, isIID, list(epsilons.keys())[eid], vers)
	if not os.path.exists(wfile_path):
		os.makedirs(wfile_path)

	rfilepath = os.path.join(os.getcwd(), 'res', dataset, model, isIID, list(epsilons.keys())[eid], vers)
	algs = ['proj_dims']
	[proj_dims] = read_results(rfilepath, algs)
	DimensionComp(wfile_path, list(epsilons.values())[eid], proj_dims)
    '''
	#######################################
	
