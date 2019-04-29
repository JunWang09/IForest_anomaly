import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #display chinese characters
plt.rcParams['axes.unicode_minus'] = False #display minus sign
import os
from joblib import Parallel, delayed

def load_dim_id(fileName):
	i = 0
	data = {}
	with open(fileName, 'r') as f:
		for line in f:
			if i != 0:
				tokens = line.split('\t')
				data[int(tokens[0])] = tokens[1]
			i += 1
	return data

def plot_data_single_row(d, dim_id):
	fig, ax = plt.subplots(figsize=(20,8))
	ax.plot(d['y7'], 'k', label='y7')
	ax.plot(d['y3'], 'c', label='y3')
	ax.plot(d['y1'], 'c', label='y1')
	ax.plot(d['t'], 'c', label='today')
	if len(d['predicted']) > 0:
		ax.plot(d['predicted'], [d['t'][i] for i in d['predicted']], 'bo', label='predicted')
	plt.legend(loc='upper left', prop=matplotlib.font_manager.FontProperties(size=12))
	plt.title('dim_instance:' + dim_id[d['dim_id']], fontsize=16)
	plt.savefig('plot/' + str(d['dim_id']) + '.png')

def plot_data(data, abnorm, dim_id_file):
	dim_id = load_dim_id('data/' + dim_id_file)
	for i in range(len(data)):
		data[i]['predicted'] = []
		for k in abnorm.keys():
			if float(data[i]['dim_id']) == float(k):
				data[i]['predicted'] += abnorm[k]
		plot_data_single_row(data[i], dim_id)

	#Parallel(n_jobs=8, verbose=5)(delayed (plot_data_single_row)(data[i], dim_id) for i in list(range(len(data))))

def save_plots(fileName, abnorm, length):
	abnorm_id = []
	for k in sorted(abnorm.keys()):
		abnorm_id.append(int(k))
	nonAbnorm_id = list(set(range(1, length+1)) -set(abnorm_id))

	writer = pd.ExcelWriter('plot/' + fileName)
	workbook = writer.book
	worksheet = workbook.add_worksheet('plot')
	k = 1
	for id_num in abnorm_id + nonAbnorm_id:
		worksheet.write('A'+str(k), str(id_num))
		worksheet.insert_image('A'+str(k+2), 'plot/'+str(id_num)+'.png', {'x_scale':0.8, 'y_scale':0.6})
		k += 30
	writer.save()

	for id_num in range(1, length+1):
		os.remove('plot/' + str(id_num) + '.png')



		










