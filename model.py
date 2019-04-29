import os
import numpy as np
import _pickle as pickle
from sklearn.ensemble import IsolationForest
rn = np.random.RandomState(123)

def load_training_data(fileName):
	i = 0
	data = []
	with open('data/' + fileName, 'r') as f:
		for line in f:
			if i != 0:
				tokens = [float(x.strip()) for x in line.split('\t')]
				data.append(tokens)
			i += 1
	return data

def normalize(X):
	mat = np.array(X)

	minimum = [min(mat[:][i]) for i in range(6, len(mat[0]) - 3)]
	maximum = [max(mat[:][i]) for i in range(6, len(mat[0]) - 3)]

	for i in range(6, len(mat[0]) - 3):
		X[:, i] = (X[:,1] - minimum[i-6])/(maximum[i-6] - minimum[i-6])

	return X.tolist()

def trim_down_matrix(m):
	trimmed = []
	for row in m:
		new_row = row[6:-2]
		trimmed.append(new_row)

	return trimmed

def train_model(X):
	train = trim_down_matrix(X)
	model = IsolationForest(n_estimators=500, max_samples=0.6, contamination=0.0004, max_features=0.9, n_jobs=8, random_state=rn)
	model.fit(train)

	return model

def save_model(model, mName):
	with open('cache/' + mName + '.pkl', 'wb') as f:
		pickle.dump(model, f)

def load_model(mName):
	with open('cache/' + mName + '.pkl', 'rb') as f:
		return pickle.load(f)

def remvNeighborDups(a):
	b = []
	for i in range(len(a)):
		if (i==0)|((a[i]!=a[i-1]+1)&(a[i]!=a[i-1]+2)):
			b.append(a[i])
	return b

def score(model, data, scoredName):
	X = trim_down_matrix(data)
	predicted = model.predict(X)

	abnorm = {}
	for i in range(len(data)):
		if predicted[i] != 1:
			dim_id = int(float(data[i][0]))
			if dim_id not in abnorm:
				abnorm[dim_id] = []
			abnorm[dim_id].append(int(float(data[i][-2])))

	for k in abnorm.keys():
		abnorm[k] = remvNeighborDups(abnorm[k])

	fo = open('output/' + scoredName, 'w')
	for k in sorted(abnorm.keys()):
		fo.write(str(k))
		fo.write('\t')
		fo.write(str(abnorm[k]))
		fo.write('\n')

	fo.close()
	return abnorm

	

















