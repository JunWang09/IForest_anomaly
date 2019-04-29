import os
os.chdir('Users/jwang/Desktop/anomalyDetection')
import numpy as np
import _pickle as _pickle
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
import matplotlib.pyplot as pyplot
rn = np.random.RandomState(123)

from feature import load_data, preprocess_data, engineer_features, save_training_data
from model import load_training_data, train_model, save_model, load_model, score
from plot import plot_data, save_plots

import argparse
import logging
logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s"),
					filename = 'main.log',
					filemode = 'a',
					level = logging.INFO,
					datefmt = '%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s"),
								datefmt = '%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger(_name_)

def main(raw_data, version, trainModel=False):
	logging.info('loading data {} ...'.format(raw_data))
	data 				= load_data(raw_data, 1440)

	logging.info('data preprocessing ...')
	data_preprocessed   = preprocess_data(data)

	logging.info('data engineering ...')
	data_feature 		= engineer_features(data_preprocessed)

	logging.info('saving features ...')
	save_training_data(data_feature, 'feature_{}.txt'.format(version), '\t')

	X 					= load_training_data('feature_{}.txt'.format(version))

	if trainModel == True:
		logging.info('training model ...')
		model = train_model(X)
		save_model(model, 'model_iForest')

	logging.info('scoring data ...')
	model 				= load_model('model_iForest')
	abnorm 				= score(model, X, 'abnormal_{}.txt'.format(version))

	logging.info('plot scored data ...')
	plot_data(data, abnorm, 'dim_id_2_tab.txt')
	save_plots('plot_scored_{}.xlsx'.format(version), abnorm, len(data))

if _name_ == '_main_':
	main('errorcode_0315_tab.txt', '20180315', False)

	


















