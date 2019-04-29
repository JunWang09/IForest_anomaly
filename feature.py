import os
import numpy as np
import _pickle as pickle

MIN_PER_DAY = 1440
DATA_WINDOW_SIZE = 60
MIN_ALERT_LEVEL_PRE_PROCESSING = 5
ALERT_WINDOW_SIZE_THRESHOLD = 5
SIGNIFICANCE_LEVEL = 3
MINIMUM_ALERT_LEVEL_FE = 50
MDATA_KEY_LIST = ['t', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']

def check_num_of_elements(d, num_element_req):
	return len(d) == num_element_req

def covert_to_float(x):
	return 0.0 if x == '' else float(x)

def covert_to_int(x):
	return 0 if x == '' else int(x)

def load_data(fileName, MIN_PER_DAY):
	i = 0
	data = []
	with open('data/' + fileName) as f:
		for line in f:
			if i != 0:
				row = {}
				tokens = line.split('\t')
				row['t'] = [covert_to_float(x.strip()) for x in tokens[2].split(',')]
				row['y1'] = [covert_to_float(x.strip()) for x in tokens[3].split(',')]
				row['y2'] = [covert_to_float(x.strip()) for x in tokens[4].split(',')]
				row['y3'] = [covert_to_float(x.strip()) for x in tokens[5].split(',')]
				row['y4'] = [covert_to_float(x.strip()) for x in tokens[6].split(',')]
				row['y5'] = [covert_to_float(x.strip()) for x in tokens[7].split(',')]
				row['y6'] = [covert_to_float(x.strip()) for x in tokens[8].split(',')]
				row['y7'] = [covert_to_float(x.strip()) for x in tokens[9].split(',')]

				for k,v in row.items():
					if len(v) < 10:
						row[k] = [0 for i in range(0, MIN_PER_DAY)]

				row['dim_id'] = int(tokens[0].strip())
				row['dt'] = tokens[10]
				data.append(row)
		i += 1
	return data
	
def all_points_stat(row):
	d = row['y1'] + row['y2'] + row['y3'] + row['y4'] + row['y5'] + row['y6'] + row['y7']
	m = np.mean(d)
	std = np.std(d)
	#mean and std of abnormality interval lengths
	abnormality_lengths = []
	i = 0
	while i < len(d):
		if (d[i] - m)/std >= SIGNIFICANCE_LEVEL:
			length = 1
			i += 1
			while i < len(d):
				if (d[i] - m)/std >= SIGNIFICANCE_LEVEL:
					length += 1
					i += 1
				else:
					i += 1
					break
			abnormality_lengths.append(length)
		else:
			i += 1
	len_mean = np.mean(abnormality_lengths)
	len_std = np.std(abnormality_lengths)

	return m if m>0 else 1.0, std if std>0 else 1.0, len_mean if len_mean>0 else 1.0, len_std if len_std > 0 else 1.0

def nn_points_stat(row):
	index = row['index']
	low_index = max(0, index - 60)
	high_index = max(1439, index + 60)
	nn_data = []
	for i in range(1,8):
		nn_data += row['y' + str(i)][low_index:high_index]

	nn_mean = np.mean(nn_data)
	nn_std = np.std(nn_data)

	historical_high_cnts_nn = 0
	for n in nn_data:
		if (n - nn_mean)/nn_std >= 3:
			historical_high_cnts_nn += 1

	return 1.0 if nn_mean==0.0 else nn_mean,1.0 if nn_std==0.0 else nn_std,float(historical_high_cnts_nn)/float((high_index-low_index)*6)

def preprocess_data(data):
	training_data = []
	cnt = 0
	for i in range(len(data)):
		row = data[i]
		cnt += 1
		mean, std, abn_len_mean, abn_len_std = all_points_stat(row)

		for i in range(MIN_PER_DAY):
			if row['t'][i] == '' or float(row['t'][i] < MIN_ALERT_LEVEL_PRE_PROCESSING):
				continue
			data_point = {}
			data_point['index'] = i
			data_point['dim_id'] = row['dim_id']
			data_point['t'] = row['t']
			data_point['y1'] = row['y1']
			data_point['y2'] = row['y2']
			data_point['y3'] = row['y3']
			data_point['y4'] = row['y4']
			data_point['y5'] = row['y5']
			data_point['y6'] = row['y6']
			data_point['y7'] = row['y7']

			data_point['mean'], data_point['std'], data_point['abn_len_mean'], data_point['abn_len_std'] = mean, std, abn_len_mean, abn_len_std
			data_point['nn_mean'], data_point['nn_std'], data_point['pct_of_historical_high_nn'] = nn_points_stat(data_point)

			training_data.append(data_point)
	
	return training_data

def num_of_std_higher_than_avg(row):
	return float((row['t'][row['index']] - row['mean']))/float(row['std'])

def num_of_std_higher_than_avg_nn(row):
	return float((row['t'][row['index']] - row['nn_mean']))/float(row['nn_std'])

def is_3_std_higher(row):
	return 1 if row['num_of_std_higher_than_avg']>=3 else 0

def is_3_std_higher_nn(row):
	return 1 if row['num_of_std_higher_than_avg_nn'] >=3 else 0

def time_since_last_high(row):
	if row['is_3_std_higher'] != 1:
		return 0
	cnt = 1
	i = row['index'] - 1
	while i >= 0:
		if (row['t'][i] - row['mean'])/row['std'] < 3:
			break
		i -= 1
		cnt += 1

	return cnt

def pct_of_higher_value(row):
	whole_time_data = row['t'][0:row['index'] + 1] + row['y1'] + row['y2'] + row['y3'] + row['y4'] + row['y5'] + row['y6'] + row['y7']
	arr = [x for x in whole_time_data if x>=row['t'][row['index']]]	
	return float(len(arr))/float(len(whole_time_data))

def pct_historical_high(row):
	cnt = 0
	for (k, v) in row.items():
		if k not in MDATA_KEY_LIST or k == 't':
			continue
		if (row[k][row['index']] - row['mean'])/row['std'] >= 3:
			cnt += 1
	return float(cnt)/float(len(row)-1)

def is_historically_high(row):
	return 1 if row['num_of_historical_high'] >= 1 else 0

def num_of_std_higher_than_avg_abn_length(row):
	return (float(row['time_since_last_high']) - float(row['abn_len_mean'])) / float(row['abn_len_std'])

def is_above_minimum_trigger_level(row):
	return 1 if row['t'][row['index']] > MINIMUM_ALERT_LEVEL_FE else 0

def engineer_features(data):
	for row in data:
		row['num_of_std_higher_than_avg']				= num_of_std_higher_than_avg(row)
		row['is_3_std_higher']							= is_3_std_higher(row)
		row['num_of_std_higher_than_avg_nn']			= num_of_std_higher_than_avg_nn(row)
		row['is_3_std_higher_nn']						= is_3_std_higher_nn(row)
		row['time_since_last_high']						= time_since_last_high(row)
		row['pct_of_historical_high']					= pct_historical_high(row)
		row['pct_of_higher_value']						= pct_of_higher_value(row)
		row['num_of_std_higher_than_avg_abn_length'] 	= num_of_std_higher_than_avg_abn_length(row)
		row['error_count']								= row['t'][row['index']]
		row['is_above_minimum_trigger_level']			= is_above_minimum_trigger_level(row)

	return

def save_training_data(data, fileName,delim='\t'):
	fo = open('data/' + fileName, 'w')
	fo.write('dim_id\tcurent_value\ty1\ty2\ty3\ty4\ty5\ty6\ty7\tnum_of_std_higher_than_avg\tis_3_std_higher\tnum_of_std_higher_than_avg_nn\ttime_since_last_high\tpct_of_higher_value\torg_index\tserial_id\n')
	i = 1
	for d in data:
		fo.write(str(d['dim_id']))
		fo.write(delim)
		fo.write(str(d['t'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y1'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y2'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y3'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y4'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y5'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y6'][d['index']]))
		fo.write(delim)
		fo.write(str(d['y7'][d['index']]))
		fo.write(delim)
		fo.write(d['num_of_std_higher_than_avg'])
		fo.write(delim)
		fo.write(d['is_3_std_higher'])
		fo.write(delim)
		fo.write(d['num_of_std_higher_than_avg_nn'])
		fo.write(delim)
		fo.write(d['time_since_last_high'])
		fo.write(delim)
		fo.write(d['pct_of_higher_value'])
		fo.write(delim)
		fo.write(str(d['index']))
		fo.write(delim)
		fo.write(str(i))
		fo.write('\n')
		i += 1
	fo.close()	


















