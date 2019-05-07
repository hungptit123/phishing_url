import csv
import numpy as np
import pickle
import Configuration as CF
import Vector_creator as VC
import Feature

def properities():
	reader = csv.reader(open(CF.DIR_LABEL, "r"))
	for row in reader:
		return row

def pre_data():
	# return file csv extract feature from url
	reader = csv.reader(open(CF.DIR_DATA_TRAIN, "r"))
	writer = csv.writer(open(CF.DIR_DATA_TRANFORM, "w"))
	# writer.writerow(properities())
	i = 0
	arr = []
	for row in reader:
		if i > 2404:
			print (row[0])
			feature = VC.Construct_Vector(row[0])
			feature.extend(Feature.generate(row[0]))
			feature.append(int (row[1]))
			# print (feature)
			writer.writerow(feature)
			# break
		i += 1
		# if i%10==0:
		# 	print ("i = ", i)
# pre_data()

def load_data(DIRECTORY):
	reader = csv.reader(open(DIRECTORY, "r"))
	i = 0
	data = []
	for row in reader:
		if i > 0:
			data.append(row)
		i += 1
	data = np.asarray(data)
	print ("shape of data: {}".format(data.shape))
	pickle.dump(data, open(CF.DIR_TRAIN, "wb"))

def get_data_train(DIRECTORY):
	data = pickle.load(open(DIRECTORY, "rb"))
	x_train = data[ : , : 26]
	y_train = data[ : ,26 : ]
	x_train = x_train.astype(np.float32)
	print ("shape of x_train: {}".format(x_train.shape))
	print ("shape of y_train: {}".format(y_train.shape))
	return x_train, y_train




