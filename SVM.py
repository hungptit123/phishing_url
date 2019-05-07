import tensorflow as tf 
import numpy as np 
from sklearn import svm
import pickle
import cv2
import csv

def readData():
	file = open("Dataset/train.csv","r")
	reader = csv.reader(file)
	data = []
	i = 0
	for row in reader:
		data.append(row)
	data = np.asarray(data)
	# print (data.shape)
	# np.random.shuffle(data)
	dataX = data[:, :39]
	dataX = dataX.astype(np.float32)
	dataY = data[:, 39:]
	dataY = dataY.astype(np.int8)
	return dataX, dataY
# readData()

def minize(dataX):
	x_max = np.max(dataX, axis = 0)
	x_min = np.min(dataX, axis = 0)
	avange = np.sum(dataX, axis = 0)/2401
	for i in range(39):
		# if abs(x_max[i]) < 1:
		# 	x_max[i] = 1
		# dataX = dataX/abs(x_max[i])
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		if r < 0:
			r = abs(r)
		dataX[: , i:i+1] = (dataX[: , i:i+1] - avange[i])/r
	return dataX
def run_svm():
	m = 2000
	dataX, dataY = readData()
	dataX = minize(dataX)
	dataY = dataY.reshape(dataY.shape[0])
	x_train = dataX[:m, :]
	x_test = dataX[m:, :]
	y_train = dataY[:m]
	y_test = dataY[m:]

	c = 100000.0
	model = svm.SVC(C = c, kernel = 'rbf', gamma = 0.1)
	model.fit(x_train, y_train)
	# pickle.dump(model, open("save_model/model_c100", "wb"))

	# print ("da save model")

	# load model
	# model = pickle.load(open("save_model/model_c100", "rb"))
	result = model.score(x_test, y_test)
	print (result)
run_svm()