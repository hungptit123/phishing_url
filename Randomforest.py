from sklearn.ensemble import RandomForestClassifier as rfc
import csv
import numpy as np 
import pickle
import Prepare_Data as PD
import time

label = ["SAFE", "PHISHING"]

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
	avange = np.sum(dataX, axis = 0)/2759
	coefficient = []
	for i in range(39):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		if r < 0:
			r = abs(r)
		coefficient.append((avange[i], r))
		# pickle.dump(coefficient, open("Dataset/coefficient", "wb"))
		dataX[: , i:i+1] = (dataX[: , i:i+1] - avange[i])/r
	return dataX

def normalize(x_test):
	coefficient = pickle.load(open("Dataset/coefficient", "rb"))
	for i in range(39):
		r = coefficient[i][1]
		avange = coefficient[i][0]
		x_test[:, i:i+1] = (x_test[:, i:i+1] - avange)/r
	return x_test

filename = 'finalized_model.sav'
def Random_forest_train():
	# print (dataY.shape)
	m = 2200
	dataX, dataY = readData()
	dataX = minize(dataX)
	dataY = dataY.reshape(dataY.shape[0])
	x_train = dataX[:m, :]
	x_test = dataX[m:, :]
	y_train = dataY[:m]
	y_test = dataY[m:]
	model = rfc(n_estimators = 100)
	model.fit(x_train, y_train)
	# save the model to disk
	pickle.dump(model, open(filename, 'wb'))
	score = model.score(x_test, y_test)
	print(score*100)
# Random_forest_train()

def Accuracy_Randomforest(url):
	print ("load model: ......")
	model = pickle.load(open(filename, "rb"))
	print ("model load finsh.")
	st = time.time()
	feature = PD.extract_feature(url)
	print ("Feature: ")
	print (feature)
	x_test = []
	x_test.append(feature)
	x_test = np.asarray(x_test)
	x_test = normalize(x_test) 
	predict = model.predict(x_test)
	print ("Label: ", label[predict[0]])
	print ("time: ", time.time()-st)
url = "https://www.skype.com/en/"
Accuracy_Randomforest(url)