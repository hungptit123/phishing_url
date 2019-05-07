from sklearn.ensemble import RandomForestClassifier as rfc
import csv
import numpy as np 
import pickle

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
	data_normalize = []
	for i in range(39):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		if r < 0:
			r = abs(r)
		data_normalize.append((avange[i], r))
		dataX[: , i:i+1] = (dataX[: , i:i+1] - avange[i])/r
	return dataX

m = 2700
dataX, dataY = readData()
dataX = minize(dataX)
dataY = dataY.reshape(dataY.shape[0])
x_train = dataX[:m, :]
x_test = dataX[m:, :]
y_train = dataY[:m]
y_test = dataY[m:]
filename = 'finalized_model.sav'
def Random_forest_train():
	# print (dataY.shape)
	model = rfc(n_estimators = 100, min_samples_split = 2)
	model.fit(x_train, y_train)
	# save the model to disk
	pickle.dump(model, open(filename, 'wb'))
# Random_forest_train()

def Accuracy_Randomforest():
	model = pickle.load(open(filename, "rb"))
	score = model.score(x_test, y_test)
	print(score*100)
Accuracy_Randomforest()