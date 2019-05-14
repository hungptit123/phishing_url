import tensorflow as tf 
import csv
import numpy as np 

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
	for i in range(39):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		if r < 0:
			r = abs(r)
		dataX[: , i:i+1] = (dataX[: , i:i+1] - avange[i])/r
		# X2[: , i:i+1] = (X2[: , i:i+1] - avange[i])/r
	return dataX

def tensorLogistc():

	dataX, dataY = readData()
	dataX = minize(dataX)
	sample = 2200
	# dataY = dataY.reshape(dataY.shape[0])
	x_train = dataX[:sample, :]
	x_test = dataX[sample:, :]
	y_train = dataY[:sample]
	y_test = dataY[sample:]

	m = dataX.shape[0]
	n = 39
	m_train = x_train.shape[0] #size of train

	# minize(X1,X2,m_train,n)

	# build the model
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	#set up weight and bias
	W = tf.Variable(tf.ones([1,n]))
	b = tf.Variable(tf.ones([1]))

	#sigmoid function
	sigmoid = tf.sigmoid(tf.add(tf.multiply(W,x), b))
	# print (sigmoid)
	#costfuntion
	cost = -1.0/m_train*tf.reduce_sum(tf.add(tf.multiply(tf.log(sigmoid), y), tf.multiply(tf.log(1-sigmoid), (1-y))))

	optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(10001):
			sess.run(optimizer, feed_dict = {x : x_train, y : y_train})
			# if i%1000==0:
			# 	print (sess.run(cost, feed_dict = {x : x_train, y : y_train}))
		re = sess.run(W)
		res = np.matmul(x_test, re.T)
		for i in range(m-m_train):
			if res[i] >= 0.5:
				res[i] = 1
			else :
				res[i] = 0
		print ("accurary: ", 100 - 100.0*np.sum(np.abs(res-y_test))/(m-m_train))
tensorLogistc()