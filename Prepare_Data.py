import csv
import numpy as np
import pickle
import Configuration as CF
import Vector_creator as VC
import Feature
import time

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
		if i > 0:
			st = time.time()
			print (row)
			feature = VC.Construct_Vector(row[0])
			feature.extend(Feature.generate(row[0]))
			feature.append(int (row[1]))
			# print (feature)
			writer.writerow(feature)
			print (time.time() - st)
			# break
		i += 1
		# if i%10==0:
		# 	print ("i = ", i)
# pre_data()

def extract_feature(url):
	feature = VC.Construct_Vector(url)
	feature.extend(Feature.generate(url))
	return feature
