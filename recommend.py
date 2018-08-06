import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

def predictByTopN (n, u_sim, dataMatrix, testMatrix,user_train_nb,itmes_train_nb):
	predictMatrix = np.zeros(testMatrix.shape)
	for uId in range(predictMatrix.shape[0]):
		maxNB = sum([1 for i in u_sim[uId,:] if not np.isnan(i) and i > 0 ]) -1
		indexTopN = np.argsort([-1 if np.isnan(i) else i for i in u_sim[uId,:]  ])[::-1][1:maxNB+1][:n]
		if indexTopN.size == 0:
			continue
		else:
			for tId in range(predictMatrix.shape[1]):
				user = user_train_nb + uId
				item = itmes_train_nb + tId
				denominator = np.sum(u_sim[uId,:][indexTopN])
				numerator = u_sim[uId,:][indexTopN].dot(dataMatrix[:,item][indexTopN])
				predictMatrix[uId, tId] = numerator/denominator
	c_values = testMatrix[testMatrix.nonzero()].flatten()
	p_values = predictMatrix[testMatrix.nonzero()].flatten()
	rmse = np.sqrt(np.average((c_values - p_values) ** 2))
	print("{}\t{:.4f}".format(n,rmse))
	with open("output.log", "a") as output:
		print("{}\t{:.4f}".format(n,rmse),file=output)
	return rmse


def predictByTopNwithSVD (n, u_sim, dataMatrix, testMatrix,user_train_nb,itmes_train_nb):
	from scipy.sparse.linalg import svds
	u, s, vt = svds(dataMatrix, k=20)
	SVD_prediction = np.dot(np.dot(u, np.diag(s)), vt)
	predictMatrix = np.zeros(testMatrix.shape)
	for uId in range(predictMatrix.shape[0]):
		maxNB = sum([1 for i in u_sim[uId,:] if not np.isnan(i) and i > 0 ]) -1
		indexTopN = np.argsort([-1 if np.isnan(i) else i for i in u_sim[uId,:]  ])[::-1][1:maxNB +1][:n]
		if indexTopN.size == 0:
			continue
		else:
			for tId in range(predictMatrix.shape[1]):
				user = user_train_nb + uId
				item = itmes_train_nb + tId
				denominator = np.sum(u_sim[uId,:][indexTopN])
				numerator = u_sim[uId,:][indexTopN].dot(SVD_prediction[:,item][indexTopN])
				predictMatrix[uId, tId] = numerator/denominator
	c_values = testMatrix[testMatrix.nonzero()].flatten()
	p_values = predictMatrix[testMatrix.nonzero()].flatten()
	rmse = np.sqrt(np.average((c_values - p_values) ** 2))
	print("{}\t{:.4f}".format(n,rmse))
	with open("output.log", "a") as output:
		print("{}\t{:.4f}".format(n,rmse),file=output)
	return rmse

def main(inputFile):
	np.seterr(divide='ignore', invalid='ignore')
	Ratings_Names = ['User_ID', 'Item_ID', 'Rating', 'Time_Stamp']
	if inputFile == "u.data":
		df = pd.read_csv(inputFile, skiprows=1, sep='\t', names=Ratings_Names)
	elif inputFile == "ratings.dat":
		df = pd.read_csv(inputFile, skiprows=1, sep='::', names=Ratings_Names, engine='python')

	userTotal = max(df.User_ID)
	itemTotal = max(df.Item_ID)
	Rating_matrix = np.zeros((userTotal, itemTotal))
	for entry in df.itertuples():
		Rating_matrix[entry.User_ID-1, entry.Item_ID-1] = entry.Rating

	print("Get {} users and {} items.".format(userTotal,itemTotal))
	with open("output.log", "a") as output:
		print("Get {} users and {} items.".format(userTotal,itemTotal),file=output)
	user_test_nb = int(userTotal*0.20)
	itmes_test_nb = int(itemTotal*0.20)
	bar_width = 0.8/4
	user_train_nb = userTotal - user_test_nb
	itmes_train_nb = itemTotal - itmes_test_nb
	n_list = [5, 10, 20, 40, 100, user_train_nb]
	x = np.arange(len(n_list))

	trainMatrix = np.array([ list(line[:itmes_train_nb]) for line in Rating_matrix ])
	dataMatrix =  Rating_matrix[:user_train_nb]
	testMatrix =  np.array([ list(line[itmes_train_nb:]) for line in Rating_matrix[user_train_nb:] ])

	print("Top N\tRmase")
	with open("output.log", "a") as output:
		print("Top N\tRmase",file=output)
	# Rating Values
	sim = np.dot(trainMatrix, trainMatrix.T)
	norms = np.array([np.sqrt(np.diagonal(sim))])
	u_sim = sim / (norms * norms.T)
	u_sim = np.array([ list(line[:user_train_nb]) for line in u_sim[user_train_nb:] ])
	print("*"*20+"Rating values"+"*"*20)
	with open("output.log", "a") as output:
		print("*"*20+"Rating values"+"*"*20,file=output)
	perf_rmse = list()
	for n in n_list:
		rmse = predictByTopN(n, u_sim, dataMatrix, testMatrix, user_train_nb,itmes_train_nb)
		perf_rmse.append(rmse)
	r_plt=plt.bar(x, perf_rmse, width=bar_width, label='Rating') #,fc = 'blue'

	print("*"*20+"Rating values with SVD"+"*"*20)
	with open("output.log", "a") as output:
		print("*"*20+"Rating values with SVD"+"*"*20,file=output)
	perf_rmse = list()
	for n in n_list:
		rmse = predictByTopNwithSVD(n, u_sim, dataMatrix, testMatrix, user_train_nb,itmes_train_nb)
		perf_rmse.append(rmse)
	x = x+bar_width
	p_plt=plt.bar(x, perf_rmse, width=bar_width, label='SVD') #,fc = 'yellow'

	# Normalised Coorelation
	Normalised_matrix = np.zeros(trainMatrix.shape)
	for i in range(len(trainMatrix)):
		line = trainMatrix[i]
		avg = np.sum(line)/sum([0 if n==0 else 1 for n in line])
		newLine = [ n-avg if n > 0 else 0 for n in line]
		Normalised_matrix[i] = newLine

	print("*"*20+"Normalised values"+"*"*20)
	with open("output.log", "a") as output:
		print("*"*20+"Normalised values"+"*"*20,file=output)
	sim = np.dot(Normalised_matrix, Normalised_matrix.T)
	norms = np.array([np.sqrt(np.diagonal(sim))])
	u_sim = sim / (norms * norms.T)
	u_sim = np.array([ list(line[:user_train_nb]) for line in u_sim[user_train_nb:] ])
	perf_rmse = list()
	for n in n_list:
		rmse = predictByTopN(n, u_sim, dataMatrix, testMatrix, user_train_nb,itmes_train_nb)
		perf_rmse.append(rmse)
	x = x+bar_width
	p_plt=plt.bar(x, perf_rmse, width=bar_width, label='Rating_Normalised') # ,fc = 'red'

	print("*"*20+"Normalised values with SVD"+"*"*20)
	with open("output.log", "a") as output:
		print("*"*20+"Normalised values with SVD"+"*"*20,file=output)
	perf_rmse = list()
	for n in n_list:
		rmse = predictByTopNwithSVD(n, u_sim, dataMatrix, testMatrix, user_train_nb,itmes_train_nb)
		perf_rmse.append(rmse)
	x = x+bar_width
	p_plt=plt.bar(x, perf_rmse, width=bar_width, label='SVD_Normalised') #,fc = 'yellow'
	delta = max(perf_rmse) - min(perf_rmse)

	plt.legend(loc=1, borderaxespad=0.)
	plt.xticks(x , n_list)
	plt.ylim(min(perf_rmse) - 10*delta)
	plt.ylabel('RMSE')
	plt.title('RMSEs with varied Top N values by Testing dataSet {}'.format(inputFile))
	plt.rcParams["figure.figsize"] = [2090,1248]
	plt.savefig('{}.png'.format(inputFile), dpi=300)
	plt.clf()

if __name__ == '__main__':
	if "u.data" in os.listdir() and "ratings.dat" in os.listdir():
		with open("output.log", "w") as output:
			output.write("")
		inputFile = "u.data"
		print("Start run program with 100k dataSet. It will take about 1 min.")
		with open("output.log", "a") as output:
			print("Start run program with 100k dataSet.",file=output)
		main(inputFile)
		print("Dataset 100k has been tested. please check the output.log, u.data.png and ratings.dat.png ")
		# 1:35 - ?
		print("It is going to run program with 1M dataSet. It will take about 20 mins. Do you wanna continue?")
		c = input("Press y or Y to continue.")
		if c.lower() == "y":
			inputFile = "ratings.dat"
			with open("output.log", "a") as output:
				print("Start run program with 1M dataSet.",file=output)
			main(inputFile)
			print("Dataset 1M has been tested. please check the output.log, u.data.png and ratings.dat.png ")
		print("Program run completely without errors.")
	else:
		print("Error: Unknown file or path, please move \"u.data\" \"ratings.dat\" to the same folder and run the program.")
		exit(-1)

