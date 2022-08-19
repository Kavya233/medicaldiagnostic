import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.svm import SVC

def process(path):
	df = pd.read_csv(path)
	
	X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9]]
	Y = df.iloc[:, [10]]
	print(X)
	print(Y)

	#spliting the dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25, random_state =0 )

	dt = SVC()
	dt.fit(X_train, y_train); # Build a forest of trees from training set
	y_pred = dt.predict(X_test)
	print(len(y_test))
	print(len(y_pred))

	result2=open("results/resultSVM.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred.round())
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	

	print("---------------------------------------------------------")
	print("MSE VALUE FOR SVM IS %f "  % mse)
	print("MAE VALUE FOR SVM IS %f "  % mae)
	print("R-SQUARED VALUE FOR SVM IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR SVM IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE SVM IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('results/SVMMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/SVMMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('SVM Metrics Value')
	plt.savefig('results/SVMMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
