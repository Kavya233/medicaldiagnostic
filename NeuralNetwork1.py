#Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

def process(path):
	# importing the dataset
	dataset = pd.read_csv(path)
	X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9]]
	Y = dataset.iloc[:, [10]]

	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

	# Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	# Intinialising the NN
	classifier = Sequential()

	# Adding the input layer and the first Hidden layer 
	classifier.add(Dense(activation="relu", input_dim=10, units=7, kernel_initializer="uniform"))

	# Adding the output layer 
	classifier.add(Dense(activation="sigmoid", input_dim=10, units=3, kernel_initializer="uniform"))

	# Compiling the NN
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	y_binary = to_categorical(y_train)
	
	# Fitting the ANN to the training set
	classifier.fit(X_train, y_binary, batch_size=10, nb_epoch=400)

	# Fitting classifier to the Training set
	# Create your classifier here

	# Predicting the Test set results
	y_pred = classifier.predict_classes(X_test)
	# save the model for later use
	classifier.save('model.h5')
	print("Prediction==",y_pred)
	print(y_test)
	print("X_test==",X_test)
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	ac=accuracy_score(y_test,y_pred)
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR NeuralNetwork IS %f "  % mse)
	print("MAE VALUE FOR NeuralNetwork IS %f "  % mae)
	print("R-SQUARED VALUE FOR NeuralNetwork IS %f "  % r2)
	
	print("RMSE VALUE FOR NeuralNetwork IS %f "  % rms)
	print ("ACCURACY VALUE NeuralNetwork IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('results/NeuralNetworkMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/NeuralNetworkMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('Metrics Value')
	 
	plt.savefig('results/NeuralNetworkMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

