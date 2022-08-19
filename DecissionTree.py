import os
import h5py as h5
import numpy as np
import pandas as pd
from sklearn import tree,naive_bayes
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#parse each mat structure into a python patient object
class Patient(object):
    PID = ""
    image=""
    label=""
    tumorBorder=""
    tumorMask=""
    
    def __init__(self, PID, image, label,tumorBorder,tumorMask):
        self.PID = PID
        self.image = image
        self.label = label
        self.tumorBorder=tumorBorder
        self.tumorMask=tumorMask


def process(path):
	f=[]
	file_path=path

	for i in range(1000):
	     f.append(h5.File(os.path.join(file_path,str(i+1)+".mat"),'a'))
     
	p=[]
	for i in range(1000):
	    print(i)
	    p.append(Patient('','','','',''))
	    p[i].image=np.array(f[i]['/cjdata/image'])
	    p[i].PID=np.array(f[i]['/cjdata/PID'])
	    p[i].label=f[i]['/cjdata/label'][0][0]
	    p[i].tumorBorder=f[i]['/cjdata/tumorBorder'][0]
	    p[i].tumorMask=list(f[i]['/cjdata/tumorMask'])[0]
	
	columns =['PID', 'image', 'label', 'tumorBorder', 'tumorMask']    

	d={'PID':[], 'image':[], 'label':[], 'tumorBorder':[], 'tumorMask':[]}
	
	for i in range(1000):
	    d['PID'].append(p[i].PID)
	    d['image'].append(p[i].image)
	    d['label'].append(p[i].label)
	    d['tumorBorder'].append(p[i].tumorBorder)
	    d['tumorMask'].append(p[i].tumorMask)
    

	#making dataframe
	Patient_data=pd.DataFrame(list(d.values()),columns)

	Patient_data=Patient_data.transpose()
	Patient_data.to_csv("data.csv",index=False)

	#preprocessing image data for training x_train
	
	#files with corrupted images from (256,256) to (512,512)
	count=0
	for i in range(1000): 
	     if np.array(p[i].image).shape[0]!= 512 or np.array(p[i].image).shape[1]!=512:
	         #print("worst")
	         p[i].image=np.concatenate((np.zeros((256,256)),np.zeros((256,256))),axis=0)
	         p[i].image=np.concatenate((p[i].image,np.zeros((512,256))),axis=1)
	         print("corrupted image",i)
	         count+=1

	""" training images only having (512,512)"""
	image_train=[]
	label_train=[]
	         

	for i in range(1000):
	    image_train.append(p[i].image)
	    label_train.append(p[i].label)
	    
	print(image_train) 
	print(label_train)

	image_train=np.array(image_train)
	label_train=np.array(label_train)

	image_train= image_train.reshape(image_train.shape[0],image_train.shape[1]*image_train.shape[2])

	label_train=label_train.reshape(label_train.shape[0],)

	image_train, label_train = shuffle(image_train, label_train, random_state=42)

	Tumour_classifier = tree.DecisionTreeClassifier()
	#testing the model

	print("dataset split")
	X_train, X_test, y_train, y_test = train_test_split(image_train, label_train, test_size=0.3, random_state=42)
	Tumour_classifier.fit(X_train, y_train)
	y_pred = Tumour_classifier.predict(X_test)


	#accuracy=accuracy_score(y_test,preds)
	#print("Accuracy:", accuracy)

	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR Decission Tree IS %f "  % mse)
	print("MAE VALUE FOR Decission Tree IS %f "  % mae)
	print("R-SQUARED VALUE FOR Decission Tree IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR Decission Tree IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE Decission Tree IS %f" % ac)
	print("---------------------------------------------------------")

	result2=open("results/resultDT.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()

	result2=open('results/DTMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/DTMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('Decission Tree Metrics Value')
	fig.savefig('results/DTMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


