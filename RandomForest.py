import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def process(path):
        df=pd.read_csv(path)
        columns_to_retain = ["sg", "al", "sc", "hemo",
                                 "pcv", "wbcc", "rbcc", "htn", "classification"]

        #columns_to_retain = df.columns, Drop the columns that are not in columns_to_retain
        df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
            
        # Drop the rows with na or missing values
        df = df.dropna(axis=0)
        for column in df.columns:
                if df[column].dtype == np.number:
                    continue
                df[column] = LabelEncoder().fit_transform(df[column])
        df.head()
        print(df)
        #Split the data
        X = df.drop(["classification"], axis=1)
        y = df["classification"]
        #Feature Scaling
        x_scaler = MinMaxScaler()
        x_scaler.fit(X)
        column_names = X.columns
        X[column_names] = x_scaler.transform(X)
        #Split the data into 80% training and 20% testing 
        X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)
        model2=RandomForestClassifier()
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        print("predicted")
        print(y_pred)
        print(y_test)
        result2=open("results/resultRF.csv","w")
        result2.write("ID,Predicted Value" + "\n")
        for j in range(len(y_pred)):
            result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
        result2.close()
        
        mse=mean_squared_error(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        r2=r2_score(y_test, y_pred)
        
        
        print("---------------------------------------------------------")
        print("MSE VALUE FOR RandomForest IS %f "  % mse)
        print("MAE VALUE FOR RandomForest IS %f "  % mae)
        print("R-SQUARED VALUE FOR RandomForest IS %f "  % r2)
        rms = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE VALUE FOR RandomForest IS %f "  % rms)
        ac=accuracy_score(y_test,y_pred)
        print ("ACCURACY VALUE RandomForest IS %f" % (ac))
        print("---------------------------------------------------------")
        

        result2=open('results/RFMetrics.csv', 'w')
        result2.write("Parameter,Value" + "\n")
        result2.write("MSE" + "," +str(mse) + "\n")
        result2.write("MAE" + "," +str(mae) + "\n")
        result2.write("R-SQUARED" + "," +str(r2) + "\n")
        result2.write("RMSE" + "," +str(rms) + "\n")
        result2.write("ACCURACY" + "," +str((ac)) + "\n")
        result2.close()
        
        
        df =  pd.read_csv('results/RFMetrics.csv')
        acc = df["Value"]
        alc = df["Parameter"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
        explode = (0.1, 0, 0, 0, 0)  
        
        fig = plt.figure()
        plt.bar(alc, acc,color=colors)
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title(' Random Forest Metrics Value')
        fig.savefig('results/RFMetricsValue.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
#process("kidney_disease.csv")
