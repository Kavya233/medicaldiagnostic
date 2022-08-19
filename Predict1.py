import os
import pandas as pd
import numpy as np
import csv
import glob
from sklearn.linear_model import LogisticRegression



def process(data):
        X_train=pd.read_csv("liver_data.csv",usecols=[0,1,2,3,4,5,6,7,8,9],header=None)       
        y_train=pd.read_csv("liver_data.csv",usecols=[10],header=None)
        data=np.array(data)
        data=data.reshape(1, -1)
        X_test=data
        model2=LogisticRegression(random_state = 0)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        print("predicted")
        print(y_pred)
        result=""
        if y_pred[0]==1:
                result="No Disease"
        else:
            result="Liver Disease"
            
        
        return result
        
        

