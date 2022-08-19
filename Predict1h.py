import os
import pandas as pd
import numpy as np
import csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def process(data):
        df=pd.read_csv("heart_diab.csv", dtype=float)
        x=df.iloc[:, :-1].values
        y=df.iloc[:, -1:].values
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        
        
        data=np.array(data)
        data=data.reshape(1, -1)
        X_test=data
        model2=XGBClassifier(random_state = 0)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        print("predicted")
        print(y_pred)
        result=""
        if y_pred[0]==0:
                result="No Disease"
        elif y_pred[0]==1:
                result="Heart Disease found"
        else:
                result="Diabetics found"
                
            
        
        return result
        
        

