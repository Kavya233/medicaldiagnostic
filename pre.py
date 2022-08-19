import glob
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import pandas as pd
#import keras as k
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import csv
def process(path):
        
        df = pd.read_csv(path)
            
        #Print the first 5 rows
        df.head()
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
        df.to_csv("Preprocessed_kd_dataset.csv")
        print(df.head(5))
