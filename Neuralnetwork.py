import glob
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import tensorflow.keras as k
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
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
        X_train,  X_test, y_train, y_test = train_test_split(
                X, y, test_size= 0.2, shuffle=True)
        #Build The model

        model = Sequential()
        model.add(Dense(256, input_dim=len(X.columns),              kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
        model.add(Dense(1, activation="hard_sigmoid"))
        #Compile the model
        model.compile(loss='binary_crossentropy', 
                          optimizer='adam', metrics=['accuracy'])

        #Train the model
        history = model.fit(X_train, y_train, 
                            epochs=2000, 
                            batch_size=X_train.shape[0])
        #Save the model
        model.save("ckd.model")
        #Visualize the models accuracy and loss
        plt.plot(history.history["acc"])
        plt.plot(history.history["loss"])
        plt.title("model accuracy & loss")
        plt.ylabel("accuracy and loss")
        plt.xlabel("epoch")
        plt.legend(['acc', 'loss'], loc='lower right')
        plt.show()
        print("---------------------------------------------------------")
        print("Shape of training data: ", X_train.shape)
        print("Shape of test data    : ", X_test.shape )
        print("---------------------------------------------------------")
