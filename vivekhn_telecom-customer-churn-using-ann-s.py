# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading the dataset



df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Examining the data



df.head()
# Dropping customerID column



df=df.drop(["customerID"],axis=1)
# Examining the data types



df.dtypes
# Casting TotalCharges to float



df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors="coerce")
# checking the nulls



df.isnull().sum().plot.bar(figsize=(12,8))
# Drop Null values



df=df.dropna()

df.isnull().sum()
# Label Encoding for gender and churn



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["gender"]= le.fit_transform(df["gender"])

df["Churn"]= le.fit_transform(df["Churn"])
# Defining the X and y



X = df.loc[:,df.columns!="Churn"]

y = df["Churn"]

y=y.values.reshape(-1,1)
# Splitting the training and testing data



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# Filtering all Categorical variables



vars_categorical = list(df.select_dtypes(['object']).columns)

print(vars_categorical)
# One-Hot Encoding



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto', drop='first', sparse=False) 

encoder.fit(X_train[vars_categorical])

X_train_enc = encoder.transform(X_train[vars_categorical])

X_test_enc = encoder.transform(X_test[vars_categorical]) 

# Verify the encoded output



X_train_enc

# Loading the libraries



import tensorflow.keras

from keras.layers import Dense

from keras.models import Sequential 

# Initiate the model



model = Sequential()



# adding input layers



model.add(Dense(64, input_dim=25, activation="relu"))

model.add(Dense(32,activation="relu"))

model.add(Dense(16,activation="relu"))

model.add(Dense(8,activation="relu"))

model.add(Dense(1,activation="sigmoid"))



# printing the summary



model.summary()
# Compiling the model



model.compile(optimizer="adam",loss="binary_crossentropy", metrics=['accuracy'])

X_train_enc.shape
# Training the model



epochs_hist= model.fit(X_train_enc,y_train,epochs=100,batch_size=25)
# Plotting the accuracy



import matplotlib.pyplot as plt

plt.plot(epochs_hist.history["accuracy"])

plt.title('Model Accuracy plot')

plt.ylabel('Accuracy')

plt.xlabel('Epoch Number')

plt.legend(['Training Accuracy'])
# Predicting the test samples



y_pred=model.predict(X_test_enc)
# Converting y_pred to binary



y_pred = (y_pred > 0.5)
# Printing the confusion matrix



from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_pred,y_test)

print(cm)
# Printing the accuracy



accuracy_score(y_pred,y_test)