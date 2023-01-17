# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from collections import Counter
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv",sep=",")
df.head() #data.tail() ile aynı
Counter(df["class"])
A=df[df["class"] == "Abnormal"]
N=df[df["class"] == "Normal"]

#Scatter Plot
plt.scatter(A.pelvic_incidence,A.pelvic_radius,color="red",label="kotu",alpha= 0.3)
plt.scatter(N.pelvic_incidence,N.pelvic_radius,color="green",label="iyi",alpha= 0.3)
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.legend()
plt.show()
#Defining variables as x and y. Also y variable defines as numerical like 0 and 1.
df["class"]= [1 if each == "Normal" else 0 for each in df["class"]]
y = df["class"].values
x_data = df.drop(["class"],axis=1)

# normalization . With this formula, we prevent big numbers that will able to cause wrong prediction.
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split. Split data as train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#KNN Prediction
#knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1) #n_neighbors=k=komşu sayısı
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
prediction

print("{} nn score {}".format(3,knn.score(x_test,y_test))) #k=3
#k sayısının formülle belirlenmesi
score_list = []
for each in range(1,30): #burada range aralığını biz belirleyebiliyoruz.
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    knn2.score(x_test,y_test)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,30),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")