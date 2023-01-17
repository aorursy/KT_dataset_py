# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.DataFrame(np.random.uniform(low=0.1, high=499.9, size=(1000, )), columns=['velocity'])
df3=pd.DataFrame(np.random.uniform(low=0.1, high=89.9, size=(1000, )), columns=['angle'])
df=pd.concat([df1, df3], axis=1)
print( df.head(), df.info())

#this is where we calculate the height and drange of projectile motion
for i in df['velocity'].values:#if you want to make some math on each item of a column use df.column.values
    R=[]
    H=[]
    for j in df['angle'].values:
        g=9.81
        tmax=(2*i)*np.sin(np.deg2rad(j))/g #i used the formula with time because it give me more desirable results
        r=np.cos(np.deg2rad(j))*i*tmax
        yükseklik=abs(((np.sin(i))*i)-0.5*(tmax**2)*g)
        #yükseklik=(i**2)*((np.sin(np.deg2rad(j))**2)/(g)) # you can use this part if you want
        #r=abs((i**2)*(np.sin(np.deg2rad(2*j))/(2*g)))
        H.append(yükseklik)
        R.append(r)
np.asarray(R)
np.asarray(H)

df2=pd.DataFrame(R, columns=['Distance'])
df4=pd.DataFrame(H, columns=['Height'])
df5=pd.concat([df, df2 ,df4], axis=1)#concat the calculated parts and our random generator's velocity and angle
print(df5.head())
print(df5.info())
df5.describe()#to see mean std... of data
def draw():
    G=9.81
    v=df5['velocity'].values
    theta=df['angle'].values
    plt.figure(figsize=[15, 8])
    tmax=((2*v)*np.sin(theta))/G
    time=tmax*np.linspace(0, 1, 1000)[:,None]#we generate time here
    x=((v*time)*np.cos(theta))
    y=((v*time)*np.sin(theta))-(0.5*G*(time**2))
    plt.plot(x, y)
    plt.show()
draw()    
df5.fillna('0')
X=df5['velocity']
Y=df5['Height']
X=np.array(X).reshape(len(X), 1)#reshape the data 
Y=np.array(X).reshape(len(X), 1)
sc=StandardScaler()#scale the data
sc.fit(X, Y)
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.33)#split the data to train and test
knn=KNeighborsRegressor()#our first machine learning algoritm is KNR
knn.fit(x_train, y_train)
X2=knn.predict(x_test)
plt.figure()
plt.plot(X2, x_test)
plt.show()
print(knn.score(x_train, y_train))



ll=LassoLars()
ll.fit(x_train, y_train)
X3=ll.predict(x_test)
plt.figure()
plt.plot(X3, x_test)
plt.show() 
print("accuracy", ll.score(x_train, y_train))

svm=SVR(kernel='linear')
svm.fit(x_train, y_train)
X4=svm.predict(x_test)
print("accuracy: ", svm.score(x_train, y_train))