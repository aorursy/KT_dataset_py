# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Veriyi okuyup, uygulanabilir plot tiplerini kontrol ediyoruz.

data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

print(plt.style.available)

plt.style.use('ggplot')
# Features'ları görmek için

data.head()
# Veri tiplerini, NaN değerlerin olup olmadığını ve 

# verinin uzunluğunu kontrol ediyoruz



data.info()
data.describe()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



print('(K=3) için doğruluk: ',knn.score(x_test,y_test))
neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []



# farklı k değerleri için loop



for i, k in enumerate(neig):

    

    knn = KNeighborsClassifier(n_neighbors=k)

    # knn ile fitleme

    knn.fit(x_train,y_train)

    #train doğruluğu

    train_accuracy.append(knn.score(x_train, y_train))

    # test doğruluğu

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Test Doğruluğu')

plt.plot(neig, train_accuracy, label = 'Train Doğruluğu')

plt.legend()

plt.title('Değer VS Doğruluk')

plt.xlabel('Komşu sayıları')

plt.ylabel('Doğruluk')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("En iyi doğruluk: {} K = {} için.".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
data1 = data[data['class'] =='Abnormal']

x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)



# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# LinearRegrasyon

from sklearn.linear_model import LinearRegression

reg = LinearRegression()



predict_space = np.linspace(min(x), max(x)).reshape(-1,1)



reg.fit(x,y)



predicted = reg.predict(predict_space)

# R^2 

print('R^2 değeri: ',reg.score(x, y))



plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# Veriyi yükle.

data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

# get_dummies

df = pd.get_dummies(data)

df.head()
# drop class_Normal

df.drop("class_Normal",axis = 1, inplace = True) 

df.head()
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)

plt.xlabel('pelvic_radius')

plt.xlabel('degree_spondylolisthesis')

plt.show()
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

data3 = data.drop('class',axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data3)

labels = pipe.predict(data3)

df = pd.DataFrame({'labels':labels,"class":data['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)