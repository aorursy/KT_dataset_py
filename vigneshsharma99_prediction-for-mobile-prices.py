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
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

df.head()
df.isnull().sum()
x = df.drop(columns=['price_range'])

print(x)
y = df['price_range']

print(y)
from matplotlib import pyplot

import matplotlib.pyplot as plt

pyplot.plot(x, y)

pyplot.show()
#plotting histogram 

plt.hist(df['battery_power'],rwidth=0.9,alpha=0.3,color='blue',bins=15,edgecolor='red') 



#x and y-axis labels 

plt.xlabel('price_range') 

plt.ylabel('battery_power') 



#plot title 

plt.title('Price Range') 

plt.show();
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
nb = GaussianNB()

nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print(y_pred)



print(accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(x_test)





print("Accuracy:",accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier  

classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  

classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)  
#Creating the Confusion matrix  

from sklearn.metrics import confusion_matrix  

cm= confusion_matrix(y_test, y_pred)  
print(accuracy_score(y_test, y_pred))
#plotting histogram 

plt.hist(df['fc'],rwidth=0.9,alpha=0.3,color='blue',bins=15,edgecolor='red') 

#x and y-axis labels 

plt.xlabel('price_range') 

plt.ylabel('battery_power') 

#plot title 

plt.title('Mobile Price Classification') 

plt.show();
import seaborn as sns

sns.factorplot('fc',data=df,kind='count')
df['talk_time'].hist(bins=70)
sns.lmplot('pc','price_range',data=df)
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# predictions

rfc_predict = rfc.predict(x_test)



print("Accuracy:",accuracy_score(y_test, rfc_predict))