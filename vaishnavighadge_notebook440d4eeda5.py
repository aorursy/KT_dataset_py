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
pd.pandas.set_option('display.max_columns',None)

df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
import seaborn as sns

import matplotlib.pyplot as plt
df.isnull().sum()
catagatical=[feature for feature in df.columns if df[feature].dtypes=='O']

catagatical
numerical=[feature for feature in df.columns  if df[feature].dtypes!='O']

df[numerical].head()
#chcek discrete feature

discrete=[feature for feature in numerical if len(df[feature].unique())<25]

discrete
#chrck continous features

continous=[feature for feature in numerical if feature not in discrete  ]

continous
#analuze continous values by plotting hiatogram

for feature in continous:

    data=df.copy()

    data[feature].hist(bins=30)

    plt.xlabel(feature)

    plt.ylabel('Class')

    plt.title(feature)

    plt.show()

    
import scipy.stats as stats

def diagnostic_plots(df, variable):

    # function to plot a histogram and a Q-Q plot

    # side by side, for a certain variable

    

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    df[variable].hist()



    plt.subplot(1, 2, 2)

    stats.probplot(df[variable], dist="norm", plot=plt)



    plt.show()

for feature in continous:

    data[feature]=np.log(data[feature]+1)

    diagnostic_plots(data,feature)
#check outliers

for feature in continous:

    data=df.copy()

    if 0 in df[feature].unique():

        pass

    else:

        data[feature]=np.log(df[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()

     

   
X=df.drop(['Class'],axis=1)

X
y=df['Class']

y
#check imbalalencing

LABELS=['Normal','Fraud']

import matplotlib.pyplot as plt

count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
#from above graph we see that data imbalencing is much more so covert them into balenced dataset

#to check it in values use following piece of code



fraud = df[df['Class']==1]

normal = df[df['Class']==0]

print(fraud.shape,normal.shape)
from imblearn.combine import SMOTETomek

from imblearn.under_sampling import NearMiss

# Implementing Oversampling for Handling Imbalanced 

smk = SMOTETomek(random_state=42)

X_res,y_res=smk.fit_sample(X,y)
X_res.shape,y_res.shape
from collections import Counter

print('Original dataset shape {}'.format(Counter(y)))

print('Resampled dataset shape {}'.format(Counter(y_res)))
count=pd.value_counts(y_res)

count.plot(kind='bar',rot=0,color=['violet'])

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=0)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

y_pred
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))