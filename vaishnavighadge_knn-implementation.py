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
df=pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

df.head()
df.shape
missing=[feature for feature in df.columns if df[feature].isnull().sum()>1]

for feature in missing:

    print('feature is {} with nan values {}'.format(feature,np.rounf(df[feature],4)))
df.dtypes
nm=[feature for feature in df.columns if df[feature].dtypes!='O']

nm
discrete_feature=[feature for feature in nm if len(df[feature].unique())<25]

print("Discrete Variables Count: {}".format(len(discrete_feature)))
discrete_feature
con=[feature for feature in nm if feature  not in  discrete_feature ]

con
df['fruit_name'].unique()
dict1={'apple':1, 'mandarin':2, 'orange':3, 'lemon':4}

dict1
df['fruit_name_new']=df['fruit_name'].map(dict1)
df.head(30)
df=df.drop(['fruit_name','fruit_label'],axis=1)
df.head()
df['fruit_subtype'].unique()
df['fruit_subtype']=pd.get_dummies(df['fruit_subtype'],drop_first=True)
df.head(30)
df.dtypes
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
for feature in 'width','height','mass','color_score':

    data=df.copy()

    data[feature].hist(bins=30)

    plt.xlabel(feature)

    plt.ylabel('fruit_name_new')

    plt.show()
df.head()
X=df.drop(['fruit_name_new'],axis=1)

X.head()
y=df['fruit_name_new']

y.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X,y)
scaled_feature=scaler.transform(df.drop(['fruit_name_new'],axis=1))
df_feat = pd.DataFrame(scaled_feature,columns=df.columns[:-1])

df_feat.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(scaled_feature,y,test_size=0.20,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


pred = knn.predict(X_test)
pred


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))



print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# NOW WITH K=25

knn = KNeighborsClassifier(n_neighbors=25)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=25')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.linear_model import LogisticRegression

reg=LogisticRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

y_pred
print(confusion_matrix(y_pred,y_test))
reg.score(X_test,y_test)