import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
my_filepath = "../input/breast-cancer-csv/breastCancer.csv"
my_data=pd.read_csv("../input/breast-cancer-csv/breastCancer.csv")
my_data.head()
my_data['class'].value_counts()
my_data.shape
my_data.dtypes
my_data['bare_nucleoli']
my_data[my_data['bare_nucleoli']=='?']
my_data[my_data['bare_nucleoli']=='?'].sum()
digits_in_bare_nucleoli=my_data.bare_nucleoli.str.isdigit()
digits_in_bare_nucleoli
my_df=my_data.replace('?',np.nan)
my_df.bare_nucleoli
my_df.median()
my_df.describe
my_df=my_df.fillna(my_df.median())
my_df.bare_nucleoli
my_df.dtypes
my_df['bare_nucleoli']=my_df['bare_nucleoli'].astype('int64')
my_df.dtypes
my_df.head()
my_df.drop('id',axis=1,inplace=True)
my_df.head()
my_df.describe().T
import seaborn as sns
sns.distplot(my_df['class'])
my_df.hist(bins=20, figsize=(40,40),layout=(6,3));
plt.figure(figsize=(25,20))

sns.boxplot(data=my_df, orient='h')
my_df.corr()
plt.figure(figsize=(40,20))



sns.heatmap(my_df.corr(), vmax=1, square=True, annot=True,cmap='viridis')

plt.title('Correlation Between Different Attributes')

plt.show()
try:

    sns.distplot(my_df)

except RuntimeError as re:

    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

        sns.distplot(my_df, kde_kws={'bw': 0.1})

    else:

        raise re
try:

    sns.pairplot(my_df)

except RuntimeError as re:

    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

        sns.pair(my_df, kde_kws={'bw': 0.1})

    else:

        raise re
my_df.head()
X=my_df.drop('class',axis=1)

y=my_df['class']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5, weights='distance')
KNN.fit(X_train,y_train)
predicted_1=KNN.predict(X_test)

predicted_1
from scipy.stats import zscore
print("KNeighborsClassifier Algorithm has predicted {0:2g}%".format(KNN.score(X_test,y_test)*100))
from sklearn.svm import SVC
svc=SVC(gamma=0.025, C=3)

svc.fit(X_train,y_train)
prediction_2=svc.predict(X_test)

prediction_2
print("Support Vector Machine Algorithm has predicted {0:2g}%".format(svc.score(X_test,y_test)*100))
Knn_Predictions=pd.DataFrame(predicted_1)

Svc_Predictions=pd.DataFrame(prediction_2)
df_new=pd.concat([Knn_Predictions,Svc_Predictions],axis=1)
df_new.columns=[['Knn_Predictions','Svc_Predictions']]

df_new
from sklearn.metrics import classification_report
print('KNN Classification Report')



print('>>>'*10)



print(classification_report(y_test,predicted_1))
print('SVC Classification Report')



print('>>>'*10)



print(classification_report(y_test,prediction_2))
!pip install lazypredict
import lazypredict

from lazypredict.Supervised import LazyClassifier
data = my_df

X=my_df.drop('class',axis=1)

y=my_df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =1)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

models,predictions = clf.fit(X_train, X_test, y_train, y_test)

models
print('Top 10 Performing Models')

print('>>>'*10)

models.head(10)
from sklearn import metrics



print('KNN Confusion Matrix')



cm=metrics.confusion_matrix(y_test,predicted_1, labels=[2,4])



df_cm=pd.DataFrame(cm, index=[i for i in [2,4]],columns=[i for i in ['Predict M','predict B']])



plt.figure(figsize=(10,8))

sns.heatmap(df_cm, annot=True)
from sklearn import metrics



print('SVC Confusion Matrix')



cm=metrics.confusion_matrix(y_test,prediction_2, labels=[2,4])



df_cm=pd.DataFrame(cm, index=[i for i in [2,4]],columns=[i for i in ['Predict M','predict B']])



plt.figure(figsize=(10,8))

sns.heatmap(df_cm, annot=True)