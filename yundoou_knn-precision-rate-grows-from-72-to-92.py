import pandas as pd

import numpy as np

import matplotlib

matplotlib.style.use('seaborn')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

path='../input/glass.csv'



df=pd.read_csv(path)

df.head()
df.shape
df.describe()
type=df['Type'].groupby(df['Type']).count()

type
type.plot('bar')
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



# get column titles except the last column

features=df.columns[:-1].tolist()



# get data set features

X=df[features].values

# get labels

y=df['Type'].values



# split data to train data set and test data set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# store scores of KNN model by K=1 to 31

scores=[]



# loop k from 1 to 31, and get cross validation score of each K value

for k in range(1,32):

    knn=KNeighborsClassifier(k)

    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)

    score_mean=score_val.mean()

    scores.append(score_mean)



# get index of maxium score along axis, default axis=0 for 1 dimensional array

best_k=np.argmax(scores)+1

print(best_k)

# generate KNN model

knn=KNeighborsClassifier(best_k)

# fit with train data set

knn.fit(X_train,y_train)

# get Modes presicion rate using test set

print("prediction precision rate:",knn.score(X_test,y_test))
df3=df[df['Type']==3]
df3=pd.concat([df3]*4)
df5=df[df['Type']==5]
df5=pd.concat([df5]*5)
df6=df[df['Type']==6]
df6=pd.concat([df6]*7)
df7=df[df['Type']==7]
df7=pd.concat([df7]*2)
df1=df[df['Type']==1]
df2=df[df['Type']==2]
df_balanced=pd.concat([df1,df2,df3,df5,df6,df7])
df_balanced.shape
df.head()
type=df_balanced['Type'].groupby(df_balanced['Type']).count()

type
type.plot('bar')
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



# df.columns is column labels property

features=df_balanced.columns[:-1].tolist()

X=df_balanced[features].values

y=df_balanced['Type']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

score=[]

for i in range(32):

    knn=KNeighborsClassifier(k)

    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)

    score_mean=score_val.mean()

    scores.append(score_mean)

best_K=np.argmax(scores)+1

print('best K is:',best_K)

knn=KNeighborsClassifier(best_K)

knn.fit(X_train,y_train)

print("prediction precision rate:",knn.score(X_test,y_test))

df_balanced.iloc[:,:-1].boxplot()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing



# df.columns is column labels property

features=df_balanced.columns[:-1].tolist()

X=df_balanced[features].values

y=df_balanced['Type']



# normalization

min_max_scaler=preprocessing.MinMaxScaler()

X_minmax=min_max_scaler.fit_transform(X)



X_train,X_test,y_train,y_test=train_test_split(X_minmax,y,test_size=0.2,random_state=1)

score=[]

for i in range(32):

    knn=KNeighborsClassifier(k)

    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)

    score_mean=score_val.mean()

    scores.append(score_mean)

best_K=np.argmax(scores)+1

print('best K is:',best_K)

knn=KNeighborsClassifier(best_K)

knn.fit(X_train,y_train)

print("prediction precision rate:",knn.score(X_test,y_test))
X
X_minmax
df_balanced.head()
corr=df_balanced.iloc[:,:-1].corr()

corr
from pandas.plotting import scatter_matrix

sm=scatter_matrix(df_balanced.iloc[:,:-1], alpha=1, figsize=(10, 10), diagonal='kde')



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from sklearn import decomposition



# df.columns is column labels property

features=df_balanced.columns[:-1].tolist()

X=df_balanced[features].values

y=df_balanced['Type']



# PCA

pca=decomposition.PCA(n_components=7)

pca.fit(X)

X=pca.transform(X)

print("Primary Components:",X)



# normalization

min_max_scaler=preprocessing.MinMaxScaler()

X_minmax=min_max_scaler.fit_transform(X)



X_train,X_test,y_train,y_test=train_test_split(X_minmax,y,test_size=0.2,random_state=1)

score=[]

for i in range(32):

    knn=KNeighborsClassifier(k)

    score_val=cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10)

    score_mean=score_val.mean()

    scores.append(score_mean)

best_K=np.argmax(scores)+1

print('best K is:',best_K)

knn=KNeighborsClassifier(best_K)

knn.fit(X_train,y_train)

print("prediction precision rate:",knn.score(X_test,y_test))

result=knn.predict(X_test)

print(result)

myarray = np.asarray(y_test.tolist())

print(myarray)
s=pd.DataFrame(X_test)

t=pd.DataFrame(y_test)

s.head()
t=t.reset_index()
t.head()
del t['index']
t.head()
X_test_u=pd.concat([s,t],axis=1)

X_test_u=X_test_u.drop_duplicates()

X_test_u.shape
X_test=X_test_u.iloc[:,:-1].values

y_test=X_test_u['Type']
print("prediction precision rate:",knn.score(X_test,y_test))

result=knn.predict(X_test)

print(result)

myarray = np.asarray(y_test.tolist())

print(myarray)