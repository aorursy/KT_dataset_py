# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing,tree, linear_model

import datetime

import keras



from keras.models import Sequential

from keras.layers import Dense

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import Counter

from subprocess import check_output

import numpy as np

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test.shape

train

#test=test.fillna()

#train=train.fillna(-999)

test

#train['live']

test.info()

train.info()

#pc=pd.get_dummies(train.Survived,prefix='Survived')

#train=pd.concat((train,pc),1, ignore_index=True)

#train.drop('Survived',1)

'''

pc=pd.get_dummies(train.Survived,prefix='Survived')

train=pd.concat((train,pc))

train.drop('Survived',1)

'''

train
p=Counter(train["Embarked"])

X=train.drop(['PassengerId','Survived','Ticket','Cabin','Name'],1)

Y=train[['Survived']]

id_=test['PassengerId']

test=test.drop(['PassengerId','Ticket','Cabin','Name'],1)

test['Age']=test.Age.fillna(test.Age.mean())

X['Age']=X.Age.fillna(X.Age.mean())

test['Fare']=test.Fare.fillna(test.Fare.mean())

X['Embarked']=X.Embarked.fillna('C')

wh=pd.DataFrame(X)


s=Counter(X['Embarked'])

s=dict(s)

q={}

k=0

for i in s:

    if type(i)!=int:

        q[i]=k

    k+=1

q

p=[]

for i in X['Embarked']:

    if type(i)!=int:

        p.append(q[i])

    else:

        p.append(i)

X['Embarked']=p

X

s=Counter(X['Sex'])

s=dict(s)

q={}

k=0

for i in s:

    if type(i)!=int:

        q[i]=k

    k+=1

q

p=[]

for i in X['Sex']:

    if type(i)!=int:

        p.append(q[i])

    else:

        p.append(i)

X['Sex']=p

X



s=Counter(test['Sex'])

s=dict(s)

q={}

k=0

for i in s:

    if type(i)!=int:

        q[i]=k

    k+=1

q

p=[]

for i in test['Sex']:

    if type(i)!=int:

        p.append(q[i])

    else:

        p.append(i)

test['Sex']=p





s=Counter(test['Embarked'])

s=dict(s)

q={}

k=0

for i in s:

    if type(i)!=int:

        q[i]=k

    k+=1

q

p=[]

for i in test['Embarked']:

    if type(i)!=int:

        p.append(q[i])

    else:

        p.append(i)

test['Embarked']=p





#X=np.array(X)

#Y=np.array(Y)

#test=np.array(test)





#X[X['Embarked']=='Q']
model=Sequential()

model.add(Dense(64,activation='relu',input_dim=7))

#model.add(Dense(64,activation='relu'))

#model.add(Dense(64,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X=preprocessing.scale(X)

test=preprocessing.scale(test)

model.fit(X[:],Y[:],batch_size=64,epochs=100)

#model.evaluate(test,)

pred=model.predict(test)

print(model.evaluate(X[760:],Y[760:]))

#pred=[round(i) for i in pred]

pred=[round(i[0]) for i in pred]

kp=0



fil={'PassengerId':[],'Survived':[]}

for i,j in zip(id_,pred):

    fil['PassengerId'].append(i)

    fil['Survived'].append(int(j))

fil=pd.DataFrame(fil)



print(fil)

fil.to_csv('titanic_pred2.csv',index=False)
#geek=pd.read_csv('titanic_pred2.csv')
#geek
Z = pd.concat((X,Y),axis = 1)

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.title('Korelacja cech')

sns.heatmap(Z.corr(),cmap = colormap)


md = linear_model.LogisticRegression()

X_train, X_test, y_train, y_test = model_selection.train_test_split(

     X, Y,test_size=0.33, random_state=42)



md.fit(X_train,y_train)

l = md.score(X_train,y_train)

l1 = md.score(X_test,y_test)
md1 = tree.DecisionTreeClassifier()

md1.fit(X_train,y_train)

tr = md1.score(X_train,y_train)

tr

tr1 = md.score(X_test,y_test)

sb = tree.export_graphviz(md1,out_file=None, feature_names=X.columns,  

                         class_names='01',  

                         filled=True, rounded=True,  

                         special_characters=True, max_depth = 1 )

import graphviz

graph = graphviz.Source(sb)

graph

X
zw = pd.DataFrame({'Nazwa':['z_treningowy','z_testowy'],'Skuteczność':[l,l1]})

sns.barplot(x='Skuteczność', y = 'Nazwa', data = zw, color = 'b')

md1.get_params
zw = pd.DataFrame({'Nazwa':['z_treningowy','z_testowy'],'Skuteczność':[tr,tr1]})

sns.barplot(x='Skuteczność', y = 'Nazwa', data = zw, color = 'b')
X['Pclass']