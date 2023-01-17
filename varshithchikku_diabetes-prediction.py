import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.describe(include = 'all').T
df.dtypes
plt.style.use('ggplot')
print(plt.style.available)
plt.figure(figsize = (10,10))
df.plot()
current_palette = sns.color_palette()
sns.countplot(x = 'Outcome', data = df)
df['Outcome'].value_counts()

#df['Age'].value_counts().head()
df.groupby('Outcome').Age.mean()
plt.figure(figsize=(10,10))
sns.jointplot(x= 'Outcome', y='Age', data = df, kind = 'kde')

g = sns.FacetGrid( col = 'Pregnancies', data = df)
#g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
#bins = np.linspace(0, 20, 20)
g.map(plt.hist, "Outcome", lw=0)
plt.figure(figsize=(10,10))
sns.violinplot(x= 'Outcome', y= 'Pregnancies', data = df)
plt.show()
plt.figure(figsize=(10,10))
sns.boxplot(x= 'Outcome', y='Glucose', data = df)
#plt.figure(figsize=(10,10))
sns.barplot(x= 'Outcome', y='BloodPressure', data = df)
plt.figure(figsize=(10,12))
sns.factorplot(x= 'Outcome', y='DiabetesPedigreeFunction', data = df)
plt.figure(figsize=(10,10))
sns.boxplot(x= 'Outcome', y='SkinThickness', data = df)
plt.figure(figsize=(10,10))
sns.swarmplot(x= 'Outcome', y='BMI', data = df)
data = df[df.Outcome == 1]
data.head()
sns.pairplot(data=df,hue='Outcome')
plt.show()
plt.figure(figsize = (12,10))
#data['Age'].value_counts(sort = False).plot(kind = 'bar')
sns.countplot(data['Age'])
plt.figure(figsize = (10,8))
sns.swarmplot(x= 'Outcome', y='Age', data = data)
sns.swarmplot(x= 'Outcome', y='BMI', data = data,color = '#228B22')
plt.ylabel('BMI and Age')
#data['Pregnancies'].value_counts(sort = False).plot(kind = 'bar')
plt.figure(figsize = (10,10))
sns.countplot(data['Pregnancies'])
plt.figure(figsize = (20,8))
#data['Glucose'].value_counts(sort = False).plot()
#sns.swarmplot(x= 'Outcome', y='Glucose', data = data,color = '#228B22')
sns.countplot(data['Glucose'])
corr = df.corr()
plt.figure(figsize = (12,12))
sns.heatmap(corr,annot=True )
plt.figure(figsize = (12,12))
sns.heatmap(data.corr(),annot=True )
sns.swarmplot(x ='SkinThickness', data = df)
#df[df.SkinThickness != 99]
df.drop(df.index[579], inplace = True)
X = df.drop('Outcome', axis =1)
#X.head()
y =df['Outcome']
#y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=0)
cross_val_score(clf, X, y, cv=10)

clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf1 = RandomForestClassifier(max_depth=2, random_state=0)
clf1.fit(X_train, y_train)
clf1.predict(X_test)
clf1.score(X_test, y_test)
'''
'''
from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf2.fit(X_train, y_train)
clf2.predict(X_test)
clf2.score(X_test, y_test)
'''
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential() 

#Input later
model.add(Dense(units=500, 
                input_dim=8, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
#Hidden layer 1
model.add(Dense(units=200,  
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))

#Output layer
model.add(Dense(units=1,
                kernel_initializer='uniform', 
                activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',   
              optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=X_train, y=y_train,  
                          validation_split=0.2, epochs=20, 
                          batch_size=50, verbose=2) 

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(X_test, y_test)
print('accuracy=',scores[1])
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
pre = clf.predict(X_test) 
print(clf.score(X_test, y_test))
y_score = clf.decision_function(X_test)
#print(pre)
from sklearn.model_selection import cross_val_score
'''
scores = cross_val_score(clf, X, y, cv=5)
y_score = clf.decision_function(X_test)
print(scores)
'''
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
y_score = clf.decision_function(X_test)
print(scores)
sns.heatmap(matrix, annot=True)
'''
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print(average_precision)
