import os

print(os.listdir("../input"))

import numpy as np

import pandas as pd

import seaborn as sns 

from sklearn import svm

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.models import Sequential 

from sklearn.linear_model import LogisticRegression

from keras.layers import Dense

from keras import regularizers

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier


tdata = pd.read_csv("../input/Integrated.csv",header=None,na_values=[-9])



new_data = tdata[[2,3,8,9,14,15,16,17,18,31,57]].copy()



new_data.columns = ['Age','Sex','Chest Pain','Blood Pressure','Smoking Years','Fasting Blood Sugar','Diabetes History','Family history Cornory','ECG','Pulse Rate','Target']



print(new_data.info())



new_data['Blood Pressure'].fillna(new_data['Blood Pressure'].mean(),inplace=True)

new_data['Smoking Years'].fillna(new_data['Smoking Years'].mean(),inplace=True)

new_data['Fasting Blood Sugar'].fillna(new_data['Fasting Blood Sugar'].mode()[0],inplace=True)

new_data['Diabetes History'].fillna(new_data['Diabetes History'].mode()[0],inplace=True)

new_data['Family history Cornory'].fillna(new_data['Family history Cornory'].mode()[0],inplace=True)

new_data['ECG'].fillna(new_data['ECG'].mean(),inplace=True)

new_data['Pulse Rate'].fillna(new_data['Pulse Rate'].mean(),inplace=True)





print(new_data.info())
class0 = new_data.loc[(new_data['Target'] == 0)]

class0 = class0[0:150][:]

class1 = new_data.loc[(new_data['Target'] == 1)]

class2 = new_data.loc[(new_data['Target'] == 2)]

class3 = new_data.loc[(new_data['Target'] == 3)]

class4 = new_data.loc[(new_data['Target'] == 4)]

new_data = class0.append(class1)

new_data = new_data.append(class2)

new_data = new_data.append(class3)

new_data = new_data.append(class4)





print(new_data.info())

print(new_data.describe())



sns.set(style="ticks", color_codes=True)

ax = sns.pairplot(new_data,palette="husl")

plt.show()





print(new_data['Target'].value_counts())



df_dummies = pd.get_dummies(new_data['Target'])



new_data = pd.concat([new_data , df_dummies], axis=1)

new_data.columns

features = [                   'Age',                    'Sex',

                   'Chest Pain',         'Blood Pressure',

                'Smoking Years',    'Fasting Blood Sugar',

             'Diabetes History', 'Family history Cornory',

                          'ECG',             'Pulse Rate',

                                               0,

                              1,                        2,

                              3,4]





new_data = new_data[features]

#new_data.head()

data = new_data.values



#________ Target and Features Split ______



X = data[:,:-1]

y = data[:,-1]



y = y.reshape((y.shape[0],1))



#_____ Normalization _____



n_X = preprocessing.normalize(X)

n_y = preprocessing.normalize(y)

n_y = n_y.reshape((n_y.shape[0],))



 #__________________  Train_Test_Split ___________________



training_X, testing_X, training_y, testing_y = train_test_split(n_X,n_y,test_size=0.20,random_state=42)



print('Training data: '+str(training_X.shape) +' '+ str(training_y.shape))

print('Testing  data: '+str(testing_X.shape) +' '+str(testing_y.shape))



#______________  Using Support Vector Machine _____



print('Support Vector Machine')

clf = svm.SVC(kernel='rbf',C=5,gamma='auto')

clf.fit(training_X,training_y)

r = clf.score(testing_X,testing_y)

print(r)

#_______  Using Logistic Regression _______



print('Logistic Regression')

clf = LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial')

clf.fit(training_X,training_y)

clf.predict(testing_X)

r = clf.score(testing_X,testing_y)

print(r)

# _____ Using KNN _____

print('K Nearest Neighbors')

K_Values = range(1,26)

scores = {}

scores_l = []



for k in K_Values:

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(training_X,training_y)

    y_prediction = knn.predict(testing_X)

    scores[k] = metrics.accuracy_score(testing_y,y_prediction)

    print('for k = ' + str(k) + ' the accuracy is : '+ str(scores[k]))

    scores_l.append(scores[k])



    

training_y = np.expand_dims(training_y, axis=1)


#_______________ Using Neural Net ___________

print('Artificial Neural Net')

model = Sequential()

model.add(Dense(units=32,activation='sigmoid',input_dim=14,kernel_regularizer=regularizers.l2(0.01),

                activity_regularizer=regularizers.l1(0.01)))

model.add(Dense(units=1024,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),

                activity_regularizer=regularizers.l1(0.01)))

model.add(Dense(units=1024,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),

                activity_regularizer=regularizers.l1(0.01)))

model.add(Dense(units=5,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(training_X,training_y,epochs=50,batch_size=12)

loss_and_metrics = model.evaluate(testing_X,testing_y,batch_size=12)

loss_and_metrics
