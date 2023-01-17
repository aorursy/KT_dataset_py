import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# importing necessary libraries 

from sklearn import svm

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 
# load the faulty data of gas turbine engine

#training_data = pd.read_csv('GT_actuator_fault.csv')

training_data = pd.read_csv('../input/GT_faulty_sensor_data_T2.csv')



# reduce data size

training_data= training_data.sample(n = 500)

print(training_data.shape)



training_data = training_data.values

#training_data = np.take(training_data,np.random.permutation(training_data.shape[0]),axis=0,out=training_data);



X= training_data[:,[0,1]]

y = training_data[:,2] 
dot_size = 60

def show_scatter(X, y, filename):

    plt.scatter(X[:,0], X[:,1], c=y, s=dot_size, cmap = 'cool')

    plt.grid()

    plt.show()

    



show_scatter(X, y, 'Compressor exit temperature')

# show_scatter(X_noisy1, y_noisy1, 'KNN-smile-data-clean')
# dividing X, y into train and test data 

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0, 10))

training = scaler.fit_transform(X)





X_train, X_test, y_train, y_test = train_test_split(training, y, random_state = 0) 

X_train.shape
show_scatter(X_train, y_train, 'Compressor exit temperature')
# training a DescisionTreeClassifier 

from sklearn.tree import DecisionTreeClassifier 

dtree = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train) 

dtree_predictions = dtree.predict(X_test) 

  

# creating a confusion matrix 

cm = confusion_matrix(y_test, dtree_predictions) 

print(cm)
# training a linear SVM classifier 

from sklearn.svm import SVC 



svm_linear = SVC(kernel='linear', C = 1).fit(X_train, y_train) 

svm_predictions = svm_linear.predict(X_test) 



# creating a confusion matrix 

cm = confusion_matrix(y_test, svm_predictions) 

print(cm)
# training a KNN classifier 

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train) 

  



# creating a confusion matrix 

knn_predictions = knn.predict(X_test)  

cm = confusion_matrix(y_test, knn_predictions) 

print(cm)

# training a Naive Bayes classifier 

from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_train, y_train) 

gnb_predictions = gnb.predict(X_test) 



# creating a confusion matrix 

cm = confusion_matrix(y_test, gnb_predictions) 

print(cm)
# create a mesh to plot in

h = 0.02

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1

y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1



xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



# title for the plots

titles = ['Descision tree',

            'LinearSVC',

            'K neighbour',

            'Naive Bayesl']

C =1.0
plt.figure(figsize=(10,8))

for i, clf in enumerate((dtree, svm_linear, knn, gnb)):

    plt.subplot(2, 2, i + 1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='cool', alpha=1.0)

    # Plot also the training points

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='spring')

    plt.xlabel('Fuel flow rate (L/hr)')

    plt.ylabel('Compressor exit temperature (K)')

    plt.title(titles[i])

    



plt.show()