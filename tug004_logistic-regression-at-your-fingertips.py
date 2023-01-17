import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, LogisticRegression

from sklearn.preprocessing import MinMaxScaler,LabelEncoder, label_binarize

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc

from keras.datasets import mnist

import warnings 

warnings.filterwarnings('ignore')
train_set = pd.read_csv('../input/income-predictions-dataset2-class-classification/test_new.csv',names = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen'])

test_set = pd.read_csv('../input/income-predictions-dataset2-class-classification/train_new.csv',names = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen'])





total_set = train_set.append(test_set)

total_set.head()


y_train = total_set.iloc[0:30161,-1]

y_test = total_set.iloc[30161:,-1]



total_set.drop('fifteen',axis = 1,inplace = True)



total_set = pd.get_dummies(total_set)



x_train = total_set.iloc[0:30161,:]

x_test = total_set.iloc[30161:,:]





scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)



x_train  = np.c_[np.ones(len(x_train)),x_train]

x_test  = np.c_[np.ones(len(x_test)),x_test]

def sigmoid(z):

    return 1 / (1 + np.exp(-z))
iter = 2000

theta = np.zeros(x_train.shape[1]).T  

alpha = 0.1

n = len(x_train)



cost = np.empty(iter)

for i in range(iter):

    z = x_train.dot(theta)

    Y_pred = sigmoid(z)

    cost[i] = (-1 / n) * (y_train.T.dot(np.log(Y_pred)) + (1 - y_train).T.dot(np.log(1 - Y_pred)) )

    theta = theta - (alpha / n) * (x_train.T.dot(Y_pred - y_train)) 

    

h_theta = x_test.dot(theta)

h_theta[h_theta < 0.5] = 0

h_theta[h_theta >= 0.5] = 1



print("Model accuracy is:" , accuracy_score(y_test,h_theta) * 100)

iter = 2000

theta = np.zeros(x_train.shape[1]).T  

alpha = 0.1

n = len(x_train)

l1_param = 0.01



cost = np.empty(iter)

accuracy = np.empty(iter)

for i in range(iter):

    z = x_train.dot(theta)

    Y_pred = sigmoid(z)

    cost[i] = (-1 / n) * (y_train.T.dot(np.log(Y_pred)) + (1 - y_train).T.dot(np.log(1 - Y_pred)) + (l1_param * (np.sum(np.abs(theta)))))  

    theta = theta - (alpha / n) * (x_train.T.dot(Y_pred - y_train)) 

    h_theta = x_test.dot(theta)

    h_theta[h_theta < 0.5] = 0

    h_theta[h_theta >= 0.5] = 1

    accuracy[i] = accuracy_score(y_test,h_theta)

    





print("Model accuracy is:" , accuracy_score(y_test,h_theta) * 100)



plt.plot(np.arange(0,iter),cost,'r-',label = 'Cost')

plt.plot(np.arange(0,iter),accuracy,'b-',label = 'Accuracy')

plt.xlabel("No of iterations")

plt.ylabel("Percentage")

plt.title("Number of iterations v/s Cost function and Accuracy")

plt.legend()

plt.show()

iter = 2000

theta = np.zeros(x_train.shape[1]).T  

alpha = 0.1

n = len(x_train)

l2_param = 0.01



cost = np.empty(iter)

accuracy = np.empty(iter)



for i in range(iter):

    z = x_train.dot(theta)

    Y_pred = sigmoid(z)

    cost[i] = (-1 / n) * (y_train.T.dot(np.log(Y_pred)) + (1 - y_train).T.dot(np.log(1 - Y_pred)) + (l2_param * (np.sum(theta**2))))  

    theta = theta - (alpha / n) * (x_train.T.dot(Y_pred - y_train)) 

    h_theta = x_test.dot(theta)

    h_theta[h_theta < 0.5] = 0

    h_theta[h_theta >= 0.5] = 1

    accuracy[i] = accuracy_score(y_test,h_theta)

    





print("Model accuracy is:" , accuracy_score(y_test,h_theta) * 100)



plt.plot(np.arange(0,iter),cost,'r-',label = 'Cost')

plt.plot(np.arange(0,iter),accuracy,'b-',label = 'Accuracy')

plt.xlabel("No of iterations")

plt.ylabel("Percentage")

plt.title("Number of iterations v/s Cost function and Accuracy")

plt.legend()

plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()



nsamples, nx, ny = x_train.shape

x_train = x_train.reshape((nsamples,nx*ny))



nsamples, nx, ny = x_test.shape

x_test = x_test.reshape((nsamples,nx*ny))
logreg_with_L1 = LogisticRegression(penalty='l1', multi_class='ovr', solver='liblinear')

logreg_with_L1.fit(x_train, y_train)

print('Training accuracy: ',logreg_with_L1.score(x_train, y_train) * 100)

print('Testing accuracy: ',logreg_with_L1.score(x_test, y_test) * 100)
(x_train, y_train), (x_test, y_test) = mnist.load_data()



nsamples, nx, ny = x_train.shape

x_train = x_train.reshape((nsamples,nx*ny))



nsamples, nx, ny = x_test.shape

x_test = x_test.reshape((nsamples,nx*ny))
logreg_with_L2 = LogisticRegression(penalty='l2', multi_class='ovr', solver='liblinear',max_iter = 1000)

logreg_with_L2.fit(x_test, y_test)

print('Training accuracy: ',logreg_with_L2.score(x_train, y_train) * 100)

print('Testing accuracy: ',logreg_with_L2.score(x_test, y_test) * 100)
probabs = logreg_with_L2.predict_proba(x_train)

classes = range(10)

y_test = label_binarize(y_train, classes)

for i in range(10):

    preds = probabs[:,i]    

    fpr, tpr, threshold = roc_curve(y_test[:, i], preds)

    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

plt.show()