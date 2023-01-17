#Importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Reading dataset
data = pd.read_csv('../input/logistic-regression/Social_Network_Ads.csv')
data.head()
#Checking for missing data
data.info()
#Properties of data
data.describe()
#Using label encoder to encode the categorical feature - Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data.head()
#Plotting heatmap of correlation matrix of features
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='inferno',mask=np.triu(data.corr(),k=1))
#Replacing space in column headers with '_'
data.columns = data.columns.str.replace(' ', '_')
data.info()
#Dropping user_id and gender
data.drop(labels = ['User_ID','Gender'], axis = 1, inplace = True)
data.info()
#checking for duplicates
sum(data.duplicated())
#dropping ALL duplicate values
data.drop_duplicates(keep = False, inplace = True)
#Plotting boxplot of features to find outliers
plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)
sns.boxplot(data['Age'],color='yellow')
plt.subplot(3,3,2)
sns.boxplot(data['EstimatedSalary'], color='yellow')

plt.show()
#Viewing class distribution
plt.figure(figsize=(6, 4))
sns.countplot('Purchased', data=data)
plt.title('Class Distributions')
#Getting X and y
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#reshaping
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#adding bias variable
X_new = np.concatenate((np.ones((len(X_train), 1), dtype = int), X_train), axis = 1)
print(X_new.shape)
#Sigmoid activation function
def sigmoid_function(X, theta):
    """
    Arguments: 
      X - Feature matrix X also containing the bias variable
      theta - parameters being optimized

    Returns:
      h - hypothesis function 1 / (1 + e ^-(X * theta))
    """

    h = (1 / (1 + np.exp(-np.matmul(X, theta))))
    return h
#Computing cost
def cost_function(m, h, y):
    """
    Arguments: 
      m - size of training set
      h - hypothesis function (X * theta)
      y - dependent variable

    Returns:
      J - Computed cost    
    """
    
    J = (np.sum((-y * np.log(h)) - ((1 - y) * (np.log(1 - h)))) / m)
    return J
#gradient descent
def grad_func(m, X, y, theta, alpha, h):
    """
    Arguments: 
      m - size of training set
      X - Feature matrix X also containing the bias variable
      y - dependent variable
      theta - parameters being optimized
      alpha - learning rate 
      h - hypothesis function (X * theta)

    Returns:
      theta - Optimized parameters    
    """

    theta = theta - ((alpha/m) * (np.matmul((h - y).T, X))).T
    return theta
#Logistic Regression from scratch
m = len(X_new)
alpha = 0.01403
cost = []
epochs = 2000

#theta initialization
theta = np.zeros((X_new.shape[1], 1),dtype=float)

for i in range(epochs):
  h = sigmoid_function(X_new, theta)
  J = cost_function(m, h, y_train)
  cost.append(J)
  theta = grad_func(m, X_new, y_train, theta, alpha, h)

#Plotting cost function vs epochs
plt.plot(list(range(epochs)), cost, '-r')
plt.title("Cost function vs epochs")
plt.xlabel("epochs")
plt.ylabel("J - cost function")
plt.show()
#Preparing test data

#adding bias variable
Xt_new = np.concatenate((np.ones((len(X_test), 1), dtype=int), X_test), axis=1)
print(Xt_new.shape, y_test.shape)
#Predicting results
y_pred = sigmoid_function(Xt_new, theta)
print(y_pred.shape)

y_pred_new = []
for x in y_pred:
    y_pred_new.append(1 if(x > 0.5) else 0)

y_pred_new = np.array(y_pred_new)
y_pred_new = y_pred_new.T
y_pred_new = y_pred_new.reshape(len(y_pred_new), 1)

print(y_pred_new.shape)
#Confusion matrix visualized:
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred_new)

sns.heatmap(cf_matrix, annot=True)
#Metrics based result
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_new))