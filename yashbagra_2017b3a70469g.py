# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/minor-project-2020/train.csv", header=0)
df
df[(df['target']==1)]
df.info()
df.describe()
fig, axs = plt.subplots(ncols=10, nrows=9, figsize=(10, 100))

index = 0

axs = axs.flatten()

for k,v in df.items():

    sns.distplot(v, ax=axs[index],  kde_kws = {'bw' : 1})

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Compute the correlation matrix

corr = df.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(100, 100))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
from sklearn.model_selection import train_test_split



y = df["target"]

X = df.drop(["target","id"], axis=1)





X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state = 121,stratify=y)
from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler

scaled_X_train = scaler.fit_transform(X_train)     #Fit and transform the data



scaled_X_train
from scipy import optimize
def sigmoid(z):

    z = np.array(z)

    g = np.zeros(z.shape)

    g = 1 / (1 + np.exp(-z))

    return g
m, n = scaled_X_train.shape

print (scaled_X_train.shape)

X_train=scaled_X_train
X_train = np.concatenate([np.ones((m, 1)), X_train], axis=1)
X_train.shape
def costFunctionReg(theta, X, y, lambda_):

    

    # Initialize some useful values

    m = y.size  # number of training examples

    J = 0

    grad = np.zeros(theta.shape)

    h = sigmoid(X.dot(theta.T))

    

    temp = theta

    temp[0] = 0

    

    J = (1/m) * np.sum(-y.dot(np.log(h))-(1-y).dot(np.log(1-h))) + (lambda_/(2*m))*np.sum(np.square(temp))

    

    grad = (1 / m) * (h - y).dot(X) 

    grad = grad + (lambda_ / m) * temp

    

    return J, grad
# Initialize fitting parameters

initial_theta = np.zeros(X_train.shape[1])



# Set regularization parameter lambda to 1 (you should vary this)

lambda_ = 1



res = optimize.minimize(costFunctionReg,

                        initial_theta,

                        (X_train, y_train, lambda_),

                        jac=True,

                        method='BFGS', 

                        options={"maxiter":10000, "disp":True})



# the fun property of OptimizeResult object returns

# the value of costFunction at optimized theta

cost = res.fun



# the optimized theta is in the x property of the result

theta = res.x



print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))



print('theta:')

print(theta)
print (theta.shape)
def predict(theta, X):

    

    m = X.shape[0] # Number of training examples





    p = np.round(sigmoid(X.dot(theta.T)))



    return p
m=X_val.shape[0]

print(X_val)

X_val = np.concatenate([np.ones((m, 1)), X_val], axis=1)

p = predict(theta, X_val)

print('Train Accuracy: {:.2f} %'.format(np.mean(p == y_val) * 100))
from sklearn.metrics import roc_curve, auc



FPR, TPR, _ = roc_curve(y_val, p)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)

df=pd.read_csv("/kaggle/input/minor-project-2020/test.csv", header=0)
df.head()
df=df.drop(["id"],axis=1)
X=df.to_numpy()
print(X.shape)
X = np.concatenate([np.ones((200000, 1)), X], axis=1)
print(X)
p = predict(theta, X)
print(p.shape)
df=pd.read_csv("/kaggle/input/minor-project-2020/test.csv", header=0)

df.head()
my_submission = pd.DataFrame({'id': df.id, 'target': p})

my_submission.to_csv('submission.csv', index=False)