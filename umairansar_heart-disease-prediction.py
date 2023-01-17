# Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
%matplotlib inline

from collections import Counter
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
rpt = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
rpt[20:41]
# Checkig datatypes of each column and shape
print(rpt.dtypes, '\n')
print("Shape: ", rpt.shape)

# Checking for NaN and correcting
print("NaN exists: ", rpt.isnull().values.any())
print("NaN count: ", rpt.isnull().values.sum())
print("NaN count for each attriue:")
print(rpt.isnull().sum())
print("\n")

# Dropping all rows with any NaN
rpt.dropna(axis = 0, inplace = True)
print("New Shape after dropping rows: ", rpt.shape)
print("\n")
# Depictong Correlations
rpt.corr()
# Creating X and y
X = np.asarray(rpt[['male', 'age', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']])
y = np.asarray(rpt['TenYearCHD']) #do not do [['TenYearCHD']] it will give shape (2924,1) instead of needed (2924,)

# Normalizing X
X = preprocessing.StandardScaler().fit_transform(X)
print("Shape X:", X.shape)
print("Shape y:", y.shape)
def visualizeIn2D(X, y):
    '''
    Visualizing Data in 2 dim using features: sysBP, diaBP. 
    '''
    # Class labels in a dict
    counter = Counter(y)
    print(counter)
    
    plt.figure(figsize = (15, 10))
    for label, _ in counter.items():
        row = np.where(y == label)[0]
        plt.scatter(X[row, 0], X[row, 1], label=str(label))
    plt.legend()
    plt.show()
# Calculating number of classes
NumOfClasses = len(rpt.groupby('TenYearCHD').size().values)

#Creating new X1 because X is already ndarray and normalized
X1 = np.asarray(rpt[['sysBP', 'diaBP']])

# Visualize
visualizeIn2D(X1, y)
def Ktimes_train_LR(LR, kf, X, y):
    '''
    Since we are not using train_test_split where say 80% was used for training and 20% was used for testing,
    in KFold our training is done on k-1 sets and testing is done on 1 set. So if n_sets equals 5, then 4 sets
    (or 80%) are used for training and 1 set (or 20%) is used for testing.
    '''
    scores = list()
    
    for train_index, test_index in kf.split(X):
        
        # Splitting Data into train test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train
        LR.fit(X_train, y_train)
        
        # Predicting
        y_hat = LR.predict(X_test)
        y_hat_prob = LR.predict_proba(X_test)
        
        # Scoring
        scores.append(accuracy_score(y_test, y_hat))
        
    '''
    The y_hat and y_hat_proba returned below hold the values from the "LAST" iteration for KFold training.
    '''
    report = (X_train, X_test, y_train, y_test, y_hat, y_hat_prob, scores)
    return report


# Init the main Logistic model - the lesser the C, the greater the regularization 
LR = LogisticRegression(C = 0.0000001, solver = 'liblinear')

# Init KF
kf = KFold(n_splits = 5, random_state = 7)

# Call for K times training and predicting
(X_train, X_test, y_train, y_test, y_hat, y_hat_prob, scores) = Ktimes_train_LR(LR, kf, X, y)

# Priting the jaccard_scores
print("Scores: ", scores)
'''
The above block of code and these 3 lines of code below do exactly the same work, except that the we do not
have access to (or do not need) X_train, X_test, y_train, y_test, y_hat, y_hat_prob. We just get back scores.
Of course, we can always split the data using other functions like ShuffleSplit or train_test_split to get
access of X_train, X_test and so on.
'''
kfold = KFold(n_splits = 5, random_state = 7)
cv_result = cross_val_score(LR, X, y, cv = kfold, scoring = 'accuracy')
print(cv_result)
# Plotting the confusion matrix
confusion_matrix(y_test, y_hat, labels = [1, 0])
plot_confusion_matrix(LR, X_test, y_test, labels = [1, 0], cmap=plt.cm.Blues) #test acc
plot_confusion_matrix(LR, X_train, y_train, labels = [1, 0], cmap=plt.cm.Blues) #train acc
# Classification Report
print("\t\t\t *TEST REPORT*")
print(classification_report(y_test, y_hat)) #test acc
print('\n')
print("\t\t\t *TRAIN REPORT*")
print(classification_report(y_train, LR.predict(X_train))) #train acc

# Log-loss
print("LogLoss: ", log_loss(y_hat, y_hat_prob))
# Checking Label Imballance
print("Label Imballance:")
print(rpt.groupby('TenYearCHD').size())
print("\n")

# Checking Gender Discrepancy 
print("Gender Distribution:")
print(rpt.groupby('male').size())
# Import Additional Libraries
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# Defining Pipeline
over = SMOTE(sampling_strategy = 0.4, k_neighbors = 3, random_state = 7)
under = RandomUnderSampler(sampling_strategy = .55)
steps = [('o', over), ('u', under), ('model', LR)]
pipe1 = Pipeline(steps = steps)

# Training and Evaluating
cv_result = cross_val_score(pipe1, X, y, cv = kfold, scoring = 'accuracy')
print("Mean Accuracy: ", np.mean(cv_result))
print('\n')

pipe = Pipeline(steps =[('o', over), ('u', under)])
X2, y1 = pipe.fit_resample(X1, y)
counter = Counter(y1)
print(counter)

# Call for K times training and predicting
(X_train, X_test, y_train, y_test, y_hat, y_hat_prob, scores) = Ktimes_train_LR(pipe1, kf, X, y)
plot_confusion_matrix(pipe1, X_test, y_test, labels = [1, 0], cmap=plt.cm.Blues) #test acc
visualizeIn2D(X2, y1)
print(classification_report(y_test, y_hat)) #test acc
from sklearn.preprocessing import binarize

cm2 = 0
y_hat2 = binarize(y_hat_prob,0.4999856)[:,1]
cm2 = confusion_matrix(y_test,y_hat2)
print ('With',0.4999856,'threshold the Confusion Matrix is ','\n',cm2,)
print(classification_report(y_test, y_hat2)) 
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_test, y_hat_prob[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Heart Disease Predictor (ROC curve)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid(True)
print("Area under curve: ", roc_auc_score(y_test,y_hat_prob[:,1]))