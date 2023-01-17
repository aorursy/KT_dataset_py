# IMPORT MODULES

import numpy as np
from numpy import ma
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
%matplotlib inline
from matplotlib import ticker, cm
from matplotlib.pyplot import figure
import seaborn as sns

from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Modules imported...YAY !")
# LOAD DATA

dfRaw = pd.read_csv('../input/creditcard.csv')
print(dfRaw.shape)
print(dfRaw.columns)
data = dfRaw.copy()
normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

print(type(data))
print("data ", data.shape)
print("normal_data ", normal_data.shape)
print("fraud_data ", fraud_data.shape)
print("Percent fraud ", round(100*492/284807, 4),"%")
print("_"*100)
print(data.head())
# Features' Prob DISTRIBUTION

plt.figure()
matplotlib.style.use('ggplot')
pca_columns = list(data)[:-1]
normal_data[pca_columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(16,2))
# PLOT AMOUNT - Norm vs Fraud

normal_data["Amount"].loc[normal_data["Amount"] < 500].hist(bins=100);
plt.figure()
fraud_data["Amount"].loc[fraud_data["Amount"] < 500].hist(bins=100);
plt.figure()
print("Mean", normal_data["Amount"].mean(), fraud_data["Amount"].mean())
print("Median", normal_data["Amount"].median(), fraud_data["Amount"].median())
# PLOT TIME - Norm vs Fraud

normal_data["Time"].hist(bins=100);
plt.figure()
fraud_data["Time"].hist(bins=100);
plt.figure()
# data.plot.scatter("Time","Amount", c="Class")
data.plot.scatter("V1","V2", c="Class")
data.plot.scatter("V2","V3", c="Class")
data.plot.scatter("V1","V3", c="Class")

#  SCALER

data = dfRaw.copy()

print(data.shape)
scl = StandardScaler()
all_cols = list(data)[:] 
pca_columns = list(data)[:-1] 
Xcopy = data[pca_columns]
XcopyALL = data[all_cols]
Xscaled = scl.fit_transform(Xcopy)
OnlyClass = data['Class'].values.reshape(-1,1)
data = np.concatenate((Xscaled, OnlyClass), axis=1)
data = pd.DataFrame(data, columns = XcopyALL.columns)

normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

print(data.shape)
#print(data.head)
print("_"*100)
print("data ", data.shape)
print("normal_data ", normal_data.shape)
print("fraud_data ", fraud_data.shape)
print("Percent fraud ", round(100*492/284807, 4),"%")
# Features' Prob DISTRIBUTION AFTER Scaler

plt.figure()
matplotlib.style.use('ggplot')
pca_columns = list(data)[:]
data[pca_columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(16,2))
print("data['Time'].mean()  ", data['Time'].mean())
print("data['Amount'].mean()  ", data['Amount'].mean())
# CREATE the TRAIN, VALIDATION and TEST sets
# Fraud data is ONLY in the CV and TEST - not in TRAIN

normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

print("normal_data ", normal_data.shape)
print("fraud_data ", fraud_data.shape)

pca_columns = list(data)[:-1] 
normal_pca_data = normal_data[pca_columns]
fraud_pca_data = fraud_data[pca_columns]

num_test = 75000
shuffled_data = normal_pca_data.sample(frac=1)[:-num_test].values
X_train = shuffled_data

X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])
y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])

X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[246:]])
y_test = np.concatenate([np.zeros(num_test), np.ones(246)])


print("normal_pca_data ", normal_pca_data.shape)
print("fraud_pca_data", fraud_pca_data.shape)
print("Fraud data divided between valid and test with NONE in the training")
print("X_train ", X_train.shape)
print("X_valid ", X_valid.shape)
print("y_valid ", y_valid.shape)
print("X_test ", X_test.shape)
print("y_test ", y_test.shape)

# Get Epsilon as the max prob 

p = multivariate_normal(mean=np.mean(X_train,axis=0), cov=np.cov(X_train.T))

x = p.pdf(fraud_pca_data)
print("max prob of x on fraud_pca_data", max(x))
x = p.pdf(X_valid)
print("max prob of x on X_valid", max(x))

epsilons = [1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20]
eps = epsilons[-1]

pred = (x <= eps)
f = f1_score(y_valid, pred,average='binary')
print("F1 score on y_valid", round(f,4), " with epsilon ", eps)
# CONFUSION MATRIX and F1 SCORE

x = p.pdf(X_valid)
print("max prob of x on X_valid", max(x))

eps = 5e-15

print("epsilon ", eps)
print("_"*50)
pred = (x<eps)
CM = confusion_matrix(y_valid,pred)
tn, fp, fn, tp = confusion_matrix(y_valid,pred).ravel()

print(CM)
print("_"*50)
print("TP ", tp)
print("FP ", fp)
print("TN ", tn)
print("FN ", fn)
print("_"*50)

# F1 Score
#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))
precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_valid,pred, average='binary')
print("precision ", round((precision), 4))
print("recall ", round((recall), 4))
print("F1 score ", round((fbeta_score), 4))

# Find the best EPSILON in terms of Recall, Precision and F1 Score

validation = []
for thresh in np.linspace(eps*1e-150, eps*1e2, 100):
    pred = (x<= thresh)
    prec, recall, F1, support = precision_recall_fscore_support(y_valid, pred, average='binary')
    validation.append([thresh, recall, prec, F1])
    
x = np.array(validation)[:, 0]
y1 = np.array(validation)[:, 1]
y2 = np.array(validation)[:, 2]
y3 = np.array(validation)[:, 3]

# 3 CHARTS - Recall, Precision and F1 score

plt.plot(x, y1)
plt.title("Recall")
plt.xscale('log')
plt.show()
plt.plot(x, y2)
plt.title("Precision")
plt.xscale('log')
plt.show()
plt.plot(x, y3)
plt.title("F1 score")
plt.xscale('log')
plt.show()
# Recall, Precision and F1 Score on same chart
df=pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3 })
 
# multiple line plot
plt.xscale('log')
plt.plot( 'x', 'y1', data=df, marker='', color='green', linewidth=2,  label="Recall")
plt.plot( 'x', 'y2', data=df, marker='', color='blue', linewidth=2, label="Precision")
plt.plot( 'x', 'y3', data=df, marker='o', color='red', markerfacecolor='orange',linewidth=3, markersize=6, label="F1 Score")
plt.legend()

#  SCALER & PCA ... instead of creating a pipeline...

data = dfRaw.copy()

# Scaler
print(data.shape)
scl = StandardScaler()
all_cols = list(data)[:] 
pca_columns = list(data)[:-1] 
Xcopy = data[pca_columns]
XcopyALL = data[all_cols]
Xscaled = scl.fit_transform(Xcopy)
OnlyClass = data['Class'].values.reshape(-1,1)
data = np.concatenate((Xscaled, OnlyClass), axis=1)
data = pd.DataFrame(data, columns = XcopyALL.columns)

normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

print(data.shape)
print("data ", data.shape)
print("normal_data ", normal_data.shape)
print("fraud_data ", fraud_data.shape)
print("Percent fraud ", round(100*492/284807, 4),"%")
print("_"*100)

# PCA
print("AFTER PCA")

pca = PCA(n_components = 2) 

all_cols = list(data)[:] 
pca_columns = list(data)[:-1] 
Xcopy = data[pca_columns]
XcopyALL = data[all_cols]
dataPostPCA = pca.fit_transform(Xcopy)
OnlyClass = data['Class'].values.reshape(-1,1)
data = np.concatenate((dataPostPCA, OnlyClass), axis=1)
data = pd.DataFrame(data, columns = [0,1,'Class'])

normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

print(data.shape)
print("data ", data.shape)
print("normal_data ", normal_data.shape)
print("fraud_data ", fraud_data.shape)
print("Percent fraud ", round(100*492/284807, 4),"%")
#print("_"*100)
#print(data.head)
# Get the Multi Variate Gaussian Prob Distribution Function for the 2 dimensions post PCA

normal_data=normal_data.copy()
fraud_data= fraud_data.copy()
normal_data = normal_data.drop('Class', axis=1)
fraud_data = fraud_data.drop('Class', axis=1)
print(normal_data.columns)
print(fraud_data.columns)

p = multivariate_normal(mean=np.mean(normal_data,axis=0), cov=np.cov(normal_data.T))

# View the FRAUD on a 2 dims (Post PCA) Guassian distribution of the normal data
# Reducing from 30 dims to 2 - helps with the visualization but surely doesn't help with separating the Fraud from the Normal

x, y = np.mgrid[-5.0:12.0:.01, -5.0:5.0:.01] 
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(mean=np.mean(normal_data,axis=0), cov=np.cov(normal_data.T)) # mean and covariance matrix for 2 dims dataset

fig, ax = plt.subplots()
cs = ax.contourf(x, y, rv.pdf(pos))
cbar = fig.colorbar(cs)
plt.scatter(fraud_data[0],fraud_data[1], edgecolor="r") # Location on chart of the anomaly points
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
#plt.show()
# View some NORMAL on a 2 dims (Post PCA) Guassian distribution of the normal data
# Reducing from 30 dims to 2 - helps with the visualization but surely doesn't help with separating the Fraud from the Normal

SampleNormal = normal_data[-500:]

x, y = np.mgrid[-5.0:12.0:.01, -5.0:5.0:.01] 
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(mean=np.mean(normal_data,axis=0), cov=np.cov(normal_data.T)) # mean and covariance matrix for 2 dims dataset

fig, ax = plt.subplots()
cs = ax.contourf(x, y, rv.pdf(pos))
cbar = fig.colorbar(cs)
plt.scatter(SampleNormal[0],SampleNormal[1], edgecolor="b") # Location on chart of the anomaly points
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
#plt.show()