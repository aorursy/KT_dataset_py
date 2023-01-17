import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics     
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
import itertools
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/oranges-vs-grapefruit/citrus.csv') ## Modified input data from original
le = preprocessing.LabelEncoder()
df['name'] = le.fit_transform(df['name']) ## Modified name column contains only object values.
df.head()
df = df.select_dtypes(exclude=['object'])
# Dividing the data into quantiles and doing the outlier analysis.

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
# Heatmap showing the correlation of various columns with each other.

ax = plt.figure(figsize = (20,10))
ax = sns.heatmap(df.corr(),cmap = 'gist_heat')
# The features with skewed or non-normal distribution.
skew_df = df[['diameter','weight','red','green','blue']] #Introduce our parameters
fig , ax = plt.subplots(2,2,figsize = (20,10))
col = skew_df.columns
for i in range(2):
    for j in range(2):
        ax[i][j].hist(skew_df[col[2 * i + j]] , color = 'k')
        ax[i][j].set_title(str(col[2 * i + j]))
        ax[i][j].set_axis_bgcolor((1, 0, 0))
#We changed Attribution to name.
target = df['name']
train = df.drop('name',axis = 1)
train.shape
pd.value_counts(target).plot(kind = 'bar',cmap = 'BrBG')
plt.rcParams['axes.facecolor'] = 'blue'
plt.title("Count of classes")
train_accuracy = []
test_accuracy = []
models = ['Perceptron' , 'Multi layer perceptron'] #We applied the Perceptron model.
#Defining a function which will give us train and test accuracy for each classifier.
def train_test_error(y_train,y_test):
    train_error = ((y_train==Y_train).sum())/len(y_train)*100
    test_error = ((y_test==Y_test).sum())/len(Y_test)*100
    train_accuracy.append(train_error)
    test_accuracy.append(test_error)
    print('{}'.format(train_error) + " is the train accuracy")
    print('{}'.format(test_error) + " is the test accuracy")
X_train, X_test, Y_train, Y_test = train_test_split(
    train, target, test_size=0.33, random_state=42)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.linear_model import Perceptron
per = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_train,Y_train)
train_predict = per.predict(X_train)
test_predict = per.predict(X_test)
train_test_error(train_predict , test_predict)
class_names=['0' , '1']
confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')
probs = per.predict(X_test)
#preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train,Y_train)
train_predict = mlp.predict(X_train)
test_predict = mlp.predict(X_test)
train_test_error(train_predict , test_predict)
confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')
probs = mlp.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))
roc_score = np.asarray([0.904524571998,0.5]) #Adjusted values 
results = DataFrame({"Roc score" : roc_score, "Test Accuracy" : test_accuracy , "Train Accuracy" : train_accuracy} , index = models)
results