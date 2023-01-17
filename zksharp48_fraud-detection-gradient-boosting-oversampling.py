import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")

df.head()
from sklearn.preprocessing import StandardScaler

df_scaled = df #Make Duplicate



df_scaled['normAmount'] =  StandardScaler().fit_transform(df_scaled['Amount'].values.reshape(-1, 1))

df_scaled = df_scaled.drop(['Amount'],axis=1)

df_scaled.head()
# Number of data points in the minority class && Indices Picking

number_records_fraud = len(df_scaled[df_scaled.Class == 1]) #492 Fraud Cases

fraud_indices = np.array(df_scaled[df_scaled.Class == 1].index)

print("Number of Fraud Cases: ", number_records_fraud)





# Number of data points in the majority class && Indices Picking

number_records_normal = len(df_scaled[df_scaled.Class != 1])

normal_indices = df_scaled[df_scaled.Class == 0].index

print("Number of Normal Cases: ", number_records_normal)



# Get fraud Transactions by Filtering

df_fraud = df_scaled.iloc[fraud_indices] 

X_fraud = df_fraud.ix[:,df_fraud.columns != 'Class']

y_fraud = df_fraud.ix[:,df_fraud.columns == 'Class']



# Get Normal Transactions by Filtering

df_normal = df_scaled.iloc[normal_indices] #Get normal Transaction by Filtering

X_normal = df_normal.ix[:,df_normal.columns != 'Class']

y_normal = df_normal.ix[:,df_normal.columns == 'Class']



# Make X,y for classfication

X = df_scaled.ix[:, df_scaled.columns != 'Class']

yy = df_scaled.ix[:, df_scaled.columns == 'Class']

y = np.asarray(yy['Class'])
from imblearn.over_sampling import SMOTE

# Apply SMOTE's

kind = 'regular'

sm = SMOTE(kind='regular')

X_res, y_res = sm.fit_sample(X, y)



print("esampled Dataset has shape: ", X_res.shape)

print("Number of Fraud Cases (Real && Synthetic): ", np.sum(y_res))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y)



print("Number transactions train dataset: ", len(X_train))

print("Number transactions test dataset: ", len(X_test))

print("Total number of transactions: ", len(X_train)+len(X_test))







X_train_res, X_test_res, y_train_res, y_test_res= train_test_split(X_res, y_res)



print("")

print("Number transactions train dataset: ", len(X_train_res))

print("Number transactions test dataset: ", len(X_test_res))

print("Total number of transactions: ", len(X_train_res)+len(X_test_res))
from sklearn.ensemble import GradientBoostingClassifier

est = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=1,

                                random_state=0, verbose = 1)

est.fit(X_train_res, y_train_res)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')