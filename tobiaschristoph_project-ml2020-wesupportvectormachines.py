# Quick load dataset and check
import pandas as pd
import os
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
os.listdir('data')
if ~running_local:
    path = "data/final-project-dataset/"
else:
    path = "./"

filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")
data_train.describe()
data_test.describe()

from sklearn.tree import DecisionTreeClassifier

## Select target and features
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]

clf = DecisionTreeClassifier()
clf = clf.fit(data_X,data_Y)
y_pred = clf.predict(data_X)
sum(y_pred==data_Y)/len(data_Y)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)
sum(y_pred==y_val)/len(y_val)
def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
sum(y_pospred==y_pos)/len(y_pos)
#print(x_val.shape)
#print(X_pos.shape)
#print(y_val.shape)
#print(y_pos)
#print(y_pospred)
X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
#print(sum(y_negpred==y_neg))
print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))
## Your work
from sklearn.model_selection import train_test_split
# our data to train
data_train.head()

#X_train = x_train
#y_train = y_train
#X_test = x_val
#y_test = y_val

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:2000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:2000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:600]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:600]
print('y_test_2 shape: ', y_test_2.shape)



print(X_test_2.shape)
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
# creating a svm classifier
clf = svm.SVC(kernel='linear', C=1)
# training said classifies
clf.fit(X_train_2, y_train_2)
#clf.score(X_test_2, y_test_2)

# predict response
y_pred_SVM = clf.predict(X_test_2)


from sklearn import metrics
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)

print('our predicted values: ' ,y_pred_SVM)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
all_hists = X_train.hist(bins=20, figsize=(50,25))
data_X[data_X == -1].count()

data_X = data_X.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_X[data_X == -1].count()
all_hists = X_train.hist(bins=20, figsize=(50,25))
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:2000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:2000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:600]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:600]
print('y_test_2 shape: ', y_test_2.shape)
from sklearn import metrics
clf_no_neg = svm.SVC(kernel='linear', C=1)
# training said classifies
clf_no_neg.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM_no_neg = clf_no_neg.predict(X_test_2)
# checking how accurately the prediction was
accuracy_no_neg = metrics.accuracy_score(y_test_2, y_pred_SVM_no_neg)
print('Accuracy of :',accuracy_no_neg)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' ,sum(y_pred_SVM_no_neg))
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=y_test_2, y_pred=y_pred_SVM_no_neg)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
from sklearn import metrics
clf = svm.SVC(kernel='linear', class_weight='balanced', C=1.0)
# training said classifies
clf.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_pred_SVM))
clf = svm.SVC(kernel='rbf', class_weight='balanced', C=1.0)
# training said classifies
clf.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_pred_SVM))
conf_mat = confusion_matrix(y_true=y_test_2, y_pred=y_pred_SVM)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
!pip install imblearn
import imblearn
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_2)

plot_2d_space(X_train_pca, y_train_2, 'Imbalanced dataset (2 PCA components)')
# import the SMOTETomek
from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE(sampling_strategy='minority')

# fit the object to our training data
x_train_smote, y_train_smote = smote.fit_sample(X_train_2, y_train_2)

clf = svm.SVC(kernel='linear', C=1.0)
# training said classifies
clf.fit(x_train_smote, y_train_smote)
#prediction
y_predict_smote = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_predict_smote)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_predict_smote))
print(y_predict_smote)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:15000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:15000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:3500]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:3500]
print('y_test_2 shape: ', y_test_2.shape)
smote = SMOTE(sampling_strategy='minority')
x_train_smote, y_train_smote = smote.fit_sample(X_train_2, y_train_2)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x_train_smote, y_train_smote)
#prediction
y_predict_smote = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_predict_smote)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_predict_smote))
data_test.shape
data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_test.shape
data_test.describe()
data_train.shape
data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_train.shape
data_test_2 = data_test.drop(columns=['id'])
data_train_X = data_train.drop(columns=['id'])
print('train shape:', data_train_X.shape)
print('test shape:', data_test_2.shape)
#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]
# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:100000]
print('X_train_2 shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:100000]
print('y_train_2 shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_train_smote_END, y_train_smote_END = smote.fit_sample(X_train_END, y_train_END)
X_train_smote_END.shape
clf = svm.SVC(kernel='linear', C=1.0)
#fitting
clf.fit(X_train_smote_END, y_train_smote_END)
# now predict the data_test values
y_predict_smote = clf.predict(data_test_2)
y_predict_smote.shape
four_percent_of_all = 144880 * 0.04
print('4 % of all points would be:', four_percent_of_all)
print('# of our predicted values: ' , sum(y_predict_smote))
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_predict_smote, True) 
data_out.to_csv("/Users/hercules/ml/data/final-project-dataset/submission.csv",index=False)
from io import StringIO
output = StringIO()
data_out.to_csv(output)
output.seek(0)
print(output.read())
filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")

data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_train_X = data_train.drop(columns=['id'])
data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_test_2 = data_test.drop(columns=['id'])

#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]

# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:10000]
print('X_train_END shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:10000]
print('y_train_END shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_train_smote_END, y_train_smote_END = smote.fit_sample(X_train_END, y_train_END)
print('X_train shape: ', X_train_smote_END.shape)
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train_smote_END, y_train_smote_END)
test_10k = data_test_2[0:10000]
predictions = rf.predict(test_10k)
print(10000*0.04)
print('# of our predicted values: ' , sum(predictions))
print('without smote sampleing it was at 478')
predict_all = rf.predict(data_test_2)
print(148000*0.04)
print('# of our predicted values: ' , sum(predict_all))
print('without smote sampleing it was at 7169')

filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")

data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_train_X = data_train.drop(columns=['id'])
data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_test_2 = data_test.drop(columns=['id'])

#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]

# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:30000]
print('X_train_END shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:30000]
print('y_train_END shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train_END, y_train_END)
# create prediction of our test set:
predict_all = rf.predict(data_test_2)
print(148000*0.04)
print('# of our predicted values: ' , sum(predict_all))
cnt = 0
for i in range(predict_all.size):
    if predict_all[i] > 0.5:
        predict_all[i] = 1
        cnt = cnt + 1
    else:
        predict_all[i] = 0
cnt
data_out.describe()

