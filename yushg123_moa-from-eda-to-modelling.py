

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.express as px

train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
print("There are {} features and {} rows in the train data.".format(train.shape[1], train.shape[0]))

print("There are {} features and {} rows in the test data.".format(test.shape[1], test.shape[0]))
train.head()
print(train.info())

print()

print(test.info())
train.describe()
test.describe()
a = train['cp_type'].value_counts().reset_index()

fig = px.pie(a, values='cp_type', names='index', title='CP Type')

fig.show()



a = train['cp_dose'].value_counts().reset_index()

fig = px.pie(a, values='cp_dose', names='index', title='CP Dose')

fig.show()

a = test['cp_type'].value_counts().reset_index()

a['Percentage'] = a['cp_type'] / len(test)





b = test['cp_dose'].value_counts().reset_index()

b['Percentage'] = b['cp_dose'] / len(test)



a
b
a = train.groupby('cp_time').count()





fig = px.bar(a, x=['24', '48', '72'], y='sig_id')

fig.show()



a = test.groupby('cp_time').count()





fig = px.bar(a, x=['24', '48', '72'], y='sig_id')

fig.show()
a = train.groupby('cp_dose').max()

a
import plotly.figure_factory as ff



# Add histogram data

x1 = train['g-0']

x2 = train['g-10']

x3 = test['g-0']

x4 = test['g-10']



# Group data together

hist_data = [x1, x2, x3, x4]



group_labels = ['Train G-0', 'Train G-10', 'Test G-0', 'Test G-10']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()


# Add histogram data

x1 = train['c-0']

x2 = train['c-10']

x3 = test['c-0']

x4 = test['c-10']



# Group data together

hist_data = [x1, x2, x3, x4]



group_labels = ['Train C-0', 'Train C-10', 'Test C-0', 'Test C-10']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()
import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

import cuml, cudf

from cuml.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline



colsToDrop = ['sig_id', 'cp_time', 'cp_type', 'cp_dose']

tsneData = train.drop(colsToDrop, axis=1)





tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(tsneData)



plt.scatter(train_2D[:,0], train_2D[:,1], s = 0.5)

plt.title("Compressed Training Data")

plt.show()


colsToDrop = ['sig_id', 'cp_time', 'cp_type', 'cp_dose']

tsneData = test.drop(colsToDrop, axis=1)





tsne = TSNE(n_components=2)

test_2D = tsne.fit_transform(tsneData)



plt.scatter(test_2D[:,0], test_2D[:,1], s = 0.5)

plt.title("Compressed Test Data")

plt.show()
train_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
print("Target data has {} prediction variables".format(train_target.shape[1] - 1))
train_target.head()
train_target.describe()
a = train_target['acat_inhibitor'].value_counts().reset_index()

fig = px.pie(a, values='acat_inhibitor', names='index', title='acat_inhibitor')

fig.show()
train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



objCols = ['cp_type', 'cp_time', 'cp_dose']

for col in objCols:

    le.fit(train[col])

    train[col] = le.transform(train[col])

    test[col] = le.transform(test[col])

    

train.head()
from sklearn.ensemble import ExtraTreesClassifier



rf = ExtraTreesClassifier(n_estimators=100, criterion="entropy", max_depth = 15, max_features = "sqrt", random_state = 1, bootstrap=True, max_samples = 1000, n_jobs=-1)
rf.fit(train.drop(['sig_id'], axis=1), train_target.drop(['sig_id'], axis=1))
preds = rf.predict(test.drop(['sig_id'], axis=1))

preds.shape
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



for i in range(len(sub.columns)):

    if i != 0:

        col = sub.columns[i]

        sub[col] = preds[:, i - 1]

sub
sub.to_csv('submissionExtraTrees.csv', index=False)
from cuml.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(train.drop(['sig_id'], axis=1), train_target.drop(['sig_id'], axis=1))
preds = knn.predict(test.drop(['sig_id'], axis=1))
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



for i in range(len(sub.columns)):

    if i != 0:

        col = sub.columns[i]

        sub[col] = preds[:, i - 1]

sub
sub.to_csv('submissionKNN.csv', index=False)
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import KFold



print("Tensorflow Version: " + tf.__version__)
def create_model(show_summary = True):

    X_input = Input((875,))

    X = Dropout(0.075)(X_input)

    X = Dense(1024, activation='relu')(X)

    X = Dropout(0.2)(X)

    X = Dense(2056, activation='relu')(X)

    X = Dropout(0.3)(X)

    X = Dense(1024, activation='relu')(X)

    X = Dropout(0.1)(X)

    X = Dense(600, activation='relu')(X)

    X = Dropout(0.05)(X)

    X = Dense(400, activation='relu')(X)

    X = Dropout(0.01)(X_input)

    X = Dense(206, activation='sigmoid')(X)    #We need sigmoid not Softmax because labels are mostly independent of each other.

    

    model = Model(inputs = X_input, outputs = X)

    opt = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['AUC'])   #Closest default loss function (to competition metric) is logloss or categorical cross entropy. I will also be using AUC just as a side metric for supervision, but shouldn't be taken very seriously.

    if show_summary:

        model.summary()

    return model
model = create_model()
nFolds = 5



kf = KFold(n_splits = nFolds)

trd = train.drop(['sig_id'], axis=1).values

targetd = train_target.drop(['sig_id'], axis=1).values

testd = test.drop(['sig_id'], axis=1).values



preds = np.zeros((3982, 206))



fold = 1

for train_index, test_index in kf.split(trd):

    

    print("Fold  " + str(fold))

    fold += 1

    

    x_train, x_test = trd[train_index], trd[test_index]

    y_train, y_test = targetd[train_index], targetd[test_index]



    model = create_model(False)

    history = 0

    history = model.fit(x_train, y_train, batch_size = 32, epochs=65, shuffle = True, validation_data = (x_test, y_test), verbose=0)

    

    fig = plt.figure(figsize=(9, 5))



    fig.add_subplot(1, 2, 1)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper right')

    

    fig.add_subplot(1, 2, 2)

    plt.plot(history.history['auc'])

    plt.plot(history.history['val_auc'])

    plt.title('Model AUC')

    plt.ylabel('AUC')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()



    foldPred = model.predict(testd)

    preds += foldPred / nFolds
preds
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



for i in range(len(sub.columns)):

    if i != 0:

        col = sub.columns[i]

        sub[col] = preds[:, i - 1]

sub
sub.to_csv('submission.csv', index = False)