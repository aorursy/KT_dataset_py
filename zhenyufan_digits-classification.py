import pandas as pd

import numpy as np

import warnings

import hypertools as hyp

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import datasets, svm, metrics

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error



from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import ROCAUC
digits = load_digits()
print("The type of this dataset is %s" % type(digits))

print("The keys of this dataset are %s" % digits.keys())



print("The shape of this dataset images is {0}".format(digits.images.shape))

print("The shape of this dataset data is {0}".format(digits.data.shape))
plt.gray() 

plt.matshow(digits.images[0]) 
fig = plt.figure(figsize=(6, 6))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



for i in range(64):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # label the image with the target value

    ax.text(0, 7, str(digits.target[i]))
data = digits.data

hue = digits.target

hyp.plot(data, '.', hue=hue)
hue = digits.target.astype('str')

hyp.plot(data, '.', reduce='UMAP', hue=hue, ndims=2)
X = digits.data

y = digits.target
warnings.filterwarnings('ignore')



test_size = np.linspace(0.1, 1, num=9, endpoint=False)

random_state = np.arange(0, 43)



grid_results= []

for testsize in test_size:

    for randomstate in random_state:

        try:

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

            log = LogisticRegression()

            log.fit(X_train, y_train)

            y_train_pred = log.predict(X_train)

            y_test_pred = log.predict(X_test)     

            grid_results.append([testsize, randomstate, mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)])

            grid_frame = pd.DataFrame(grid_results)

            grid_frame.rename(columns={0:'Test Size', 1:'Random State', 2:'MSE of Train', 3:'MSE of Test'}, inplace=True)

        except Exception:

            print(Exception.with_traceback())

            print('error')

            continue

grid_frame.head()
grid_frame_best = grid_frame[grid_frame['MSE of Test'] == grid_frame['MSE of Test'].min()]

grid_frame_best
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# LogisticRegression

model_log = LogisticRegression()

# ConfusionMatrix

cm = ConfusionMatrix(model_log, classes=[0,1,2,3,4,5,6,7,8,9])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.poof()
classes=[0,1,2,3,4,5,6,7,8,9]

visualizer = ROCAUC(model_log, classes=classes)



visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)  

g = visualizer.poof()            
model_log.fit(X_train, y_train)

predicted = model_log.predict(X_test)

expected = y_test

print(metrics.classification_report(expected, predicted))
fig = plt.figure(figsize=(6, 6)) 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



# plot the digits: each image is 8x8 pixels

for i in range(64):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,

              interpolation='nearest')



    # label the image with the target value

    if predicted[i] == expected[i]:

        ax.text(0, 7, str(predicted[i]), color='green')

    else:

        ax.text(0, 7, str(predicted[i]), color='red')
model_svc = svm.SVC()

cm = ConfusionMatrix(model_svc, classes=[0,1,2,3,4,5,6,7,8,9])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.poof()
visualizer = ROCAUC(model_svc, classes=classes)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)  

g = visualizer.poof() 
model_svc.fit(X_train, y_train)

predicted = model_svc.predict(X_test)

expected = y_test

print(metrics.classification_report(expected, predicted))
fig = plt.figure(figsize=(6, 6)) 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



# plot the digits: each image is 8x8 pixels

for i in range(64):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,

              interpolation='nearest')



    # label the image with the target value

    if predicted[i] == expected[i]:

        ax.text(0, 7, str(predicted[i]), color='green')

    else:

        ax.text(0, 7, str(predicted[i]), color='red')
model_nb = GaussianNB()

cm = ConfusionMatrix(model_nb, classes=[0,1,2,3,4,5,6,7,8,9])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.poof()
visualizer = ROCAUC(model_nb, classes=classes)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)  

g = visualizer.poof() 
model_nb.fit(X_train, y_train)

predicted = model_nb.predict(X_test)

expected = y_test

print(metrics.classification_report(expected, predicted))
fig = plt.figure(figsize=(6, 6)) 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



# plot the digits: each image is 8x8 pixels

for i in range(64):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,

              interpolation='nearest')



    # label the image with the target value

    if predicted[i] == expected[i]:

        ax.text(0, 7, str(predicted[i]), color='green')

    else:

        ax.text(0, 7, str(predicted[i]), color='red')