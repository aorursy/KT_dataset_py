import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, confusion_matrix



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/iris/Iris.csv')
data.head()
data.drop('Id',axis=1, inplace=True)
data.info()



# no missing data
data.describe()



# there is no huge gap between data
sns.countplot(data['Species'])



# we got balance data with 3 targets

# we may want to try to normalize the data min(0.1), max(7.9)
sns.pairplot(data, hue=data.columns[4], height=5)



# we can easily separate them with linear models

# we will plot petal_length vs petal_width
sns.FacetGrid(data, hue='Species', size=4).map(plt.scatter, 'PetalWidthCm', 'PetalLengthCm').add_legend();



# we can separate them lineary

# for Iris-setosa we can just say if petal_wdit <0.75 & petal_length < 2 then "SETOSA"
# lets try to see the mean of the length and width for each class



data['length'] = data['PetalLengthCm'] + data['SepalLengthCm']

data['width'] = data['PetalWidthCm'] + data['SepalWidthCm']
plt.plot(data.groupby(data['Species']).mean()['length'], '-o')

plt.plot(data.groupby(data['Species']).mean()['width'], '-o')

plt.legend(['length','width']);



# we can see the difference .
# see if we got outliers



plt.subplots(3,2, figsize=(15,10))

for i,col in enumerate(data.columns.drop('Species')):

    plt.subplot(3,2,i+1)

    sns.boxplot(data[col])

    

# we got some outliers in sepal_width , we can consider them as normal data .
# we will not use the new features



data.drop(['length','width'], axis=1, inplace=True)
data.head()
# first let's see if we got duplicated rows



print(data.duplicated().any())

data[data.duplicated(keep=False)]



#let's drop them
data.drop_duplicates(keep='first',inplace=True)

print(data.duplicated().any())
# we will shuffle and split the data

# we will take 10 samples for testing



#shuffle

np.random.seed(42) #same order every time

data = data.sample(frac=1)



# 10 samples

data_10 = data[:10]

data.drop(data.index[:10], inplace=True)



# split the data to X and y

X_train = data.drop('Species', axis=1)

y_train = data['Species']

X_10 = data_10.drop('Species', axis=1)

y_10 = data_10['Species']
# train the model



model = SVC(kernel='rbf',random_state=42)

model.fit(X_train, y_train)
# let's build a function to evaluate the data



def score(y_true, y_pred):

    rec = recall_score(y_true, y_pred, average=None)

    pred = precision_score(y_true, y_pred, average=None)

    f1 = f1_score(y_true, y_pred, average=None)

    cm = confusion_matrix(y_true, y_pred)

    

    return 'rec: {}\n pred: {}\n f1: {}\n cm:\n {}\n'.format(rec, pred, f1, cm)
# evaluate the model

y_pred_train = model.predict(X_train)

y_pred_cross = cross_val_predict(model, X_train, y_train, cv=20)



print('train_data:\n',score(y_train, y_pred_train))

print('cross_data:\n' ,score(y_train, y_pred_cross))



#that looks good, no OF and good accuracy
#let's predict the 10 samples



pred = pd.Series(model.predict(X_10))

true = pd.Series(np.array(y_10))



print('true\t\t\tpred\n')

for x, y in zip(true,pred):

    print(x, '==>', y)

    

# we can see that the model predict all the "setosa" True

# but got 2 False in virginica & versicolor



#that's not bad :)