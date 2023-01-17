import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
dataset = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
dataset.head()
dataset.info()
dataset.isnull().count()
dataset.describe()
X = dataset.drop(['species'], axis=1)

Y = dataset['species']
X.shape, Y.shape
sns.heatmap(dataset.corr(), annot = True);

#annot = True adds the numbers onto the squares
sns.set_style("whitegrid")

sns.pairplot(dataset, hue="species", markers='+',size=4)

plt.show()
#Splitting the data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
#create the model instance

model = LogisticRegression()

#fit the model on the training data

model.fit(X_train, y_train)

#the score, or accuracy of the model

model.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=10)

print(np.mean(scores))
df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)

df_coef
predictions = model.predict(X_test)

#compare predicted values with the actual scores

compare_df = pd.DataFrame({'actual': y_test, 'predicted': predictions})

compare_df = compare_df.reset_index(drop = True)

compare_df
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

#Logistic Regression

pd.DataFrame(confusion_matrix(y_test, predictions))
#KNN metric

pd.DataFrame(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
probs = model.predict_proba(X_test)

#put the probabilities into a dataframe for easier viewing

Y_pp = pd.DataFrame(model.predict_proba(X_test), 

             columns=['class_0_pp', 'class_1_pp', 'class_2_pp'])

Y_pp.head()