# Importing the necessary packages

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')
# Listing the files

os.listdir("../input/iris")
path = "../input/iris"
# Loading the data

iris = pd.read_csv(f"{path}/Iris.csv")
# Check the first two rows of the dataset

iris.head(2)
# Checking for any missing values and inconsistent data-types in the dataset

iris.info()
# Dropping the unwanted columns

iris = iris.drop(['Id'], 1)
# Let us start by understanding the distribution of the data

iris.describe()
# Function to perform Univariate Analysis

def plot_distribution(data):

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(23,3))

    fig.suptitle('Univariate distribution of data')

    

    for ax, col in zip(ax.flatten(), data.columns):

        if data[col].dtype in ['int64', 'float64', 'bool']:

            sns.distplot(data[col], ax=ax, hist=False)

        else:

            sns.countplot(data[col])
# Plot the distribution

plot_distribution(iris)
sns.pairplot(iris, hue='Species', markers='+', height=3)

plt.show()
sns.heatmap(iris.corr(), annot=True)

plt.show()
from sklearn.ensemble import RandomForestClassifier
# Get feature importances for tree based models

def get_feature_importances(model, cols):

    return pd.DataFrame({"Column": cols, "Importance": model.feature_importances_}).sort_values('Importance', ascending=False)
base_x, base_y = iris.iloc[:, :-1], iris.iloc[:, -1]
model = RandomForestClassifier(n_estimators=100)

model.fit(base_x, base_y)
get_feature_importances(model, iris.columns[:-1])
plt.figure(figsize=(5, 5))

setosa = iris.loc[iris.Species == "Iris-setosa"]

versicolor = iris.loc[iris.Species == "Iris-versicolor"]

virginica = iris.loc[iris.Species == "Iris-virginica"]

ax = sns.kdeplot(setosa.PetalLengthCm, setosa.PetalWidthCm, cmap="Reds", shade=True, shade_lowest=False)

ax = sns.kdeplot(versicolor.PetalLengthCm, versicolor.PetalWidthCm, cmap="Greens", shade=True, shade_lowest=False)

ax = sns.kdeplot(virginica.PetalLengthCm, virginica.PetalWidthCm, cmap="Blues", shade=True, shade_lowest=False)
iris['Species'][(iris['PetalLengthCm'] < 3) & (iris['PetalWidthCm'] < 1)].value_counts()
# Remove the highly correlated columns

rem_cols = ['SepalLengthCm', 'SepalWidthCm']



data = iris.drop(rem_cols, 1)
data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Species'] = le.fit_transform(data['Species'])
features = data[['PetalLengthCm', 'PetalWidthCm']]

target = data['Species']



SEED = 10

SIZE= 0.3
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=SEED, test_size=SIZE, stratify=target)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import f1_score, confusion_matrix, classification_report
models = {'LR': LogisticRegression(random_state=SEED),

          'KNN': KNeighborsClassifier(), 

          'SVM': SVC(random_state=SEED), 

          'Decision Tree': DecisionTreeClassifier(random_state=SEED)}

res, preds = {}, {}
for name, model in models.items():

    res[name] = model.fit(x_train, y_train)

    preds[name] = model.predict(x_test)

    print(f"[{name}]: {f1_score(preds[name], y_test, average='weighted')}")
from mlxtend.plotting import plot_decision_regions
# Decision Boundary on train set

f, ax = plt.subplots(2,2, figsize=(15,5))

plt.subplot(1, 2, 1)

ax1 = plot_decision_regions(np.array(x_train), np.array(y_train), clf=res['SVM'])

ax1.set_title("Performance on Train set")

plt.subplot(1, 2, 2)

ax2 = plot_decision_regions(np.array(x_test), np.array(y_test), clf=res['SVM'])

ax2.set_title("Performance on Test set")

plt.show()
import joblib

joblib.dump(res['SVM'], 'iris_svm.pkl')