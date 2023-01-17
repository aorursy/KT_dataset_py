import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/iris.csv")
df.head()
df.info()
df.isnull().sum()
iris = df
iris.describe()
print(df.shape)
print("What categories are there and how many instances for each category?\n")
print(iris["variety"].value_counts())
print("\n\nWhat are the unique categories?")
print(iris["variety"].unique())
# How many unique values are there
print("\n\nHow many unique categories there are?")
print(iris["variety"].nunique())
print("\n\nWhat is the shape of our dataframe?")
print(iris.shape)
iris.loc[iris["variety"] == "Iris-setosa", ["variety"]] = "Setosa"
iris.loc[iris["variety"] == "Iris-virginica", ["variety"]] = "Virginica"
iris.loc[iris["variety"] == "Iris-versicolor", ["variety"]] = "Versicolor"
iris.head()
import seaborn as sns
sns.countplot(x='variety', data=iris)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.scatterplot(x = 'sepal.length' ,y='sepal.width', data=iris)
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal.length', y='sepal.width', data=iris, hue='variety')
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.scatterplot(x='sepal.length', y='sepal.width', hue='variety', data=iris)
plt.subplot(1,2,2)
sns.scatterplot(x='sepal.width', y='sepal.length', hue='variety', data=iris)
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.scatterplot(x='petal.length', y='petal.width', hue='variety', data=iris)
plt.subplot(1,2,2)
sns.scatterplot(x='petal.width', y='petal.length', hue='variety', data=iris)
# height parameter decides the size of the plot here
sns.lmplot(x="sepal.length", y="sepal.width", hue="variety", data=iris, height=8, markers=["o", "x", "^"])
sns.lmplot(x="petal.length", y="petal.width", hue="variety", data=iris, height=8)
#We will use median of each feature and try to compare them 
species = ["Setosa", "Versicolor", "Virginica"]
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
d = {"Median":[], "Features":[],  "Species":[]}
for s in species:
    for f in features:
        d["Median"].append(iris[iris["variety"] == s][f].mean())
        d["Features"].append(f)
        d["Species"].append(s)

        
new_df = pd.DataFrame(data=d)
new_df
plt.figure(figsize=(12, 6))
sns.lineplot(x="Features", y="Median", hue="Species", data=new_df)
plt.figure(figsize=(12, 6))
sns.pointplot(x="Features", y="Median", hue="Species", data=new_df)
#univariate analysis
plt.figure(figsize=(8, 6))
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=True, size=7)
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
sns.stripplot(x="variety", y="petal.length", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 2)
sns.stripplot(x="variety", y="petal.width", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 3)
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 4)
sns.stripplot(x="variety", y="sepal.width", data=iris, jitter=True, size=7)
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
sns.stripplot(x="variety", y="petal.length", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 2)
sns.stripplot(x="variety", y="petal.width", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 3)
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 4)
sns.stripplot(x="variety", y="sepal.width", data=iris, jitter=False, size=7)
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.swarmplot(x="variety", y="petal.length", data=iris, size=7)
plt.subplot(2, 2, 2)
sns.swarmplot(x="variety", y="petal.width", data=iris, size=7)
plt.subplot(2, 2, 3)
sns.swarmplot(x="variety", y="sepal.length", data=iris, size=7)
plt.subplot(2, 2, 4)
sns.swarmplot(x="variety", y="sepal.width", data=iris, size=7)
sns.set(style='whitegrid')
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.violinplot(x="variety", y="petal.length", data=iris, size=7)
plt.subplot(2, 2, 2)
sns.violinplot(x="variety", y="petal.width", data=iris, size=7)
plt.subplot(2, 2, 3)
sns.violinplot(x="variety", y="sepal.length", data=iris, size=7)
plt.subplot(2, 2, 4)
sns.violinplot(x="variety", y="sepal.width", data=iris, size=7)
plt.figure(figsize=(10, 10))
binsize = 10
plt.subplot(2, 2, 1)
sns.distplot(a=iris["petal.length"], bins=binsize)
plt.subplot(2, 2, 2)
sns.distplot(a=iris["petal.width"], bins=binsize)
plt.subplot(2, 2, 3)
sns.distplot(a=iris["sepal.length"], bins=binsize)
plt.subplot(2, 2, 4)
sns.distplot(a=iris["sepal.width"], bins=binsize)
sns.jointplot(x="sepal.length", y="sepal.width", kind='hex', data=iris[iris["variety"] == "Setosa"])
X = iris.drop(['variety'], axis=1)
y = iris['variety']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    

print(metrics.accuracy_score(y_test, y_pred))
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
from sklearn import svm
model = svm.SVC()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print(metrics.accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print(metrics.accuracy_score(y_test,y_pred))