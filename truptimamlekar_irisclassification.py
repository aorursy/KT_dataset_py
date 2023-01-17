#Load libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
df=pd.read_csv('../input/iris/Iris.csv')

df.head()
print(df.describe())
print(df.groupby('Species').size())
df.info()
df.isnull().sum()
df=df.drop(['Id'],axis=1)

df.head()
sns.set(font_scale=1.5)

plt.figure(figsize=(8,5))

corr = (df.corr())

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap="YlGnBu",annot=True,linewidths=.5, fmt=".2f")

plt.title("Pearson Correlation of all Elements")
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.boxplot(data=df, palette="muted")
df.hist (bins=10,figsize=(20,20))

plt.show ()
sns.set(style="ticks", color_codes=True)

g = sns.pairplot(df)
g = sns.pairplot(df, hue="Species",palette="husl", markers=["o", "s", "D"])
f,ax=plt.subplots(1,1,figsize=(25,6))

ax = sns.scatterplot(x="SepalLengthCm", y="Species",color = "orange",data=df)

ax = sns.scatterplot(x="SepalWidthCm", y="Species",color = "red",data=df)

ax = sns.scatterplot(x="PetalLengthCm", y="Species",color = "green",data=df)

ax = sns.scatterplot(x="PetalWidthCm", y="Species",color = "blue",data=df)
f,ax=plt.subplots(1,1,figsize=(25,6))

df['Species'].replace([0], 'Iris_Setosa', inplace=True) 

df['Species'].replace([1], 'Iris_Vercicolor', inplace=True) 

df['Species'].replace([2], 'Iris_Virginica', inplace=True)   

sns.kdeplot(df.loc[(df['Species']=='Iris-virginica'), 'SepalLengthCm'], color='b', shade=True, Label='Iris_Virginica')

sns.kdeplot(df.loc[(df['Species']=='Iris-setosa'), 'SepalLengthCm'], color='g', shade=True, Label='Iris_Setosa')

sns.kdeplot(df.loc[(df['Species']=='Iris-versicolor'), 'SepalLengthCm'], color='r', shade=True, Label='Iris_Vercicolor')

plt.xlabel('Sepal Length') 

plt.ylabel('Probability Density') 
f,axes = plt.subplots(1,1,figsize=(3,5),sharex = True,sharey =True)

s=np.linspace(0,3,10)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

x = df['PetalWidthCm'].values

y = df['PetalLengthCm'].values

sns.kdeplot(x,y,cmap=cmap,shade=True,cut = 5)
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.violinplot(x="Species", y="SepalLengthCm",ax=ax[0][0],data=df, palette="muted")

sns.violinplot(x="Species", y="PetalWidthCm",data=df,ax=ax[0][1], palette="muted")

sns.violinplot(x="Species", y="PetalLengthCm",ax=ax[1][0],data=df, palette="muted")

sns.violinplot(x="Species", y="SepalWidthCm",ax=ax[1][1],data=df, palette="muted")
f,axes=plt.subplots (1,1,figsize=(15,4))

sns.distplot(df['SepalLengthCm'],kde=True,hist=True,color="g")

sns.distplot(df['SepalWidthCm'],kde=True,hist=True,color="r")

sns.distplot(df['PetalLengthCm'],kde=True,hist=True,color="b")

sns.distplot(df['PetalWidthCm'],kde=True,hist=True,color="yellow")

plt.xlabel('Quantity') 

#plt.ylabel('Probability Density') 
X = df.drop(['Species'],axis=1)

Y = df['Species']

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix

model = KNeighborsClassifier()

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=model.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
model = SVC()

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=model.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
model = RandomForestClassifier(n_estimators=5)

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=model.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
model = LogisticRegression()

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=model.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', 

                    max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

# Plotting the results onto a line graph, 

# `allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') # Within cluster sum of squares

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++',

                max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters - On the first two columns

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 

            s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 

            s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],

            s = 100, c = 'green', label = 'Iris-virginica')



# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 

            s = 100, c = 'yellow', label = 'Centroids')



plt.legend()
y=df['Species']
# Defining the decision tree algorithm

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(x,y)



print('Decision Tree Classifer Created')

# Install required libraries

!pip install pydotplus

!apt-get install graphviz -y
!pip install pydotplus
# Import necessary libraries for graph viz

!pip install --upgrade scikit-learn==0.20.3

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus



# Visualize the graph

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, feature_names=X.columns,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())