

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings 

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
class_ = pd.read_csv("../input/class.csv")

zoo = pd.read_csv("../input/zoo.csv")
zoo.head()
zoo.info()
zoo.describe()
zoo.drop("animal_name",axis=1,inplace=True)
color_list = [("red" if i ==1 else "blue" if i ==0 else "yellow" ) for i in zoo.hair]
unique_list = list(set(color_list))

unique_list
pd.plotting.scatter_matrix(zoo.iloc[:,:7],

                                       c=color_list,

                                       figsize= [20,20],

                                       diagonal='hist',

                                       alpha=1,

                                       s = 300,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="hair", data=zoo)

plt.xlabel("Hair")

plt.ylabel("Count")

plt.show()

zoo.loc[:,'hair'].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

x,y = zoo.loc[:,zoo.columns != 'hair'], zoo.loc[:,'hair']

knn.fit(x,y)

prediction = knn.predict(x)

print("Prediction = ",prediction)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 1)

x,y = zoo.loc[:,zoo.columns != 'hair'], zoo.loc[:,'hair']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('With KNN (K=1) accuracy is: ',knn.score(x_test,y_test)) # accuracy
k_values = np.arange(1,25)

train_accuracy = []

test_accuracy = []



for i, k in enumerate(k_values):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



    # Plot

plt.figure(figsize=[13,8])

plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')

plt.plot(k_values, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(k_values)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

x = np.array(zoo.loc[:,"eggs"]).reshape(-1,1)

y = np.array(zoo.loc[:,'hair']).reshape(-1,1)



plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('Egg')

plt.ylabel('Hair')

plt.show()
from sklearn.linear_model import LinearRegression

regression = LinearRegression()



predict_space = np.linspace(min(x),max(x)).reshape(-1,1)

regression.fit(x,y)

predicted = regression.predict(predict_space)



print("R^2 Score: ",regression.score(x,y))



plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('Egg')

plt.ylabel('Milk')

plt.show()

from sklearn.model_selection import cross_val_score

regression = LinearRegression()

k=5

cv_result = cross_val_score(regression,x,y,cv=k)

print("CV Scores: ",cv_result)

print("CV Average: ",np.sum(cv_result)/k)
from sklearn.linear_model import Ridge

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)

ridge = Ridge(alpha= 0.001,normalize = True)

ridge.fit(x_train,y_train)

ridge_predict = ridge.predict(x_test)

print("Ridge Score: ",ridge.score(x_test,y_test))
from sklearn.linear_model import Lasso

x = np.array(zoo.loc[:,['eggs','airborne','fins','legs',"hair","class_type"]])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)

lasso = Lasso(alpha = 0.0001, normalize = True)

lasso.fit(x_train,y_train)

ridge_predict = lasso.predict(x_test)

print('Lasso score: ',lasso.score(x_test,y_test))

print('Lasso coefficients: ',lasso.coef_)
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = zoo.loc[:,zoo.columns != "hair"], zoo.loc[:,"hair"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1 )

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Confisuon Matrix: \n",cm)

print("Classification Report: \n",classification_report(y_test,y_pred))
sns.heatmap(cm,annot=True,fmt="d")

plt.show()
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

#hair = 1 no = 0 

x,y = zoo.loc[:,(zoo.columns != 'hair')], zoo.loc[:,'hair']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
# grid search cross validation with 2 hyperparameter

# 1. hyperparameter is C:logistic regression regularization parameter

# 2. penalty l1 or l2

# Hyperparameter grid

param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,param_grid,cv=3)

logreg_cv.fit(x_train,y_train)



# Print the optimal parameters and best score

print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))

print("Best Accuracy: {}".format(logreg_cv.best_score_))
# get_dummies

df = pd.get_dummies(zoo)

df.head(10)
# SVM, pre-process and pipeline

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

steps = [('scalar', StandardScaler()),

         ('SVM', SVC())]

pipeline = Pipeline(steps)

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)

cv.fit(x_train,y_train)



y_pred = cv.predict(x_test)



print("Accuracy: {}".format(cv.score(x_test, y_test)))

print("Tuned Model Parameters: {}".format(cv.best_params_))
plt.scatter(zoo['hair'],zoo['tail'])

plt.xlabel('Hair')

plt.ylabel('Tail')

plt.show()
data2 = zoo.loc[:,['tail','hair']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(zoo['hair'],zoo['tail'],c = labels)

plt.xlabel('Hair')

plt.xlabel('Tail')

plt.show()
# cross tabulation table

df = pd.DataFrame({'labels':labels,"hair":zoo['hair']})

ct = pd.crosstab(df['labels'],df['hair'])

print(ct)
inertia_list = np.empty(8)

for i in range(1,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(zoo)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()

# we choose the elbow < 1
data2 = zoo.drop("hair",axis=1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data2)

labels = pipe.predict(data2)

df = pd.DataFrame({'labels':labels,"hair":zoo['hair']})

ct = pd.crosstab(df['labels'],df['hair'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(data2.iloc[:20,0:5],method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 5)

plt.show()
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100,random_state=42)

transformed = model.fit_transform(data2)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list )

plt.xlabel('Values')

plt.ylabel('Count')

plt.show()
from sklearn.decomposition import PCA

model = PCA()

model.fit(data2[0:4])

transformed = model.transform(data2[0:4])

print('Principle components: ',model.components_)
# PCA variance

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(data2)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
# apply PCA

pca = PCA(n_components = 2)

pca.fit(data2)

transformed = pca.transform(data2)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list)

plt.show()