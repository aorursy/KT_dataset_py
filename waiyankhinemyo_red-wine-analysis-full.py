#import libraries 

#structures
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

#analysis
from sklearn.metrics import confusion_matrix, accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load dataset
data = '../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'
dataset = pd.read_csv(data)
dataset.shape
dataset.dtypes
dataset.describe()
#check for missing data
dataset.isnull().any().any()
#check for unreasonable data
dataset.applymap(np.isreal)
sns_plot = sns.pairplot(dataset)
sns_plot = sns.distplot(dataset['quality'])
#create new column; "quality_class"
dataset['quality_class'] = dataset['quality'].apply(lambda value: 1 if value < 5 else 2 if value < 7 else 3)
#set x and y
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = dataset.iloc[:,0:11]
y = dataset['quality_class']

#stadardize data
X_scaled = StandardScaler().fit_transform(X)

#get feature names
X_columns = dataset.columns[:11]

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
dataset.head()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=6)
pc_X = pca.fit_transform(X_scaled)
pc_columns = ['pc1','pc2','pc3','pc4','pc5','pc6']
print(pca.explained_variance_ratio_.sum())
print(pca.explained_variance_ratio_)
#split train and test data for pca
Xpc_train, Xpc_test, ypc_train, ypc_test = train_test_split(pc_X, y, random_state=42)
#get correlation map
corr_mat=dataset.corr()
#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()

#save file
#sns_plot.get_figure().savefig('corr_mat.jpg')
#check for highly correlated values to be removed
target = 'quality'
candidates = corr_mat.index[
    (corr_mat[target] > 0.5) | (corr_mat[target] < -0.5)
].values
candidates = candidates[candidates != target]
print('Correlated to', target, ': ', candidates)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# import model
from sklearn.linear_model import LinearRegression

#instantiate
linReg = LinearRegression()

start_time = time.time()
# fit out linear model to the train set data
linReg_model = linReg.fit(X_train, y_train)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
#get coefficient values
coeff_df = pd.DataFrame(linReg.coef_, X_columns, columns=['Coefficient'])  
coeff_df
#validate model
y_pred = linReg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
df1.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print the intercept and coefficients
print('Intercept: ',linReg.intercept_)
print('r2 score: ',linReg.score(X_train, y_train))
sns_plot = sns.distplot(dataset['quality'])
#the dataset contains 6 unique values.
len(dataset['quality'].unique())
from sklearn.linear_model import LogisticRegression
logReg=LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)

start_time = time.time()
# Building a Logistic Regression Model
logReg.fit(X_train, y_train)

#print duration of model
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
# Calculate Accuracy Score
y_pred = logReg.predict(X_test)
print('Accuracy score: ', accuracy_score(y_test, y_pred))
#Calculate Confusion Matrix
print('confusion matrix: ','\n',confusion_matrix(y_test,y_pred, labels=[1,2,3]))
#apply pca
start_time = time.time()

# Building a Logistic Regression Model
logReg.fit(Xpc_train, ypc_train)

#print duration of model
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
# Calculate Accuracy Score
y_pred = logReg.predict(Xpc_test)
print('Accuracy score with PCA applied: ', accuracy_score(ypc_test, y_pred))
# Calculate Confusion Matrix
print('confusion matrix: ','\n',confusion_matrix(ypc_test,y_pred, labels=[1,2,3]))
from sklearn.neighbors import KNeighborsClassifier
k_array = np.arange(1, 17, 2)
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    print('n_neighbours: ',k)
    print('accuracy score: ',ac)
    print('confusion matrix: ','\n',confusion_matrix(y_test, y_pred))
    print('-------------------------------')
#apply pca
k_array = np.arange(1, 17, 2)
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(Xpc_train, ypc_train)
    y_pred=knn.predict(Xpc_test)
    ac = accuracy_score(ypc_test, y_pred)
    print('n_neighbours: ',k)
    print('accuracy score: ',ac)
    print('confusion matrix: ','\n',confusion_matrix(ypc_test, y_pred))
    print('-------------------------------')
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
#train model
start_time = time.time()
dt.fit(X_train,y_train)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
# Calculate Accuracy Score
dt_predict = dt.predict(X_test)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_acc_score)
# Calculate Confusion Matrix
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
print('confusion matrix: ','\n',dt_conf_matrix)
#training with Gini
def decTreeScore2(crit = 'gini',  maxDepth = 2, minSamples = 1, minSplit = 2):
    dect = DecisionTreeClassifier(criterion = crit, max_depth = maxDepth, min_samples_leaf = minSamples, 
                                 min_samples_split = minSplit, random_state= 42)
    dect.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, dect.predict(X_test))
    print(accuracy)
    return accuracy
start_time=time.time()
decTreeScore2()
today=date.today()
print("---%s seconds---"% (time.time()-start_time))
decTreeScore2(crit = 'entropy')
#if we use entropy to calculate infomation gain instead of gini score, the accuracy drops
# find the max allowed depth for the decision tree
for i in np.arange(1, 15, 1):
    decTreeScore2(maxDepth = i)
# find maximum_samples leaf of the tree
for i in np.arange(1, 10, 1):
    decTreeScore2(minSamples = i)
# find minimum_samples_split of the tree
for i in np.arange(2, 10,1):
    decTreeScore2(minSplit = i)
# decision tree model
# import graphviz and sklearn.tree
from sklearn import tree
import graphviz
from graphviz import Source
dot_data = tree.export_graphviz(dt, out_file=None, max_depth=2,class_names=True,feature_names= X_columns, filled=True, rounded=True)
graph = graphviz.Source(dot_data) 
graph
#apply pca
dt = tree.DecisionTreeClassifier(max_depth=2)
dt.fit(Xpc_train, ypc_train)
#training with Gini
def decTreeScore2(crit = 'gini',  maxDepth = 2, minSamples = 1, minSplit = 2):
    dect = DecisionTreeClassifier(criterion = crit, max_depth = maxDepth, min_samples_leaf = minSamples, 
                                 min_samples_split = minSplit, random_state= 42)
    dect.fit(Xpc_train, ypc_train)
    accuracy = accuracy_score(ypc_test, dect.predict(Xpc_test))
    print(accuracy)
    return accuracy
start_time=time.time()
decTreeScore2()
today=date.today()
print("---%s seconds---"% (time.time()-start_time))
decTreeScore2(crit = 'entropy')
#if we use entropy to calculate infomation gain instead of gini score, the accuracy drops
# use different maximum depth of the tree
for i in np.arange(1, 15, 1):
    decTreeScore2(maxDepth = i)
# use different maximum_samples leaf of the tree
for i in np.arange(1, 10, 1):
    decTreeScore2(minSamples = i)
dot_data = tree.export_graphviz(dt, out_file=None, max_depth=2,class_names=True, filled=True, rounded=True)
graph = graphviz.Source(dot_data) 
graph
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#In this step we will be importing and preparing dataset that is to be analyzed, in this case we will be using
#‘winequality-red.csv’ dataset. 
#dataset = pd.read_csv('winequality-red.csv',sep=';')
dataset['quality_class'] = dataset['quality'].apply(lambda value: 1 if value < 5 else 2 if value < 7 else 3)
dataset['quality_class'] = pd.Categorical(dataset['quality_class'], categories=[1,2,3])
dataset['quality_class'] = dataset['quality_class'].astype(int)
dataset.head()
quality_label_sums= dataset['quality_class'].value_counts()
quality_label_percentage = quality_label_sums/len('quality_class')
print(quality_label_sums)
print(quality_label_percentage)
#visualize quality_class
j = sns.countplot(x='quality_class', data=dataset)
plt.show(j)
dataset['quality_class'] = dataset['quality_class'].astype(int)
dataset = pd.get_dummies(dataset, columns=['quality_class'])
dataset.head()
Xn = dataset.iloc[:,0:11].values
Yn = dataset.iloc[:,12:].values

Xn = StandardScaler().fit_transform(Xn)

Xn_train, Xn_test, Yn_train, Yn_test = train_test_split(Xn, Yn,random_state=42)

print(Xn_train.shape, Yn_train.shape, Xn_test.shape, Yn_test.shape)
model = Sequential()
model.add(Dense(30, input_dim=11, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])

start_time = time.time()
#train model
history = model.fit(x = Xn_train, y = Yn_train,batch_size=128, epochs = 800,verbose=1,validation_data=(Xn_test, Yn_test))

#get model training duration
today= date.today()
print('---%s seconds---'%(time.time()-start_time))
# Calculation of Loss and Accuracy metrics
loss, accuracy = model.evaluate(Xn_test, Yn_test)
print('loss: ', loss, ', accuracy: ', accuracy)
predictions = model.predict(Xn_test)
print('\nPrediction:')
for i in np.arange(len(predictions)):
    print('Actual: ', Yn_test[i], ', Predicted: ', predictions[i])

predictions=np.argmax(predictions, axis=1)
Yn_test = np.argmax(Yn_test, axis=1)
# Training History - Model Accuracy
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Training History - Loss Accuracy
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Calculation of confusion matrix
#from sklearn.metrics import confusion_matrix
confusion_matrix(Yn_test, predictions)
Y = dataset.iloc[:,12:].values

X_train, X_test, Y_train, Y_test = train_test_split(pc_X, Y, random_state=42)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
model = Sequential()
model.add(Dense(30, input_dim=6, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])

start_time = time.time()
history = model.fit(x = X_train, y = Y_train,batch_size=128, epochs = 800,verbose=1,validation_data=(X_test, Y_test))

today= date.today()
print('---%s seconds---'%(time.time()-start_time))
# Calculation of Loss and Accuracy metrics
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: ', loss, ', accuracy: ', accuracy)
predictions = model.predict(X_test)
print('\nPrediction:')
for i in np.arange(len(predictions)):
    print('Actual: ', Y_test[i], ', Predicted: ', predictions[i])
    
predictions=np.argmax(predictions, axis=1)
Y_test = np.argmax(Y_test, axis=1)
# Training History - Model Accuracy
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Training History - Loss Accuracy
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Calculation of confusion matrix
confusion_matrix(Y_test, predictions)
#import libraries
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
#try to find optimal k using the elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=12, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
f3, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#Applying kmeans to the dataset, set k=2
kmeans = KMeans(n_clusters = 2)
start_time = time.time()
clusters = kmeans.fit_predict(X_scaled)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
labels = kmeans.labels_
#2D plot
colors = 'rgbkcmy'
for i in np.unique(clusters):
    plt.scatter(X_scaled[clusters==i,0],
               X_scaled[clusters==i,1],
               color=colors[i], label='Cluster' + str(i+1))
plt.legend()
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)

ax.scatter(X_scaled[:,0], X_scaled[:,3], X_scaled[:,10],c=y, edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=2: Acidity, Sugar, Alcohol', size=22)
#evaluate model
from sklearn.metrics import pairwise_distances
metrics.silhouette_score(X_scaled, labels, metric='euclidean')
kmeans.inertia_
#Applying kmeans to the dataset, set k=2
kmeans = KMeans(n_clusters = 2)
start_time = time.time()
clusters = kmeans.fit_predict(pc_X)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
labels = kmeans.labels_
#2D plot
colors = 'rgbkcmy'
for i in np.unique(clusters):
    plt.scatter(pc_X[clusters==i,0],
               pc_X[clusters==i,1],
               color=colors[i], label='Cluster' + str(i+1))
plt.legend()
#evaluate model
metrics.silhouette_score(pc_X, labels, metric='euclidean')
kmeans.inertia_
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
#plot dendrogram to determine number of clusters
plt.figure(figsize=(25, 10))
plt.title('Dendrogram')
plt.xlabel('Wine Details')
plt.ylabel('Euclidean distances')

dendrogram (
    linkage(X_scaled, 'ward')  # generate the linkage matrix
    ,leaf_font_size=8 # font size for the x axis labels
)
plt.axhline(y=8)
plt.show()
clustering = AgglomerativeClustering(linkage="ward", n_clusters=3)
#train model
start_time = time.time()
clustering.fit(X_scaled)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
#visualize clustering
colors = 'rgbkcmy'

for i in np.unique(clustering.labels_):
    plt.scatter(X_scaled[clustering.labels_ == i, 0], X_scaled[clustering.labels_ == i, 1],
                color=colors[i], label='Cluster ' + str(i + 1))

plt.legend()
plt.title('Hierarchical Clustering')
plt.xlabel(X_columns[1])
plt.ylabel(X_columns[2])
plt.show()
#evaluate model
labels = clustering.labels_
metrics.silhouette_score(X_scaled, labels, metric='euclidean')
clustering = AgglomerativeClustering(linkage="ward", n_clusters=3)
start_time = time.time()
clustering.fit(pc_X)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
#visualize clustering
colors = 'rgbkcmy'

for i in np.unique(clustering.labels_):
    plt.scatter(pc_X[clustering.labels_ == i, 0], 
                pc_X[clustering.labels_ == i, 1],
                color=colors[i], label='Cluster ' + str(i + 1))

plt.legend()

plt.title('Hierarchical Clustering')
plt.xlabel(pc_columns[0])
plt.ylabel(pc_columns[1])
plt.show()
#evaluate model
labels = clustering.labels_
metrics.silhouette_score(pc_X, labels, metric='euclidean')
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2, min_samples=7)
start_time = time.time()
clusters= dbscan.fit_predict(X_scaled)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
np.unique(clusters)
colors = 'rgbkcmy'
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    ax.scatter3D(X_scaled[clusters==i,0], X_scaled[clusters==i,1],X_scaled[clusters==i,4],
                #color=colors[i], 
                 label=label)

plt.legend()
plt.show()
#evaluate model
labels = dbscan.labels_
metrics.silhouette_score(X_scaled, labels, metric='euclidean')
dbscan = DBSCAN(eps=2, min_samples=7)
start_time = time.time()
clusters= dbscan.fit_predict(pc_X)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
np.unique(clusters)
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    ax.scatter3D(pc_X[clusters==i,0], 
                 pc_X[clusters==i,1],
                 pc_X[clusters==i,2],
                 label=label)

plt.legend()
plt.show()
#evaluate model
labels = dbscan.labels_
metrics.silhouette_score(pc_X, labels, metric='euclidean')