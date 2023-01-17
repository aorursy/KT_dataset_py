# import libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes

import numpy as np # Import Numpy for data statistical analysis 

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns # Statistical data visualization



%matplotlib inline
# Import Cancer data drom the Sklearn library

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer
cancer.keys()
print(cancer['DESCR'])

print(cancer['target_names'])
print(cancer['feature_names'])

print(cancer['data'])
cancer['data'].shape
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()
df_cancer.tail()
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count") 
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

# Let's check the correlation between the variables 



plt.figure(figsize=(20,10)) 

sns.heatmap(df_cancer.corr(), annot=True) 


# Let's drop the target label coloumns

X = df_cancer.drop(['target'],axis=1)

y = df_cancer['target']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
print(X_train.shape)



print(X_test.shape)



print(y_train.shape)



print(y_test.shape)
from sklearn.svm import SVC 

from sklearn.metrics import classification_report, confusion_matrix



svc_model = SVC()

svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
print(classification_report(y_test, y_predict))



sns.heatmap(cm, annot=True)
sns.scatterplot(x= 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
min_train = X_train.min()



range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train



X_train_scaled
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
from sklearn.svm import SVC 

from sklearn.metrics import classification_report, confusion_matrix



svc_model = SVC()

svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)



print(classification_report(y_test,y_predict))

sns.heatmap(cm,annot=True,fmt="d")
from sklearn.model_selection import GridSearchCV



param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 



grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)



grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)



print(classification_report(y_test,grid_predictions))

sns.heatmap(cm, annot=True)