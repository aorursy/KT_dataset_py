# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression,Ridge,ElasticNet,Lasso,SGDClassifier,LinearRegression

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics


from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings("ignore")



#Reading and sample examine of the Data
df = pd.read_csv('../input/Admission_Predict.csv') 
print(df.head())
print(df.columns)

pd.set_option('display.max_columns', 9)
print(df.describe())


#To Check Any Null Columns or Rows
print(df.isnull().sum())
#As there is no Null Value, no row or column is to be dropped on account of this

#'Serial No is only sequential and will not contribute to decisions so we can drop it
df.drop(columns=['Serial No.'],axis=1,inplace=True)

#Data Visualization

#Drawing histogram of each feature

df.hist()
#df['Chance of Admit '].plot(kind='density',subplots=True,sharex=False, figsize=(10,4),title="Density Curve")
plt.show()

#Drawing Density Plots of Selected Important Features
sampleFeatures=['GRE Score', 'TOEFL Score', 'CGPA']
print("Density Plots of Select Features ",sampleFeatures )

for feeture in sampleFeatures:
    sns.distplot(df[feeture], hist=False, rug=True,kde=True)
    plt.show()

#Drawing Violin plots against important feaures
print("Violin Plots of Important Features vs University Ranking")
for feeture in sampleFeatures:
    sns.catplot(y=feeture, x='University Rating', hue='Research', kind="violin", split=True, data=df)
    g = sns.catplot(y=feeture, x='University Rating', kind="violin", inner=None, data=df)
    sns.swarmplot(y=feeture, x='University Rating', color="k", size=3, data=df, ax=g.ax)

# boxplot on each feature split out by 'Research'
print("Box Plots of Important Features Researchwise")

for feeture in sampleFeatures:
    df.boxplot(column=feeture,by='Research',figsize=(70,70))
plt.show()
#The box plot shows that these features have high values by a difference of one or more quartiles for candidates having done Research
#Also some outliers are present in each of them


#Displaying Correlation between all pairs of features

pd.set_option('display.max_columns', 9)
print("Matrix of Correlation between each of two features:\n")
print(df.corr())


# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('Correlation Scatter Matrix')
plt.show()

#Showing Heatmap for visual interpretation to know whether some of the parameters are strongly correlated

plt.suptitle('Heatmap for showing Correlation between each feature, Lighter Colors show stronger Correlation ')

corrMatrix=df.corr()
sns.heatmap(corrMatrix,annot=True,linecolor='blue')

#It shows although some parameters like 'GRE Score', 'TOEFL Score', 'CGPA' are substantially correlated but are not highy correlated
#This is also corroborated by df.corr() values and scatter plot matrix. So nearly no chance to drop some features


#Ranking of features using e.g. RandomForestClassifier()
print(df.shape)
print(df.columns)

array = df.values
X = array[:,0:7]
Y = array[:,7]
#print(Y)
Y =np.where(Y >.8,1,0)

# feature extraction using RFE
from sklearn.feature_selection import RFE
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1) 
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print("\nNum of best Features: ", fit.n_features_)
print("Selected Features using RFE feature selector: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)

# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print("\nFeature importance score using ExtraTreesClassifier feature selector : \n", model.feature_importances_)

#Conclusion:
#We can see from the output of RFE that 'GRE Score', 'TOEFL Score', 'University Rating', 'CGPA' are top four features
#The output of ExtraTreesClassifier rankes them in the order of importance 


#Putting data on a uniform scale for better accuracy using MinMaxScaler

train_features=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']

#
from sklearn import preprocessing

min_max_normalizer=preprocessing.MinMaxScaler()
scaled_data=min_max_normalizer.fit_transform(df[train_features])
df_normalized=pd.DataFrame(scaled_data)

df_normalized.columns=train_features 
print("Normalized Training Features between 0 and 1:\n")
print(df_normalized.head(10))

#Using Linear Regression for Prediction

X_train,X_test,y_train,y_test = train_test_split(df_normalized[train_features],df['Chance of Admit '],random_state=42)
linear_regression=LinearRegression()

linear_regression.fit(X_train,y_train)

#Printing coefficients of linear_regression model
print("Each of coefficient of linear_regression model:\n")
print(pd.DataFrame(linear_regression.coef_,X_train.columns,columns=['Coefficient']))

y_pred=linear_regression.predict(X_test)
print(X_test.shape)
print(pd.DataFrame({'\nPrediction':y_pred,'Actual':y_test}))

print("\nError Rate with Linear Regression: ", metrics.mean_absolute_error(y_test,y_pred))


#scatter plot for predictions vs actual values
plt.suptitle('Scatter plot  for Predictions vs Actual values')
X_test1=range(len(y_test))
plt.scatter(X_test1,y_test,color='red', marker='<')
plt.scatter(X_test1,y_pred,color='yellow', marker='1')
plt.show()

#Computing cross-validated metrics on multiple models and finding the best accuracy

models = []

models.append(('LogR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('SGD', SGDClassifier(max_iter = 100)))
models.append(('Random Forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)))


# prepare configuration for cross validation test harness
seed = 7

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

y_train = np.where(y_train >.8,1,0)

print("Mean Accuracy and Mean Standard Daviation of Each Model:\n")

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Box Plot for Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.show()


# Random Forest has the best accuracy

random_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Random Forest  Accuracy : ", acc_random_forest)

