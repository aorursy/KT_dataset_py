#Importing necessary elements
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
import matplotlib.pyplot as plt
#Loading the dataset into "df"
df=pd.read_csv('../input/heart-disease-uci/heart.csv')
#looking at the first five rows of the dataset.You can see more rows by adding parameter into the head():egdf.head(10)
#would show you first ten rows
df.head()
#Gives the basic information of the dataset
df.info()
#Gives the shape
df.shape
#Describes the dataset
df.describe()
import pandas_profiling 

pandas_profiling.ProfileReport(df)
#This function calls matplotlib.pyplot.hist() , on each series in the DataFrame, resulting in one histogram per column.
df.hist(bins=50,figsize=(20,15))
#heatmap
import seaborn as sns
plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()
X = df.drop(['target'],axis=1)
y = df['target']
#Importing train_test_split from sklearn.model_selection,as it will be used to split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
rfc.fit(X_train,y_train)
print("It's show time of the Accuracy: ",rfc.score(X_test,y_test))
#I wrote a function to check for the amount of contribution that a feature that has in the decision making(output)
def plot_feature_importance(rfc):
    plt.figure(figsize=(8,6))
    n_fetures=13
    plt.barh(range(n_fetures),rfc.feature_importances_,align='center')
    plt.yticks(np.arange(n_fetures),X)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.ylim(-1,n_fetures)
    
plot_feature_importance(rfc)
#Since fbs and restecg does not contribute much towards the decison making, i'll drop those two columns from the dataset
df2=df.drop(['fbs','restecg'],axis=1)
df2.head()
X = df2.drop(['target'],axis=1)
y = df2['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
rfc.fit(X_train,y_train)
print("It's show time of the Accuracy: ",rfc.score(X_test,y_test))
#printing the test dataset
print(X_test)
#printing the test's expected result
print(y_test)
#Let us make some prediction.(you can input the values from the above printed X_test and compare it's result with y_test)
print(rfc.predict([[55,1,0,132,353,132,1,1.2,1,1,3]]))
print(rfc.predict([[48,0,2,130,275,139,0,0.2,2,0,2]]))
print(rfc.predict([[56,1,1,120,236,178,0,0.8,2,0,2]]))
from sklearn.metrics import confusion_matrix
X_test = rfc.predict(X_test)
cm = confusion_matrix(y_test,X_test)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm,annot=True)
plt.show()