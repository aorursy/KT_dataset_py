#import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#import from sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
#reading data from csv using pandas
data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head(10)
data.shape
#to see how many rows and columns we have in table
data.info()
sns.distplot(data["target"])
data["target"].value_counts()
#too see data having null values or not#in way to see any missing values.
print("Data Sum of Null Values\n")
data.isnull().sum()

null = data.isnull()
null
#the seaborn heatmap shows the clear view off entire dataset that theres no null
sns.heatmap(null)
sns.distplot(data["age"], color= "red" )
plt.title("ploting of age feature")
#visulize data as per age
#to find minimum and maximum ages
minAge =min(data["age"])
maxAge =max(data["age"])
avgAge = (data["age"]).mean()
print("Minimum age:",minAge)
print("Maximum age:",maxAge)
print("Average age:",avgAge)
#we can also understand data into age catagries 
Young_age = data[(data.age>=29)&(data.age<40)]
Meddle_age = data[(data.age>=40)&(data.age<55)]
Elderly_age = data[(data.age>55)]
print("Young age people:",len(Young_age))
print("Meddle age people:",len(Meddle_age)) 
print("Elderly age people",len(Elderly_age)) 
sns.barplot(x = ["Young ages","Meddle Ages","Elderly ages"], y =[len(Young_age),len(Meddle_age),len(Elderly_age)], color = "red")
f,ax=plt.subplots(1,2,figsize=(10,6))
data["target"].value_counts().plot.pie(ax = ax[0],autopct='%1.1f%%')
sns.countplot('target',data=data,ax=ax[1],order=data['sex'].value_counts().index)
f,ax=plt.subplots(1,2,figsize=(10,6))
data["sex"].value_counts().plot.pie(ax = ax[0],autopct='%1.1f%%')
sns.countplot('sex',data=data,ax=ax[1],order=data['sex'].value_counts().index)
plt.title("heart disease by sex")
#cheast pain types
f,ax=plt.subplots(1,2,figsize=(10,6))
data["cp"].value_counts().plot.pie(ax = ax[0],autopct='%1.1f%%',explode = [0,0.05,0.05,0.05])
sns.countplot('cp',data=data,ax=ax[1],order=data['sex'].value_counts().index)
plt.title("by type of chest pain")
#plot the target variable
#with the displot visulization we can see the average age of people suuffering from heart disease
sns.countplot(data["target"])
#where 0 shows that no heart attack and 1 shows that having an heart attack
fig, axes = plt.subplots( nrows=5, ncols=3, figsize=(15,40) )
plt.subplots_adjust( wspace=0.20, hspace=0.20, top=0.97 )
plt.suptitle("Heart Disease Dataset", fontsize=20)

axes[0,0].hist(data.age)
axes[0,0].set_xlabel("age(years)")
axes[0,0].set_ylabel("age(Number of Patients)")

axes[0,1].hist(data.sex)
axes[0,1].set_xlabel("Sex(male = 1,female = 0)")
axes[0,1].set_ylabel("Sex(Number of Patients)")

axes[0,2].hist(data.cp)
axes[0,2].set_xlabel("cp(type of chest pain)")
axes[0,2].set_ylabel("cp(Number of Patients)")

axes[1,0].hist(data.trestbps)
axes[1,0].set_xlabel("trestbps(rest blood pressure)")
axes[1,0].set_ylabel("trestbp(Number of Patients)")

axes[1,1].hist(data.chol)
axes[1,1].set_xlabel("chol(cholestrol level)")
axes[1,1].set_ylabel("chol(Number of Patients)")


axes[1,2].hist(data.fbs)
axes[1,2].set_xlabel("fbs(fasting blood sugure)")
axes[1,2].set_ylabel("fbs(Number of Patients)")


axes[2,0].hist(data.restecg)
axes[2,0].set_xlabel("restecg(Resting electrocardiology)")
axes[2,0].set_ylabel("restecg(Number of Patients)")

axes[2,1].hist(data.thalach)
axes[2,1].set_xlabel("thalach(maximum heart rate achieved)")
axes[2,1].set_ylabel("thalach(Number of Patients)")

axes[2,2].hist(data.exang)
axes[2,2].set_xlabel("exang(Exercise Inducing Angina)")
axes[2,2].set_ylabel("exang(Number of Patients)")

axes[3,0].hist(data.oldpeak)
axes[3,0].set_xlabel("oldpeak(Exercise Include Depression)")
axes[3,0].set_ylabel("oldpeak(Number of Patients)")

axes[3,1].hist(data.slope)
axes[3,1].set_xlabel("slope(slope of peak Exerice ST segment)")
axes[3,1].set_ylabel("slope(Number of Patients)")

axes[3,2].hist(data.ca)
axes[3,2].set_xlabel("ca(major vessies colored by FLurosocopy)")
axes[3,2].set_ylabel("ca(Number of Patients)")

axes[4,0].hist(data.thal)
axes[4,0].set_xlabel("thal")
axes[4,0].set_ylabel("thal(Number of Patients)")

axes[4,1].hist(data.target)
axes[4,2].axis("off")

#plot an correlation using seaborn 
corr = data.corr()
corr
#correlation between features
plt.figure(figsize =(20,15))
sns.heatmap(corr,annot= True)
#by analysing the positive and negative values we can understand the correlation between
# how positivalues and negativevalues have affect on each other in  feature.
#select the dependent and independent fetaures
#the target we have is independent features
X = data.drop("target",axis = 1)
#dependent feature 
X
#independent feature
y = data["target"]
y
#now we have we have to create both training data and independent 
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size = 0.2,random_state = 0)
#the model we are imported #from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
#fit the training parameter we created
random_forest.fit(X_train,y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train,y_train)
print(classification_report(y_test,Y_prediction))
print(confusion_matrix(y_test,Y_prediction))
#we can plot the confusion matrix in heatmap
sns.heatmap(confusion_matrix(y_test,Y_prediction),annot= True)
#feature importance in random forest
feat_importances = pd.Series(random_forest.feature_importances_,index= X.columns) 
feat_importances.nlargest(15).plot(kind = "barh")
from sklearn.preprocessing import StandardScaler
#giving dependent variable
scaler = StandardScaler()
scaler.fit(X)
data_scaled = scaler.transform(X)
data.columns
final_data = pd.DataFrame(data_scaled,columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
#not adding independent feature
final_data.head()
#just for understanding 
data.target
#the test  data
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
#have to predict
knnpredict = knn.predict(X_test)
print(classification_report(y_test,knnpredict))
print(confusion_matrix(y_test,knnpredict))
sns.heatmap(confusion_matrix(y_test,knnpredict),annot = True)
#using logistic regression with sciket learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import accuracy_score
from sklearn import metrics
model = LogisticRegression( fit_intercept=True,penalty='l2',dual=False,C=1.0)
model.fit(X_train,y_train)
modelpredict = model.predict(X_test)
print(classification_report(y_test,modelpredict))
print(confusion_matrix(y_test,modelpredict))