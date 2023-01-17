# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline


import seaborn as sns
sns.set_style('whitegrid')

from sklearn.impute import SimpleImputer as Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv") #at first we import the dataset
df.head()
print (df.shape)
df.isnull().sum() #so we can consider that we don't have null values, so we can go fast through dataset
df.info()
df.describe()
df.columns
df.rename(index=str, columns={'gender':'Gender', 
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                               inplace=True)
df.columns #here we want to make the dataset neat as some words had capital and small alphabets together
df.dtypes
for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)
print("Class Unique Values : ", df["Class"].unique())
print("Topic Unique Values : ", df["Topic"].unique())
print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())
print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())
print("Relation Unique Values : ", df["Relation"].unique())
print("SectionID Unique Values : ", df["SectionID"].unique())
print("Gender Unique Values : ", df["Gender"].unique())
df.groupby('Class').size()
# plot missing data:
df.isnull().sum().plot(kind='bar')
# Add a title and show the plot.
plt.title('Number of Missing Values Per Column')
# Create tick mark labels on the Y axis and rotate them.
plt.xticks(rotation = 45)
# Create X axis label.
plt.xlabel("Columns")
# Create Y axis label.
plt.ylabel("NaN Values");
# drop any duplicate value if exists
df.drop_duplicates()
df.shape
# so no duplicate value, no missing

#we want to visualize how many times students raised their hand in a time=x
plt.subplots(figsize=(20, 8))
df["RaisedHands"].value_counts().sort_index().plot.bar()
plt.title("Number of times vs number of students raised their hands in a certain time", fontsize=20)
plt.xlabel("Number of times", fontsize=14)
plt.ylabel("Number of student", fontsize=14)
plt.show()
df.columns
#we want to visualize how many times students visited certin resources in a particular time
plt.subplots(figsize=(20, 8))
plt.title("Number of times vs number of student visted resource in a certain time", fontsize=20)
df["VisitedResources"].value_counts().sort_index().plot.bar()
plt.xlabel("Number  of times, student visted resource", fontsize=14)
plt.ylabel("Number  of student, on particular visit", fontsize=14)
plt.show()
# we want to analysis the relation between gender and performance
df_edu=df.groupby(['Gender','Class']).size()

df_edu
df_edu.unstack().plot(kind='bar', stacked=True)
#From this bar chart we visualize that male students are more on "medium" and "lower" category while girls show better performance
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
sns.countplot(x='Topic', hue='Gender', data=df, ax=axis1)
sns.countplot(x='Nationality', hue='Gender', data=df, ax=axis2)
# We see the gender wise performance at country level and by each subject

# According to those charts Although it shows girls have better performance but when we analyzie the topic diagram, we see that, girls took less technical subjects

# As we see in the Nationality diagram, there is Gender disparity at a country level so maybe its another reason that girls  participated less than boys



# illustrate the percentage of participants from each country
df.Nationality.value_counts(normalize= True).plot(kind='bar')
# illustrate the percentage of participants according to their education level
df.StageID.value_counts(normalize=True).plot(kind='bar')
fig, axarr  = plt.subplots(2,2,figsize=(10,10))
sns.barplot(x='Class', y='VisitedResources', data=df, order=['L','M','H'], ax=axarr[0,0])
sns.barplot(x='Class', y='AnnouncementsView', data=df, order=['L','M','H'], ax=axarr[0,1])
sns.barplot(x='Class', y='RaisedHands', data=df, order=['L','M','H'], ax=axarr[1,0])
sns.barplot(x='Class', y='Discussion', data=df, order=['L','M','H'], ax=axarr[1,1])
#As expected, those that participated more (higher counts in Discussion, raisedhands, AnnouncementViews, RaisedHands), performed better.
#that thing about correlation and causation

fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
sns.pointplot(x='Semester', y='VisitedResources', hue='Gender', data=df, ax=axis1)
sns.pointplot(x='Semester', y='AnnouncementsView', hue='Gender', data=df, ax=axis2)
#As we see both visiting resources and viewing announcements, students were more vigilant in the second semester, perhaps that last minute need to boost your final grade.
#Create a new dataframe for processing data in decision tree.
df_dt=df.copy()
df_dt.head(10)
replace_map = {'Gender': {'F': 1, 'M': 2}}
replace_map1 = {'Nationality': {'Egypt': 1, 'Iran': 2, 'Iraq': 3, 'Jordan': 4, 'KW':5,'Lybia': 6, 'Morocco': 7, 'Palestine': 8 , 'SaudiArabia': 9 , 'Syria': 10,'Tunis': 11,'USA': 12,'lebanon': 13,'venzuela': 14}}
replace_map2 = {'PlaceofBirth': {'Egypt': 1, 'Iran': 2, 'Iraq': 3, 'Jordan': 4, 'KuwaIT':5,'Lybia': 6, 'Morocco': 7, 'Palestine': 8 , 'SaudiArabia': 9 , 'Syria': 10,'Tunis': 11,'USA': 12,'lebanon': 13,'venzuela': 14}}
replace_map3 = {'StageID': {'HighSchool': 1, 'MiddleSchool': 2, 'lowerlevel': 3}}
replace_map4 = {'GradeID': {'G-02':1,'G-04':2, 'G-05':3,'G-06':4,'G-07':5,'G-08':6,'G-09':7,'G-10':8,'G-11':9,'G-12':10}}
replace_map5 = {'SectionID': {'A': 1, 'B': 2, 'C': 3}}
replace_map6 = {'Topic': {'Arabic': 1, 'Biology': 2, 'Chemistry': 3, 'English': 4, 'French':5,'Geology': 6, 'History': 7, 'IT': 8 , 'Math': 9 , 'Quran': 10,'Science': 11,'Spanish': 12}}
replace_map7 = {'Semester': {'F': 1, 'S': 2}}
replace_map8 = {'Relation': {'Father': 1, 'Mum': 2}}
replace_map9 = {'ParentAnsweringSurvey': {'Yes': 1, 'No': 2}}
replace_map10 = {'ParentschoolSatisfaction': {'Bad': 1, 'Good': 2}}
replace_map11 = {'StudentAbsenceDays': {'Above-7': 1, 'Under-7': 2}}
replace_map12 = {'Class': {'M': 1, 'L': 2,'H':3}}
df_dt.replace(replace_map,inplace=True)
df_dt.replace(replace_map1,inplace=True)
df_dt.replace(replace_map2,inplace=True)
df_dt.replace(replace_map3,inplace=True)
df_dt.replace(replace_map4,inplace=True)
df_dt.replace(replace_map5,inplace=True)
df_dt.replace(replace_map6,inplace=True)
df_dt.replace(replace_map7,inplace=True)
df_dt.replace(replace_map8,inplace=True)
df_dt.replace(replace_map9,inplace=True)
df_dt.replace(replace_map10,inplace=True)
df_dt.replace(replace_map11,inplace=True)
df_dt.replace(replace_map12,inplace=True)


df_dt.head()
# Great now lets start decision tree learning

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
feature_columns=['Gender','Nationality','PlaceofBirth','StageID','SectionID','Topic','Semester','Relation','RaisedHands','VisitedResources','AnnouncementsView','Discussion','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']
X=df_dt[feature_columns]
y=df_dt.Class
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)
indextoCheckDecisionTree=y_test.index
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# now lets test accuracy of our model
Accuracy_dt=metrics.accuracy_score(y_test,y_pred)
print("Accuracy : ",Accuracy_dt)
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("xAPI-Edu")
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=feature_columns,
class_names=['Low-level','Middle-level','High-level'],
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
# Work on modifying criterion to see the accuracy.
clf=DecisionTreeClassifier(criterion="entropy")
clf=clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
Accuracy_dt=metrics.accuracy_score(y_test,y_pred)
print("Accuracy : ",Accuracy_dt)

# The accuracy decreased by a certain margin. so entropy is not a good criterion. Now lets tune parameter of max_depth
clf=DecisionTreeClassifier(criterion="entropy",max_depth=13)
clf=clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
Accuracy_dt=metrics.accuracy_score(y_test,y_pred)
print("Accuracy : ",Accuracy_dt)
#By combining entropy and max_depth accuracy increases to 0.73 which is very good for this model

# Now lets draw our modified decision tree
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("xAPI-Edu2")
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=feature_columns,
class_names=['Low-level','Middle-level','High-level'],
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
# we draw the classifier for the decision tree labels and show the number of TP,TN,FP and FN. so We use Confusion metrix
#to calculate the TP,TN,FP,FN
from sklearn.model_selection  import cross_val_score
from sklearn.metrics import classification_report
print("Confusion Metrix",metrics.confusion_matrix(y_test,y_pred))
cnf=metrics.confusion_matrix(y_test,y_pred)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report, accuracy_score

target_names = ['Low-level', 'Middle-level', 'High-level']
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.naive_bayes import GaussianNB
df_naive = df_dt.copy()
feature_columns=['Gender','Nationality','PlaceofBirth','StageID','SectionID','Topic','Semester','Relation','RaisedHands','VisitedResources','AnnouncementsView','Discussion','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']
X=df_naive[feature_columns]
y=df_naive.Class
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=109) # 70 % training and 30 % testing

gnb = GaussianNB()
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Lets try with some different split for data same as decision trees
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1) 
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
Accuracy_naive=metrics.accuracy_score(y_test,y_pred)
print("Accuracy : ",Accuracy_naive)
# To conclude on the same split , naive bayes has a better accuracy.
from sklearn.model_selection  import cross_val_score
from sklearn.metrics import classification_report
print("Confusion Metrix",metrics.confusion_matrix(y_test,y_pred))
cnf=metrics.confusion_matrix(y_test,y_pred)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report, accuracy_score

target_names = ['Low-level', 'Middle-level', 'High-level']
print(classification_report(y_test, y_pred, target_names=target_names))
#Lets use Randomn Forest Now

from sklearn.ensemble import RandomForestClassifier
df_rf = df_naive.copy()
feature_columns=['Gender','Nationality','PlaceofBirth','StageID','SectionID','Topic','Semester','Relation','RaisedHands','VisitedResources','AnnouncementsView','Discussion','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']
X=df_rf[feature_columns]
y=df_rf.Class
# Lets try with some different split for data same as decision trees and naive bayes
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1) 
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
Accuracy_rf = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", Accuracy_rf)
#Can we improve this ?
clf=RandomForestClassifier(bootstrap=True,criterion='gini',n_estimators=300)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
Accuracy_rf = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", Accuracy_rf)
# Lets calculate feature importance using Randomn Forest Classifier
dn = {'features':feature_columns,'score':clf.feature_importances_}
df = pd.DataFrame.from_dict(data=dn).sort_values(by='score',ascending=False)
      
plot= sns.barplot(x='score',y='features',data=df,orient='h')
plot.set(xlabel="Score",yLabel="features",title='Feature Importance of Randomn Forest Classifier')
plt.setp(plot.get_xtickLabels(), rotation = 90)
plt.show()
# So finally by the metric of accuracy our results are : 

print("Accuracy of decision Tree is : ", Accuracy_dt * 100,"%")
print("Accuracy of Naive Bayes is  : ", Accuracy_naive * 100, "%")
print("Accuracy of RandomnForest is : ", Accuracy_rf * 100 ,"%")