import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn import metrics #for model evalution

from sklearn.feature_selection import RFE,RFECV   #for feature selection

from sklearn.model_selection import train_test_split #for data splitting

pd.options.mode.chained_assignment = None  #hide any pandas warnings
data=pd.read_csv("../input/heart-disease-diagnosis/heart.csv")
data.head() #upper five observation
data.tail() # lower five observation
data.drop(index=[48,281],axis=0,inplace=True)  # blank value, so thats why i removed
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'slope', 'no_major_vessels', 'thalassemia', 'target']
data.head()
data.chest_pain_type.value_counts()
data['sex'][data['sex'] == 0] = 'female'

data['sex'][data['sex'] == 1] = 'male'



data['chest_pain_type'][data['chest_pain_type'] == 0] = 'typical angina'

data['chest_pain_type'][data['chest_pain_type'] == 1] = 'atypical angina'

data['chest_pain_type'][data['chest_pain_type'] == 2] = 'non-anginal pain'

data['chest_pain_type'][data['chest_pain_type'] == 3] = 'asymptomatic'



data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'

data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'

data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'



data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'



data['slope'][data['slope'] == 0] = 'upsloping'

data['slope'][data['slope'] == 1] = 'flat'

data['slope'][data['slope'] == 2] = 'downsloping'



data['thalassemia'][data['thalassemia'] == 1] = 'normal'

data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'

data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'
data.head()
data.info()       # check null value
data.age.skew()       # near to normal distribution
sns.distplot(data.age)
sns.distplot(data.resting_blood_pressure)
sns.distplot(data.cholesterol)     #positive skewed and some outlier also
sns.boxplot(data.cholesterol)    # upper outlier
sns.distplot(data.max_heart_rate_achieved)   # moderate skew to left side
g=sns.FacetGrid(data,row="sex",col="target")   #female cholestral level is very high with target 1 as compared to other

g.map(sns.distplot,"cholesterol")
data.target.value_counts()
sns.countplot(x="target", data=data )

plt.show()
sns.countplot(x='sex', data=data)

plt.xlabel('Sex')

plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart disease frequency ages wise')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart disease frequency sex wise')

plt.xlabel('Sex')

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(data.fasting_blood_sugar,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency By FBS')

plt.xlabel('FBS - (Fasting Blood Sugar )')

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
pd.crosstab(data.chest_pain_type,data.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])

plt.title('Heart disease frequency acc. to chest pain type')

plt.xlabel('chest_pain_type')

plt.ylabel('Frequency of Disease or Not')

plt.show()
data.dtypes
data=pd.get_dummies(data,drop_first=True)

data.head()       # Now,Let's see
data.shape
x=data.drop("target",axis=1).values    
y=data.target.values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0) #split the data
x_train.shape
x_test.shape
model = RandomForestClassifier(random_state=0)

model.fit(x_train, y_train)                     #fits the data into model
y_predict=model.predict(x_test)

y_predict                       #predicted value
confusion_matrix=metrics.confusion_matrix(y_test,y_predict)

confusion_matrix
classification_report=metrics.classification_report(y_test,y_predict)

print(classification_report)
accuracy=metrics.accuracy_score(y_test,y_predict)    # accuracy

accuracy
y_pred_prob=model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
metrics.auc(fpr,tpr)    #auc Area Under the Curve
Rfecv=RFECV(estimator=model)

Rfecv=Rfecv.fit(x_train,y_train)
X_train_df=data.drop("target",axis=1)    # because columns attribute not in array so thats why array convert into dataframe
list(zip(X_train_df.columns,Rfecv.support_,Rfecv.ranking_))
Rfecv.support_
X_train_df.columns[Rfecv.support_]
X_train=x_train    # create two new variable for new model

X_test=x_test
X_train
X_train=np.delete(x_train,obj=[10,11,16],axis=1)   # remove 10,11,16 columns because these are not important

X_test=np.delete(x_test,obj=[10,11,16],axis=1) 
model_new=RandomForestClassifier(random_state=0)
model_new.fit(X_train,y_train)      # fit model after removes cloumns with the help of RFECV
Y_predict=model_new.predict(X_test)

Y_predict
classification_report_new=metrics.classification_report(y_test,Y_predict)

print(classification_report_new)
print(classification_report)                          #nothing much has changed  same like older