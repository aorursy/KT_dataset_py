import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df=pd.read_csv("../input/hr-analytics-case-study/general_data.csv")

df
df.columns
df.head()
df.tail()
df.describe()
df.corr()
print(df.isnull().any())

print(df.isnull().any().any())
#df.drop(['EmployeeCount','EmployeeID','StandardHours','Over18','NumCompaniesWorked','TotalWorkingYears'],axis=1, inplace = True)

plt.figure(figsize=(50,50)) #plt is the object of matplot lib and .figure() is used to show or change properties of graphs

sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False) #heatmaps are matrix plots which can visualize data in 2D

plt.show()
df.fillna(0,inplace=True)
df.drop(['EmployeeCount','EmployeeID','StandardHours','Over18','NumCompaniesWorked','TotalWorkingYears'],axis=1, inplace = True)
df.columns
#df['NumCompaniesWorked'].value_counts()
df.columns
df.head()
df.tail()
#df.drop(["EmployeeCount","EmployeeID","StandardHours"],1,inplace= True)
df_corr = df[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',

       'MaritalStatus', 'MonthlyIncome',

       'PercentSalaryHike', 'StockOptionLevel',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']]
corr=df_corr.corr()

corr
f,ax = plt.subplots(figsize=(16, 7))

sns.heatmap(corr, annot=True)

plt.show()
print(round(df['Attrition'].value_counts(normalize = True),2))

sns.countplot(x='Attrition',data=df)

sns.pairplot(df[['Age', 'Attrition', 'BusinessTravel', 'DistanceFromHome', 'EducationField', 'Gender', 'JobLevel', 'JobRole',

       'MaritalStatus', 'MonthlyIncome',

       'PercentSalaryHike', 'StockOptionLevel',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']],hue = 'Attrition')
sns.countplot(x = "Attrition",data=df,hue="Gender")

#plt.scatter(df['Attrition'], data=df,hue="Gender")

#sns.catplot(x="Attrition", y=df, data=df)

#sns.boxplot(x="Attrition", y="Gender", data=df)
sns.countplot(x = "Attrition",data=df,hue="JobLevel")
sns.countplot(x = "Attrition",data=df,hue="MaritalStatus")
sns.countplot(x = "Attrition",data=df)

plt.show()
sns.pairplot(df[['Age','MonthlyIncome','DistanceFromHome','Gender']],hue = 'Gender',hue_order=['Male','Female'], palette={'Male':'black','Female':'yellow'},plot_kws={'alpha':0.1},height=4)
df.isnull().any()
print(df['BusinessTravel'].unique())

print(df['Department'].unique())

print(df['EducationField'].unique())

print(df['Gender'].unique())

print(df['JobRole'].unique())

print(df['MaritalStatus'].unique())
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['BusinessTravel'] = labelEncoder_X.fit_transform(df['BusinessTravel'])

df['Department'] = labelEncoder_X.fit_transform(df['Department'])

df['EducationField'] = labelEncoder_X.fit_transform(df['EducationField'])

df['Gender'] = labelEncoder_X.fit_transform(df['Gender'])

df['JobRole'] = labelEncoder_X.fit_transform(df['JobRole'])

df['MaritalStatus'] = labelEncoder_X.fit_transform(df['MaritalStatus'])
from sklearn.preprocessing import LabelEncoder

label_encoder_y=LabelEncoder()

df['Attrition']=label_encoder_y.fit_transform(df['Attrition'])
f,ax = plt.subplots(figsize=(16, 7))

sns.heatmap(df.corr(), annot=True)

plt.show()
df.corr()
df.isnull().any().any()
y=df['Attrition']

x=df.drop('Attrition',axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
log_score = cross_val_score(estimator=LogisticRegression(), X=X_test, y=y_test, cv=5)  
plt.plot(log_score)
log_score
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = DecisionTreeClassifier(random_state=42)

#clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=10)

bag_clf.fit(X_train, y_train)

bag_clf.predict(X_test)
def kpi_metrics(clf, X_train, y_train, X_test, y_test, train=True):

    '''

    print the accuracy score, classification report and confusion matrix of classifier

    '''

    if train:

        '''

        training performance

        '''

        print("Train Result:\n")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        return pd.DataFrame(res)

        

    elif train==False:

        '''

        test performance

        '''

        print("Test Result:\n")        

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))

        

        

        res = cross_val_score(clf, X_test, y_test, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        return pd.DataFrame(res)

        
import numpy as np

kpi_metrics(bag_clf, X_train, y_train, X_test, y_test, train=True)
kpi_metrics(bag_clf, X_train, y_train, X_test, y_test, train=False)