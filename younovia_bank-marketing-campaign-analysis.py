#import libraries

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

#import data

d1 = pd.read_csv('../input/bank-additional-full.csv',sep = ';', header = 0)



#we can first find out the types of each variables that decide 

#what preprocessing skills that we will need for each variables

d1.info()



#There are 21 variables in total and no missing values,but later 

#we can know that theres unknown category in several variables
d1.head()
d1.job.value_counts()
d1.marital.value_counts()
d1.education.value_counts()


d2 = d1

unknown = {"job": {"unknown": "admin."},

          "marital": {"unknown": "married"},

          "education": {"unknown": "university.degree"},

          "default": {"unknown": "no"},

          "housing": {"unknown": "yes"},

          "loan": {"unknown": "no"}}

d2.replace(unknown,inplace = True)

#replace unknown in each column to the most frequent in that column
d2.age.describe()
# I divide age starting rom 25% quantile and then add 20 to each categories using the 



def age(dataframe):

    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1

    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 52), 'age'] = 2

    dataframe.loc[(dataframe['age'] > 52) & (dataframe['age'] <= 72), 'age'] = 3

    dataframe.loc[(dataframe['age'] > 72) & (dataframe['age'] <= 98), 'age'] = 4

           

    return dataframe

age(d2).head()
#encode varables that already become dummy 

labelencoder_X = LabelEncoder()

d2.job = labelencoder_X.fit_transform(d2.job)

d2.marital = labelencoder_X.fit_transform(d2.marital)

d2.default = labelencoder_X.fit_transform(d2.default)

d2.housing = labelencoder_X.fit_transform(d2.housing)

d2.loan = labelencoder_X.fit_transform(d2.loan)

edu = {"illiterate" : 0,

       "basic.4y" : 1,

       "basic.6y" : 2,

       "basic.9y" : 3,

       "high.school" : 4,

       "professional.course" : 5,

       "university.degree" : 6}

d2['education'].replace(edu,inplace = True)



#Because I think education level has kind of ordinal, so i assign numbers to different level

d2.head()
d1.contact.value_counts()
d1.month.value_counts()
d1.day_of_week.value_counts()
d2.contact = labelencoder_X.fit_transform(d2.contact)

d2.month = labelencoder_X.fit_transform(d2.month)

d2.day_of_week = labelencoder_X.fit_transform(d2.day_of_week)
d2.head()
d1.poutcome.value_counts()

d2['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
#transform Y variable to dummy

d2.y.value_counts()

d2.y= labelencoder_X.fit_transform(d2.y)



d2.describe()
corr = d2.corr()

corr.style.background_gradient(cmap = 'coolwarm')

sns.boxplot(x = 'duration', data = d2, orient = 'v')

#Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and 

#should be discarded if the intention is to have a realistic predictive model.

#Therefore, we drop this variable
Y = d2.y

X = d2.drop('y',axis = 1)

X = X.drop('duration',axis = 1)

#train for 75%, test dataset is 25%

X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#############

#Logistic Regression



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)

logpred = logmodel.predict(X_test)

import sklearn.metrics as metrics

conlg = print(metrics.confusion_matrix(Y_test, logpred))

acclg = print(round(metrics.accuracy_score(Y_test, logpred), 4)*100)

##89.7

#since the dataset is relatviely imbalaced, we should look at ROC_AUC

problg = logmodel.predict_proba(X_test)

predslg = problg[:,1]

fprlg, tprlg, threshold = metrics.roc_curve(Y_test, predslg)

roc_auclg = metrics.auc(fprlg, tprlg)

plt.title('Receiver Operating Characteristic for Logistic Regression')

plt.plot(fprlg, tprlg, 'b', label = 'AUClg = %0.2f' % roc_auclg)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate lg')

plt.xlabel('False Positive Rate lg')

plt.show()

##0.79



#####

#Random Forest 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X_train, Y_train)

rfpred = rf.predict(X_test)

conrf = print(metrics.confusion_matrix(Y_test, rfpred))

accrf = print(round(metrics.accuracy_score(Y_test, rfpred), 4)*100)



probrf = rf.predict_proba(X_test)

predsrf = probrf[:,1]

fprrf, tprrf, thresholdrf = metrics.roc_curve(Y_test, predsrf)

roc_aucrf = metrics.auc(fprrf, tprrf)

plt.title('Receiver Operating Characteristic for Random Forest')

plt.plot(fprrf, tprrf, 'b', label = 'AUCrf = %0.2f' % roc_aucrf)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate rf')

plt.xlabel('False Positive Rate rf')

plt.show()

##0.77

#########

##XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, Y_train)

xgbpred = xgb.predict(X_test)



conxgb = print(metrics.confusion_matrix(Y_test, xgbpred))

accxgb = print(round(metrics.accuracy_score(Y_test, xgbpred), 4)*100)

##89.97

probxgb = xgb.predict_proba(X_test)

predsxgb = probxgb[:,1]

fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(Y_test, predsxgb)

roc_aucxgb = metrics.auc(fprxgb, tprxgb)





plt.title('Receiver Operating Characteristic for XGBoost')

plt.plot(fprxgb, tprxgb, 'b', label = 'AUCxgb = %0.2f' % roc_aucxgb)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate xgb')

plt.xlabel('False Positive Rate xgb')

plt.show()

##0.8


