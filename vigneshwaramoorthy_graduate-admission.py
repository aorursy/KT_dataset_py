#Importing the necessary packages



import collections



import matplotlib.pyplot as plt



import numpy as np

import pandas as pd



import scipy

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, plot_roc_curve, accuracy_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings('ignore')
#Reading the datasets

data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

data.head()
data.describe()
data.isnull().sum()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
#Removing the serial number column as it adds no correlation to any columns

data = data.drop(columns = ["Serial No."])



#The column "Chance of Admit" has a trailing space which is removed

data = data.rename(columns={"Chance of Admit ": "Chance of Admit"})



data.head()
plt.hist(data["Chance of Admit"])

plt.xlabel("Chance of Admit")

plt.ylabel("Count")

plt.show()
sns.pairplot(data)
sns.kdeplot(data["Chance of Admit"], data["GRE Score"], cmap="Blues", shade=True, shade_lowest=False)
sns.kdeplot(data["Chance of Admit"], data["University Rating"], cmap="Blues", shade=True, shade_lowest=False)
sns.kdeplot(data["GRE Score"], data["University Rating"], cmap="Blues", shade=True, shade_lowest=False)
sns.scatterplot(data["GRE Score"], data["University Rating"])
collections.Counter([i-i%0.1+0.1 for i in data["Chance of Admit"]])
data['Label'] = np.where(data["Chance of Admit"] <= 0.72, 0, 1)

print(data['Label'].value_counts())

data.sample(10)
#Checking feature importance with DTree classifier

# define the model

model = DecisionTreeClassifier()



x = data.drop(columns = ['Chance of Admit', 'Label'])

y = data['Label']



# fit the model

model.fit(x, y)



# get importance

importance = model.feature_importances_



# summarize feature importance

for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v))



feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nsmallest(7).plot(kind='barh')
x_train, x_test, y_train, y_test = x[:400], x[400:], y[:400], y[400:]

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
def plot_roc(false_positive_rate, true_positive_rate, roc_auc):

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
parameters = [

    {

        'penalty' : ['l1', 'l2', 'elasticnet'],

        'C' : [0.1, 0.4, 0.5],

        'random_state' : [0]

    }

]



gscv = GridSearchCV(LogisticRegression(),parameters,scoring='accuracy')

gscv.fit(x_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_train), y_train))

print(confusion_matrix(gscv.predict(x_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_test), y_test))

print(confusion_matrix(gscv.predict(x_test), y_test))



#Crossvalidation:

cvs = cross_val_score(estimator = LogisticRegression(), 

                      X = x_train, y = y_train, cv = 12)



print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
lr = LogisticRegression(C= 0.1, penalty= 'l2', random_state= 0)

lr.fit(x_train,y_train)



y_pred = lr.predict(x_test)

y_proba=lr.predict_proba(x_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm=confusion_matrix(y_test,y_pred)

print(cm)
parameters = [

    {

        'criterion' : ['gini', 'entropy'],

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]



gscv = GridSearchCV(DecisionTreeClassifier(),parameters,scoring='accuracy')

gscv.fit(x_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_train), y_train))

print(confusion_matrix(gscv.predict(x_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_test), y_test))

print(confusion_matrix(gscv.predict(x_test), y_test))



#Crossvalidation:

cvs = cross_val_score(estimator = DecisionTreeClassifier(), 

                      X = x_train, y = y_train, cv = 12)



print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 3, min_samples_split= 10, 

                            random_state= 0)

dt.fit(x_train,y_train)



y_pred = dt.predict(x_test)

y_proba=dt.predict_proba(x_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm=confusion_matrix(y_test,y_pred)

print(cm)
parameters = [

    {

        'n_estimators': np.arange(10, 40, 5),

        'criterion' : ['gini', 'entropy'],

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]



gscv = GridSearchCV(RandomForestClassifier(),parameters,scoring='accuracy')

gscv.fit(x_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_train), y_train))

print(confusion_matrix(gscv.predict(x_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_test), y_test))

print(confusion_matrix(gscv.predict(x_test), y_test))



#Crossvalidation:

cvs = cross_val_score(estimator = RandomForestClassifier(), 

                      X = x_train, y = y_train, cv = 12)



print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
rf = RandomForestClassifier(criterion= 'gini', max_depth= 5, 

                            min_samples_split= 10, n_estimators= 15, 

                            random_state= 0)

rf.fit(x_train,y_train)



y_pred = rf.predict(x_test)

y_proba=rf.predict_proba(x_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm=confusion_matrix(y_test,y_pred)

print(cm)
parameters = [

    {

        'learning_rate': [0.01, 0.02, 0.002],

        'n_estimators' : np.arange(10, 100, 5),

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]



gscv = GridSearchCV(GradientBoostingClassifier(),parameters,scoring='accuracy')

gscv.fit(x_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_train), y_train))

print(confusion_matrix(gscv.predict(x_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(x_test), y_test))

print(confusion_matrix(gscv.predict(x_test), y_test))



#Crossvalidation:

cvs = cross_val_score(estimator = GradientBoostingClassifier(), 

                      X = x_train, y = y_train, cv = 12)



print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
gbm = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 3, 

                                 min_samples_split= 10, n_estimators= 80, 

                                 random_state= 0)

gbm.fit(x_train,y_train)



y_pred = gbm.predict(x_test)

y_proba = gbm.predict_proba(x_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm=confusion_matrix(y_test,y_pred)

print(cm)
#for submission using the random forest

y_proba=rf.predict(x_test)

np.sqrt(mean_squared_error(y_proba, y_test))