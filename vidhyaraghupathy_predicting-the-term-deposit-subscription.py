import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
myData = pd.read_csv('../input/bank-full.csv')

myData.head()
dataShape = myData.shape

print(dataShape)
dataTypes = myData.dtypes

print(dataTypes)
print(myData.isnull().values.any())
print(myData.isnull().sum())
print(myData.isnull())
myData.describe(include=[np.number])
for column in myData[['age','balance','day', 'duration', 'campaign', 'pdays', 'previous']]:

    val = column

    q1 = myData[val].quantile(0.25)

    q3 = myData[val].quantile(0.75)

    iqr = q3-q1

    fence_low  = q1-(1.5*iqr)

    fence_high = q3+(1.5*iqr)

    df_out = myData.loc[(myData[val] < fence_low) | (myData[val] > fence_high)]

    if df_out.empty:

        print('No Outliers in the ' + val + ' column of given dataset')

    else:

        print('There are Outliers in the ' + val + ' column of given dataset')



print(sns.boxplot(myData['age']))
print(sns.boxplot(myData['balance']))
print(sns.boxplot(myData['day']))
print(sns.boxplot(myData['duration']))
print(sns.boxplot(myData['campaign']))
print(sns.boxplot(myData['pdays']))
print(sns.boxplot(myData['previous']))
myData['job'] = myData['job'].astype('category')

myData['default'] = myData['default'].astype('category')

myData['marital'] = myData['marital'].astype('category')

myData['education'] = myData['education'].astype('category')

myData['housing'] = myData['housing'].astype('category')

myData['loan'] = myData['loan'].astype('category')

myData['contact'] = myData['contact'].astype('category')

myData['month'] = myData['month'].astype('category')

myData['poutcome'] = myData['poutcome'].astype('category')

myData['Target'] = myData['Target'].astype('category')

myData.info()
myData.groupby('Target').count()
plt.figure(figsize = (10,10))

sns.heatmap(myData.corr(), annot = True)
plt.figure(figsize = (10,8))

sns.scatterplot(myData['age'], myData['balance'], hue = myData['Target'])
myData['job'].value_counts()
plt.figure(figsize = (14,6))

sns.countplot(x = 'job', hue = 'Target', data = myData )
myData['marital'].value_counts()
sns.countplot(x = 'marital',hue = 'Target', data = myData)
myData['education'].value_counts()
sns.countplot(x = 'education', hue = 'Target', data = myData)
myData['default'].value_counts()
sns.countplot(x = 'default', hue = 'Target', data = myData)
myData['housing'].value_counts()
sns.countplot(x = 'housing', hue = 'Target', data = myData)
myData['loan'].value_counts()
sns.countplot(myData['loan'], hue = myData['Target'])
myData['contact'].value_counts()
sns.countplot(myData['contact'], hue = myData['Target'])
myData['day'].value_counts()
plt.figure(figsize = (8,6))

sns.countplot(myData['day'], hue = myData['Target'])
myData['month'].value_counts()
sns.countplot(myData['month'], hue =  myData['Target'])
myData['campaign'].value_counts()
plt.figure(figsize = (12,8))

sns.countplot(myData['campaign'], hue =  myData['Target'])
myData['previous'].value_counts()
plt.figure(figsize = (12,8))

sns.countplot(myData['previous'], hue =  myData['Target'])
myData['poutcome'].value_counts()
sns.countplot(myData['poutcome'], hue =  myData['Target'])
clientData = myData.iloc[:, 0:8]

clientData.head()
def age(clientData):

    clientData.loc[clientData['age'] <= 33, 'age'] = 1

    clientData.loc[(clientData['age'] > 33) & (clientData['age'] <= 48), 'age' ] = 2

    clientData.loc[(clientData['age'] > 48) & (clientData['age'] <= 70), 'age' ] = 3

    clientData.loc[(clientData['age'] > 70) & (clientData['age'] <= 95), 'age' ] = 4

    return clientData



age(clientData)
def balance(clientData):

    clientData.loc[clientData['balance'] <= 72, 'balance'] = 1

    clientData.loc[(clientData['balance'] > 72) & (clientData['balance'] <= 1428), 'balance' ] = 2

    clientData.loc[(clientData['balance'] > 1428) & (clientData['balance'] <= 3462), 'balance' ] = 3

    clientData.loc[(clientData['balance'] > 3462) & (clientData['balance'] <= 102127), 'balance' ] = 4

    return clientData



balance(clientData)
clientData = pd.get_dummies(data = clientData, columns = ['job'], prefix = ['job'], drop_first = True)

clientData = pd.get_dummies(data = clientData, columns = ['marital'], prefix = ['marital'], drop_first = True)

clientData = pd.get_dummies(data = clientData, columns = ['education'], prefix = ['education'], drop_first = True)

clientData = pd.get_dummies(data = clientData, columns = ['default'], prefix = ['default'], drop_first = True)

clientData = pd.get_dummies(data = clientData, columns = ['housing'], prefix = ['housing'], drop_first = True)

clientData = pd.get_dummies(data = clientData, columns = ['loan'], prefix = ['loan'], drop_first = True)

clientData.head()
clientData.shape
campaignData =myData.iloc[:, 8:]

campaignData.head()
def duration(campaignData):

    campaignData.loc[campaignData['duration'] <= 103, 'duration'] = 1

    campaignData.loc[(campaignData['duration'] > 103) & (campaignData['duration'] <= 319), 'duration' ] = 2

    campaignData.loc[(campaignData['duration'] > 319) & (campaignData['duration'] <= 643), 'duration' ] = 3

    campaignData.loc[(campaignData['duration'] > 643) & (campaignData['duration'] <= 4918), 'duration' ] = 4

    

    return campaignData



duration(campaignData)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

campaignData['month'] = label_encoder.fit_transform(campaignData['month'])

campaignData
plt.figure(figsize = (10,8))

sns.heatmap(campaignData.corr(), annot = True)
campaignData = pd.get_dummies(data = campaignData, columns = ['contact'], drop_first = True)

campaignData = pd.get_dummies(data = campaignData, columns = ['poutcome'], prefix = ['poutcome'], drop_first = True)

campaignData.head()
campaignData.shape
finData = pd.concat([clientData, campaignData], axis = 1)

finData.head()
from sklearn.model_selection import train_test_split



X = finData.drop('Target', axis = 1)

y = finData['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier() 

rf.fit(X, y) 

rf.score(X, y)

feature_importances = pd.DataFrame(rf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False) * 100

feature_importances
finData.columns
from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()

X_train = ss_X.fit_transform(X_train)

X_test = ss_X.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logpred = logreg.predict(X_test)



print('Train score: {}'.format(logreg.score(X_train, y_train) * 100))

print('Test score: {}'.format(logreg.score(X_test, y_test) * 100))

print(metrics.confusion_matrix(y_test, logpred))
accuracies = {}

acc_logreg = logreg.score(X_test, y_test) * 100

accuracies['Logistic Regression'] = acc_logreg

print("Logistic Regression Accuracy: {}".format(acc_logreg))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)



expected  = y_test

predicted = nb.predict(X_test)

print(metrics.classification_report(expected, predicted))

print('Total accuracy:', np.round(metrics.accuracy_score(expected, predicted),2))



print('Train score: {}'.format(nb.score(X_train, y_train) * 100))

print('Test score: {}'.format(nb.score(X_test, y_test) * 100))

print(metrics.confusion_matrix(expected, predicted))



acc_nb = nb.score(X_test, y_test) * 100

accuracies['Naive Bayes'] = acc_nb

print("Naive Bayes Accuracy: {}".format(acc_nb))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate



myList = list(range(1,10))

neighbors = list(filter(lambda x: x % 2 != 0, myList))

ac_scores = []
for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)
MSE = [1 - x for x in ac_scores]

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
knn = KNeighborsClassifier(n_neighbors = 9, weights = 'distance')

X_z = X.apply(zscore)

X_z.describe()
X = np.array(X_z)

y = np.array(y)
print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn.score(X_test, y_test)



print('Train score: {}'.format(knn.score(X_train, y_train) * 100))

print('Test score: {}'.format(knn.score(X_test, y_test) * 100))
cm_knn = print(metrics.confusion_matrix(y_test, y_pred))

cm_knn

acc_knn = knn.score(X_test, y_test) * 100

accuracies['KNN'] = acc_knn
from sklearn import svm

svm = svm.SVC(gamma = 0.025, C = 3)

svm.fit(X_train, y_train)
svm.score(X_test, y_test)
y_pred = svm.predict(X_test)



print('Train score: {}'.format(svm.score(X_train, y_train) * 100))

print('Test score: {}'.format(svm.score(X_test, y_test) * 100))
acc_svm = svm.score(X_test, y_test) * 100

accuracies['SVM'] = acc_svm

print(metrics.confusion_matrix(y_test, y_pred))
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



dt_entropy = DecisionTreeClassifier(criterion = 'entropy')

dt_entropy.fit(X_train, y_train)





print(dt_entropy.score(X_train, y_train))

print(dt_entropy.score(X_test, y_test))
clf_pruned = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 100, min_samples_leaf = 5)

clf_pruned.fit(X_train, y_train)
print(clf_pruned.score(X_train, y_train) * 100)

acc_DT = clf_pruned.score(X_test, y_test) *100

accuracies['DT'] = acc_DT

print(acc_DT)
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image 



xvar = finData.drop('Target', axis=1)

feature_cols = xvar.columns



dot_data = StringIO()



export_graphviz(clf_pruned, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['yes','no'])
xvar = finData.drop('Target', axis=1)

feature_cols = xvar.columns

feat_importance = clf_pruned.tree_.compute_feature_importances(normalize = False)



feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')

feat_imp.sort_values(by=0, ascending=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix



rfcl = RandomForestClassifier(n_estimators = 200)

rfcl.fit(X_train, y_train)
rfcl_predict = rfcl.predict(X_test)

acc_rf = accuracy_score(y_test, rfcl_predict) * 100

accuracies['RF'] = acc_rf

accuracies['RF']
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 22)

abcl = abcl.fit(X_train, y_train)



abcl_predict = abcl.predict(X_test)

acc_abcl = accuracy_score(y_test, abcl_predict) *100

accuracies['ADA'] = acc_abcl

accuracies['ADA']
from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(n_estimators = 200,max_samples= .7, bootstrap=True, oob_score=True, random_state = 22)

bgcl = bgcl.fit(X_train, y_train)



bgcl_predict = bgcl.predict(X_test)

acc_bgcl = accuracy_score(y_test, bgcl_predict) * 100



accuracies['Bagging'] = acc_bgcl

accuracies['Bagging']
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 200,learning_rate = 0.1, random_state = 22)

gbcl = gbcl.fit(X_train, y_train)



gbcl_predict = gbcl.predict(X_test)

acc_gbcl = accuracy_score(y_test, gbcl_predict) * 100

accuracies['GBoost'] = acc_gbcl

accuracies['GBoost']
plt.figure(figsize = (15,5))

plt.yticks(np.arange(0,100,10))

sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Naive Bayes','K-Nearest Neighbors', 'Support Vector Machines', 

              'Decision Tree','Random Forest', 'Ada Boost', 'Bagging','Gradient Boost'],

    

    'Score': [acc_logreg, acc_nb, acc_knn, acc_svm, acc_DT, acc_rf, acc_abcl, acc_bgcl, acc_gbcl]

    })



models.sort_values(by='Score', ascending=False)
y_cm_lr = logreg.predict(X_test)

y_cm_nb = nb.predict(X_test)

y_cm_knn = knn.predict(X_test)

y_cm_svm = svm.predict(X_test)



y_cm_dt = clf_pruned.predict(X_test)

y_cm_rfcl = rfcl.predict(X_test)

y_cm_abcl = abcl.predict(X_test)

y_cm_bgcl = bgcl.predict(X_test)

y_cm_gbcl = gbcl.predict(X_test)



from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_cm_lr)

cm_knn = confusion_matrix(y_test, y_cm_knn)

cm_svm = confusion_matrix(y_test, y_cm_svm)

cm_nb = confusion_matrix(y_test, y_cm_nb)



cm_dt = confusion_matrix(y_test, y_cm_dt)

cm_rf = confusion_matrix(y_test, y_cm_rfcl)

cm_abcl = confusion_matrix(y_test, y_cm_abcl)

cm_bgcl = confusion_matrix(y_test, y_cm_bgcl)

cm_gbcl = confusion_matrix(y_test, y_cm_gbcl)
plt.figure(figsize = (15,15))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplots_adjust(wspace = 0.8, hspace = 0.8)



plt.subplot(3,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 12})



plt.subplot(3,3,2)

plt.title("KNN Confusion Matrix")

sns.heatmap(cm_knn, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,3)

plt.title("NB Confusion Matrix")

sns.heatmap(cm_nb, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,4)

plt.title("SVM Confusion Matrix")

sns.heatmap(cm_svm, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,5)

plt.title("Decision Tree Confusion Matrix")

sns.heatmap(cm_dt, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,6)

plt.title("Random Confusion Matrix")

sns.heatmap(cm_rf, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,7)

plt.title("Ada Boost Confusion Matrix")

sns.heatmap(cm_abcl, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,8)

plt.title("Bagging Confusion Matrix")

sns.heatmap(cm_bgcl, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(3,3,9)

plt.title("Gradient Boosting Confusion Matrix")

sns.heatmap(cm_gbcl, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})