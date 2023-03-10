import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection



from sklearn import metrics

from sklearn.metrics import accuracy_score



sns.set(style='ticks')

%matplotlib inline
import os

print(os.listdir("../input/"))
bank=pd.read_csv("../input/Bank_Personal_Loan_Modelling.csv")

bank.head()
bank.columns
"""• ID : Customer ID

• Age : Customer's age in completed years

• Experience : #years of professional experience

• Income : Annual income of the customer ($000)

• ZIP Code : Home Address ZIP code.

• Family : Family size of the customer

• CCAvg : Avg. spending on credit cards per month ($000)

• Education : Education Level. 1: Undergrad; 2: Graduate; 3:

Advanced/Professional

• Mortgage : Value of house mortgage if any. ($000)

• Personal Loan : Did this customer accept the personal loan offered in the last campaign?

• Securities Account : Does the customer have a securities account with the bank?

• CD Account : Does the customer have a certificate of deposit (CD) account with the bank?

• Online : Does the customer use internet banking facilities?

• Credit card : Does the customer use a credit card issued byUniversalBank?

"""
#1.Age

bank.apply(lambda x : sum(x.isnull()))

bank.apply(lambda x: len(x.unique()))

bank.describe().transpose()
bank.isnull().values.any()
bank.describe(include="all").transpose()

plt.figure(figsize=(18,5))

sns.set_color_codes()

sns.distplot(bank["Age"])

plt.figure(figsize=(16,4))

sns.set_color_codes()

sns.countplot(bank["Age"])

plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.barplot(bank["Age"],bank["Personal Loan"])

plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.boxplot(y=bank["Age"],x=bank["Personal Loan"])

plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.violinplot(y=bank["Age"],x=bank["Personal Loan"])

plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.distplot(bank["Experience"])





plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.distplot(bank["Income"])





plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.distplot(bank["Family"])





plt.figure(figsize=(12,4))

sns.set_color_codes()

sns.distplot(bank["CCAvg"])
sns.pairplot(bank.iloc[:,1:], hue='Personal Loan')
colormap = plt.cm.viridis # Color range to be used in heatmap

plt.figure(figsize=(15,15))

plt.title('Bank Correlation of attributes', y=1.05, size=19)

sns.heatmap(bank.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

#There is no strong correlation between any two variables.

#There is no strong correlation between any independent variable and class variable.
sns.pairplot(bank.iloc[:,1:])
bank.head(1)
bank.corr()
# there are 52 records with negative experience. Before proceeding any further we need to clean the same

bank[bank['Experience'] < 0]['Experience'].count()
#clean the negative variable

dfExp = bank.loc[bank['Experience'] >0]

negExp = bank.Experience < 0

column_name = 'Experience'

mylist = bank.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience

mylist[:1]
negExp.value_counts()

"""The following code does the below steps:



For the record with the ID, get the value of Age column

For the record with the ID, get the value of Education column

Filter the records matching the above criteria from the data frame which has records with positive experience and take the median

Apply the median back to the location which had negative experience

"""

for id in mylist:

    age = bank.loc[np.where(bank['ID']==id)]["Age"].tolist()[0]

    education = bank.loc[np.where(bank['ID']==id)]["Education"].tolist()[0]

    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]

    exp = df_filtered['Experience'].median()

    bank.loc[bank.loc[np.where(bank['ID']==id)].index, 'Experience'] = exp

bank.head()
# checking if there are records with negative experience

bank[bank['Experience'] < 0]['Experience'].count()

bank.describe().transpose()

#Influence of income and education on personal loan¶

sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=bank)

sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=bank,color='yellow')

sns.countplot(x="Securities Account", data=bank,hue="Personal Loan")

sns.countplot(x='Family',data=bank,hue='Personal Loan',palette='Set1')

sns.countplot(x='CD Account',data=bank,hue='Personal Loan')

sns.distplot( bank[bank['Personal Loan']==0]['CCAvg'], color = 'r')

sns.distplot( bank[bank['Personal Loan']==1]['CCAvg'], color = 'g')

print('Credit card spending of Non-Loan customers: ',bank[bank['Personal Loan']==0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', bank[bank['Personal Loan']==1]['CCAvg'].median()*1000)
fig, ax = plt.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(bank['Experience'],bank['Age'],c=bank['Education'].apply(lambda x:colors[x]))

plt.xlabel('Experience')

plt.ylabel('Age')
corr = bank.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
sns.boxplot(x=bank['Family'],y=bank['Income'],hue=bank['Personal Loan'])

# Looking at the below plot, families with income less than 100K are less likely to take loan,than families with 

# high income
X=bank.drop(["ID","Experience","Personal Loan"],axis=1)

y=bank["Personal Loan"]

X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=11,test_size=0.3)
X_train.shape
print("{0:0.2f}% data is in training set".format((len(X_train)/len(bank.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(X_test)/len(bank.index)) * 100))
# #replace 0 with mean value

# rep_0=SimpleImputer(missing_values=0,strategy='mean')

# cols = X_train.columns

# X_train=pd.DataFrame(rep_0.fit_transform(X_train))

# x_test=pd.DataFrame(rep_0.fit_transform(X_test))

# X_train.columns = cols

# X_test.columns = cols

# X_train.head()
# Logistic Regression

model=LogisticRegression(solver='liblinear')

model.fit(X_train, y_train)

y_predict=model.predict(X_test)

coef_df = pd.DataFrame(model.coef_)

coef_df['intercept'] = model.intercept_

print(coef_df)

model_score= model.score(X_test, y_test)

print(model_score)
cm = metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )

NNH.fit(X_train,y_train)
predicted_labels = NNH.predict(X_test)

NNH.score(X_test, y_test)

# calculate accuracy measures and confusion matrix

print("Confusion Matrix")

metrics.confusion_matrix(y_test, predicted_labels, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["M","B"]],

                   columns = [i for i in ["Predict M","Predict B"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
#The model fit substantially with a perfect score on the training set and only 91% accuracy on the test set.



#SVM requires all the features to be on a similar scale. 

#We will need to rescale our data that all the features are approximately

#on the same scale and than see the performance
#Scale the data points using MinMaxScaler
scalar=MinMaxScaler()

X_train_scaled=scalar.fit_transform(X_train)

X_test_scaled=scalar.fit_transform(X_test)
#Fit SVM Model on Scale data and give your observation

svc=SVC()

svc.fit(X_train_scaled,y_train)
print("Accutacy of Training data set : {:.2f}".format(svc.score(X_train_scaled,y_train)))

print("Accutacy of Testing data set : {:.2f}".format(svc.score(X_test_scaled,y_test)))
svc=SVC(C=1000)

svc.fit(X_train_scaled,y_train)
print("Accutacy of Training data set : {:.2f}".format(svc.score(X_train_scaled,y_train)))

print("Accutacy of Testing data set : {:.2f}".format(svc.score(X_test_scaled,y_test)))
# Here, increasing C allows us to improve the model, resulting in 98.2% train set accuracy.
# calculate accuracy measures and confusion matrix

print("Confusion Matrix")

metrics.confusion_matrix(y_test, predicted_labels, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["M","B"]],

                   columns = [i for i in ["Predict M","Predict B"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)

gnb_model=GaussianNB()
gnb_model.fit(X_train,y_train.ravel())
gnb_model_pred=gnb_model.predict(X_train)
print("Model accurace is GNB : {:.2f}".format(metrics.accuracy_score(y_train,gnb_model_pred)))

print()
print(gnb_model_pred)

print("confusing matrics")

gnb_model_pred=gnb_model.predict(X_test)

cm=metrics.confusion_matrix(y_test, gnb_model_pred, labels=[1, 0])

cm


df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],columns=[i for i in ["predict 1","predict 0"]])

#plt.figure(figsize=(7,5))

sns.heatmap(df_cm,annot=True)
cm

print("classification Report")
print(metrics.classification_report(y_test, gnb_model_pred, labels=[1, 0]))
train_set_indep = bank.drop(['Experience' ,'ID'] , axis = 1).drop(labels= "Personal Loan" , axis = 1)

train_set_dep = bank["Personal Loan"]

X = np.array(train_set_indep)

Y = np.array(train_set_dep)

X_Train = X[ :3500, :]

X_Test = X[3501: , :]

Y_Train = Y[:3500, ]

Y_Test = Y[3501:, ]

knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')

knn.fit(X_Train, Y_Train)    

predicted = knn.predict(X_Test)

acc = accuracy_score(Y_Test, predicted)

print(acc)
X=bank.drop(['Personal Loan','Experience','ID'],axis=1)

y=bank.pop('Personal Loan')
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('NB', GaussianNB()))

models.append(('LR', LogisticRegression()))

models.append(('SVM', SVC()))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=12345)

	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
