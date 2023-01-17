import pandas as pd

import numpy as np

import matplotlib.pyplot as plt # matplotlib.pyplot plots data

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")

import pylab as pl



from scipy.stats import zscore
df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
df.shape
df.info()
df.isnull().values.any() # If there are any null values in data set
df.describe()
# Observation on given data

#### we have few negative values in experience which we need to have look at and update to median/mean.

# I will prefer to median.

### let's correct the data first then move on
df['Experience'].mask(df['Experience'] < 0, df.Experience.median(), inplace=True)
# now again check for negative values in data

df.describe()
columns = list(df)[0:-1] # Excluding Outcome column which has only 

df[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 

# Histogram of first 8 columns
sns.boxplot(df[['Income']])
sns.distplot( df['Age'], color = 'g')

## Observation

# Most of the customers age fall in the age range of 30 to 60 yrs and their experience falls

# in the range of 5 to 35 years and most earn an income between 10K to 100K.
# lets check the influence of Income level on whether a customer takes a personal loan across the education levels
sns.boxplot(x="Education", y="Income", hue="Personal Loan", data=df)
# observation

# The box plots show that those with education level 1 have higher incomes. 

# But customers who go for personal loans have the same income distribution regardless of the education level.
sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=df)
# Observation

# Customers who taken loan also have higher mortgages.
sns.countplot(x="ZIP Code", data=df[df["Personal Loan"] ==1], hue ="Personal Loan",orient ='v')
# top 10 locations who appled personal loan before



zipcode_top10 = df[df["Personal Loan"]==1]['ZIP Code'].value_counts().head(10)

zipcode_top10
sns.countplot(x="Family", data=df,hue="Personal Loan")
# let's check more on if family size have any influence on whether a customer accepts a personal loan offer
familysizewith_no = np.mean( df[df["Personal Loan"] == 0]['Family'] )

familysizewith_no
familysizewith_yes = np.mean( df[df["Personal Loan"] == 1]['Family'] )

familysizewith_yes
from scipy import stats

stats.ttest_ind(df[df["Personal Loan"] == 1]['Family'], df[df["Personal Loan"] == 1]['Family'])
# observation

# There is no impact of Family size on decision to take the loan
sns.countplot(x="Securities Account", data=df,hue="Personal Loan")
sns.countplot(x="CD Account", data=df,hue="Personal Loan")
sns.countplot(x="CreditCard", data=df, hue="Personal Loan")

sns.distplot( df[df["Personal Loan"] == 0]['CCAvg'], color = 'b')

sns.distplot( df[df["Personal Loan"] == 1]['CCAvg'], color = 'y')
# observation 

# Customer who didn't take the loan looks like have less credit card score than who have taken the loan

# Hence, high credit card average lokks to be good predictor to decide whether or not a customer will take the personal loan.
sns.distplot( df[df["Personal Loan"] == 0]['Income'], color = 'b')

sns.distplot( df[df["Personal Loan"] == 1]['Income'], color = 'y')
sns.distplot( df[df["Personal Loan"] == 0]['Education'], color = 'b')

sns.distplot( df[df["Personal Loan"] == 1]['Education'], color = 'y')
# let's check the correlation
df.corr() # It will show correlation matrix
# However we want to see correlation in graphical representation so below is function for that

def plot_corr(df, size=11):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(df)
plt = sns.pairplot(df[['Age','Experience','Income','ZIP Code','Family','CCAvg' , 'Mortgage','Personal Loan','Securities Account','CD Account','Online','CreditCard']])

df.head(1)
#### Observation

# Age and Experience are extremely corelated as seen from above graphs and maps

# Income and CCAvg are also corelated as seen from above graphs and maps
#Creating family dummy Variables

fa = pd.get_dummies(df['Family'], prefix='Family')

#Adding the results to the master dataframe

bank_df1 = pd.concat([df,fa], axis=1)
#Creating education dummy Variables

ed = pd.get_dummies(df['Education'], prefix='Education')

#Adding the results to the master dataframe

bank_df1 = pd.concat([bank_df1,ed], axis=1)
# We have created dummies for the below variables, so we can drop them

##bank_df2 is our new dataset after cleaning and transformation

bank_df2=bank_df1.drop(['Education','Family'],1)
var=['Securities Account','CD Account','Online','CreditCard']

bank_df2[var]=bank_df2[var].astype('int64')
bank_df2_X = bank_df2.drop(['Personal Loan'], axis=1)

x = bank_df2.drop(['Personal Loan'], axis=1)

y = bank_df2['Personal Loan']
x.dtypes

## 4. Split the data into training and test set in the ratio of 70:30 respectively 
#Test Train Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(bank_df2_X, y, train_size=0.7, test_size=0.3, random_state=1)
#Feature Scaling library

from sklearn.preprocessing import StandardScaler

#Applying Scaling to training set

scaler = StandardScaler()

x_train[['Age','Experience','Income','CCAvg']] = scaler.fit_transform(x_train[['Age','Experience','Income','CCAvg']])

x_test[['Age','Experience','Income','CCAvg']] = scaler.transform(x_test[['Age','Experience','Income','CCAvg']])
#let's check the split of data

print("{0:0.2f}% data is in training set".format((len(x_train)/len(df.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(x_test)/len(df.index)) * 100))
x_train.describe()
x_test.describe()
#### Data Preparation

# Check hidden missing values

# As we checked missing values earlier but haven't got any. 

# But there can be lots of entries with 0 values. We must need to take care of those as well.
x_train.head()
# If We can see lots of 0 entries above.
# Replace 0s with serial mean
#from sklearn.preprocessing import Imputer

#my_imputer = Imputer()

#data_with_imputed_values = my_imputer.fit_transform(original_data)



from sklearn.impute import SimpleImputer

rep_0 = SimpleImputer(missing_values=0, strategy="mean")

cols=x_train.columns

x_train = pd.DataFrame(rep_0.fit_transform(x_train))

x_test = pd.DataFrame(rep_0.fit_transform(x_test))



x_train.columns = cols

x_test.columns = cols

x_train.head()
# ## removing negative values

x_train.mask(x_train < 0, x_train.mean(), inplace=True, axis=1)

x_test.mask(x_test < 0, x_test.mean(), inplace=True, axis=1)

x_test.head()
from sklearn import metrics



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(x_train, y_train)
#Fitting the model on test data

x_test1 = x_test.drop(['Family_2','Education_1','Age','Mortgage','ZIP Code','CCAvg','Education_2',

                        'Education_3','Experience'],

                        axis=1)

x_test.columns
y_pred = logreg.predict(x_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
#true positives (TP): These are cases in which we predicted yes, and actually took loan.

TP=1344

#true negatives (TN): We predicted no, and they actually did not took loan.

TN=5

#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")

FP=7

#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")

FN=144
Accuracy=(TP+TN)/(TP+TN+FP+FN)

print('Accuracy of logistic regression classifier on test set: {:.2%}'.format(Accuracy))



Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)

print('Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))



#Recall

Sensitivity=TP/(FN+TP)

print('Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))



Specificity=TN/(TN+FP)

print('Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))



Precision=TP/(FP+TP)

print('Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))



#ROC Curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

pl.clf()

pl.plot(fpr, tpr, label='Logistic Reg (area = %0.2f)' % logit_roc_auc)

pl.plot([0, 1], [0, 1],'r--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.05])

pl.xlabel('FP Rate')

pl.ylabel('TP Rate')

pl.title('ROC')

pl.legend(loc="lower right")

pl.savefig('Log_ROC')

pl.show()
#### Naive Bayes Model
from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes



# creatw the model

GNB1 = GaussianNB()

GNB1.fit(x_train, y_train)

predicted_labels_GNB = GNB1.predict(x_test)



GNB1.score(x_test, y_test)

print(metrics.confusion_matrix(y_test, predicted_labels_GNB))

#CALCULATING METRICES FOR CHECING MODEL

#true positives (TP): These are cases in which we predicted yes, and actually took loan.

TP=1245

#true negatives (TN): We predicted no, and they actually did not took loan.

TN=78

#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")

FP=106

#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")

FN=71
Accuracy=(TP+TN)/(TP+TN+FP+FN)

print('Accuracy of logistic regression classifier on test set: {:.2%}'.format(Accuracy))



Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)

print('Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))



#Recall

Sensitivity=TP/(FN+TP)

print('Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))



Specificity=TN/(TN+FP)

print('Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))



Precision=TP/(FP+TP)

print('Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))



#ROC Curve

GNB_roc_auc = roc_auc_score(y_test, GNB1.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, GNB1.predict_proba(x_test)[:,1])

pl.clf()

pl.plot(fpr, tpr, label='Logistic Reg (area = %0.2f)' % GNB_roc_auc)

pl.plot([0, 1], [0, 1],'r--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.05])

pl.xlabel('FP Rate')

pl.ylabel('TP Rate')

pl.title('ROC')

pl.legend(loc="lower right")

pl.savefig('Log_ROC')

pl.show()
# Build kNN Model

from sklearn.neighbors import KNeighborsClassifier
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'uniform' )
# Call Nearest Neighbour algorithm



NNH.fit(x_train, y_train)
## # Evaluate Performance of kNN Model
predicted = NNH.predict(x_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, predicted)

acc
predicted_labels = NNH.predict(x_test)

NNH.score(x_test, y_test)
# # calculate accuracy measures and confusion matrix

# from sklearn import metrics



# print("Confusion Matrix")

# cm=metrics.confusion_matrix(y_test, predicted_labels, labels=["M", "B"])



# df_cm = pd.DataFrame(cm, index = [i for i in ["M","B"]],

#                   columns = [i for i in ["Predict M","Predict B"]])

# plt.figure(figsize = (7,5))

# sns.heatmap(df_cm, annot=True)

print(metrics.confusion_matrix(y_test, predicted_labels))

#CALCULATING METRICES FOR CHECING MODEL

#true positives (TP): These are cases in which we predicted yes, and actually took loan.

TP=1346

#true negatives (TN): We predicted no, and they actually did not took loan.

TN=0

#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")

FP=5

#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")

FN=149



Accuracy=(TP+TN)/(TP+TN+FP+FN)

print('Accuracy of logistic regression classifier on test set: {:.2%}'.format(Accuracy))



Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)

print('Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))



#Recall

Sensitivity=TP/(FN+TP)

print('Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))



Specificity=TN/(TN+FP)

print('Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))



Precision=TP/(FP+TP)

print('Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))



#ROC Curve

KNN_roc_auc = roc_auc_score(y_test, NNH.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, NNH.predict_proba(x_test)[:,1])

pl.clf()

pl.plot(fpr, tpr, label='Logistic Reg (area = %0.2f)' % KNN_roc_auc)

pl.plot([0, 1], [0, 1],'r--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.05])

pl.xlabel('FP Rate')

pl.ylabel('TP Rate')

pl.title('ROC')

pl.legend(loc="lower right")

pl.savefig('Log_ROC')

pl.show()
# Call Nearest Neighbour algorithm, keeping number of neighbours as 7

NNH2 = KNeighborsClassifier(n_neighbors= 7, weights = 'uniform' )

NNH2.fit(x_train, y_train)



predicted_labels_KNN = NNH2.predict(x_test)



NNH2.score(x_test, y_test)



print(metrics.confusion_matrix(y_test, predicted_labels_KNN))

#CALCULATING METRICES FOR CHECING MODEL

#true positives (TP): These are cases in which we predicted yes, and actually took loan.

TP=1346

#true negatives (TN): We predicted no, and they actually did not took loan.

TN=0

#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")

FP=5

#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")

FN=149



Accuracy=(TP+TN)/(TP+TN+FP+FN)

print('Accuracy of logistic regression classifier on test set: {:.2%}'.format(Accuracy))



Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)

print('Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))



#Recall

Sensitivity=TP/(FN+TP)

print('Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))



Specificity=TN/(TN+FP)

print('Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))



Precision=TP/(FP+TP)

print('Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))



#ROC Curve

KNN_roc_auc = roc_auc_score(y_test, NNH2.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, NNH2.predict_proba(x_test)[:,1])

pl.plot(fpr, tpr, label='Logistic Reg (area = %0.2f)' % KNN_roc_auc)

pl.plot([0, 1], [0, 1],'r--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.05])

pl.xlabel('FP Rate')

pl.ylabel('TP Rate')

pl.title('ROC')

pl.legend(loc="lower right")

pl.savefig('Log_ROC')

pl.show()
scores =[]

for k in range(1,50):

    NNH = KNeighborsClassifier(n_neighbors = k, weights = 'distance' )

    NNH.fit(x_train, y_train)

    scores.append(NNH.score(x_test, y_test))
pl.plot(range(1,50),scores)
#All 3 models Results:-----





#Logistic 

#Accuracy of logistic regression classifier on test set: 89.93%

#Misclassification Rate: It is often wrong: 10.07%

#Sensitivity: When its actually yes how often it predicts yes: 90.32%

#Specificity: When its actually no, how often does it predict no: 41.67%

#Precision: When it predicts yes, how often is it correct: 99.48%







#Naïve Bayes

#Accuracy of logistic regression classifier on test set: 88.20%

#Misclassification Rate: It is often wrong: 11.80%

#Sensitivity: When its actually yes how often it predicts yes: 94.60%

#Specificity: When its actually no, how often does it predict no: 42.39%

#Precision: When it predicts yes, how often is it correct: 92.15%







#KNN  

#Accuracy of logistic regression classifier on test set: 89.73%

#Misclassification Rate: It is often wrong: 10.27%

#Sensitivity: When its actually yes how often it predicts yes: 90.03%

#Specificity: When its actually no, how often does it predict no: 0.00%

#Precision: When it predicts yes, how often is it correct: 99.63%
#we can see from above results that we have from graphs and different models. we can conclude
#In case we apply the model and change our approach and target only those which have been predicted as yes first, 

# it is 92.15% (Precision) that they will take loan.Hence, chances that a person takes personal loan increases.

# but Sensitivity is around 94.60% we end up offering more loans at the  end.