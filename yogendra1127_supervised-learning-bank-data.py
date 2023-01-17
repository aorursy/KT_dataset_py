# Import neccessary basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Load & read the source file to understand it in detail

Loan_df = pd.read_csv('../input/bank-sl-project/Bank_SL_Project.csv')
Loan_df.head() # displays the top 5 rows of dataframe.
# getting total number of rows and column in the dataframe
print(f" Shape of the dataframe = {Loan_df.shape}")
totalrows=Loan_df.shape[0]
print(f" Total number of rows in the dataset =  {totalrows}")
Loan_df.info() 
Loan_df.dtypes # Checking data type of each colunm to check if any type needs to be changed
# Checking Missing values in dataset

Loan_df.isna().sum()
Loan_df.isnull().sum()
Loan_df['CreditCard'].value_counts() # It is categorical Data
Loan_df['Family'].value_counts() # It is categorical Data
Loan_df['Personal Loan'].value_counts() # It is categorical Data
Loan_df['Securities Account'].value_counts() # It is categorical Data
Loan_df['CD Account'].value_counts() # It is categorical Data
Loan_df['Online'].value_counts() # It is categorical Data
Loan_df['CreditCard'].value_counts() # It is categorical Data
#Convert variables to a categorical variable as relevant

Loan_df['Family'] = Loan_df['Family'].astype('category')
Loan_df['Education'] = Loan_df['Education'].astype('category')
Loan_df['Personal Loan']=Loan_df['Personal Loan'].astype('category')
Loan_df['Securities Account']=Loan_df['Securities Account'].astype('category')
Loan_df['CD Account']=Loan_df['CD Account'].astype('category')
Loan_df['Online']=Loan_df['Online'].astype('category')
Loan_df['CreditCard']=Loan_df['CreditCard'].astype('category')
# Now Recheck the Data type of all attributes in data set
Loan_df.dtypes
# Ratio of Yes to No to identify data imbalance in Dependent Variable 'Personal Loan'

Loan_df['Personal Loan'].value_counts(normalize=True)
# Check data distribution using summary statistics
Loan_df.describe(include='all').T.round(2)
# Corelation analysis between the variables using heat map

corelation = plt.cm.viridis_r # Color range used in heat map
plt.figure(figsize=(15,10))
plt.title('Corelation between Attributes', y=1.02, size=20)
sns.heatmap(data=Loan_df.corr().round(2), linewidths=0.1, vmax=1, square=True, cmap=corelation, linecolor='black', annot=True);
Loan_df.columns
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Distribution of Age')
sns.distplot(Loan_df['Age'], color='r')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Distribution of Exp.')
sns.distplot(Loan_df['Experience'], color='g')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Distribution of Income')
sns.distplot(Loan_df['Income'], color='b')


# Box plot distribution of Data
plt.figure(figsize=(20,8))
plt.subplot(1,4,1)
plt.title('Distribution of Age')
sns.boxplot(Loan_df['Age'],orient='vertical',color='r')

# Box Subplot 2
plt.subplot(1,4,2)
plt.title('Distribution of Exp.')
sns.boxplot(Loan_df['Experience'],orient='vertical',color='g')

# Box Subplot 3
plt.subplot(1,4,3)
plt.title('Distribution of Income')
sns.boxplot(Loan_df['Income'],orient='vertical',color='b')

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Distribution of ZIP Code')
sns.distplot(Loan_df['ZIP Code'],color='r')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Distribution of Mortgage')
sns.distplot(Loan_df['Mortgage'], color='g')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Distribution of CC Avg.')
sns.distplot(Loan_df['CCAvg'], color='y')



# Box plot distribution of Data
plt.figure(figsize=(20,6))
plt.subplot(1,4,1)
plt.title('Distribution of ZIP Code')
sns.boxplot(Loan_df['ZIP Code'],orient='vertical',color='r')

# Box Subplot 2
plt.subplot(1,4,2)
plt.title('Distribution of Mortgage')
sns.boxplot(Loan_df['Mortgage'],orient='vertical',color='g')

# Box Subplot 3
plt.subplot(1,4,3)
plt.title('Distribution of CC Avg.')
sns.boxplot(Loan_df['CCAvg'],orient='vertical',color='y')


plt.figure(figsize=(16,6))
plt.subplot(1,3,1)
plt.title('Distribution of Family Members')
sns.countplot(Loan_df['Family'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Distribution of Education')
sns.countplot(Loan_df['Education'], palette='Reds')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Distribution of Account Security (1=Yes, 0=N0)')
sns.countplot(Loan_df['Securities Account'], palette='Greys')

# Subplot 4
plt.figure(figsize=(22,6))
plt.subplot(1,4,1)
plt.title('Distribution of CD Account (1=Yes, 0=N0)')
sns.countplot(Loan_df['CD Account'], palette='viridis')

# Subplot 5
plt.subplot(1,4,2)
plt.title('Distribution of Online Usage (1=Yes, 0=N0)')
sns.countplot(Loan_df['Online'], palette='RdYlGn')

# Subplot 6
plt.subplot(1,4,3)
plt.title('Distribution of Credit Card (1=Yes, 0=N0)')
sns.countplot(Loan_df['CreditCard'], palette='Accent')
# Univariate distribution of Target Variable

plt.title('Distribution of Personal Loan')
sns.countplot(Loan_df['Personal Loan'], palette='Accent')
# Relationship of Dependent Variable on Independent Attributes

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Personal Loan Vs Age')
sns.boxplot(Loan_df['Age'], Loan_df['Personal Loan'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Personal Loan Vs Exp.')
sns.boxplot(Loan_df['Experience'], Loan_df['Personal Loan'], palette='Oranges')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Personal Loan Vs Income')
sns.boxplot(Loan_df['Income'], Loan_df['Personal Loan'], palette='Blues')

# Relationship of Dependent Variable on Independent Attributes

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Personal Loan Vs ZIP Code')
sns.boxplot(Loan_df['ZIP Code'], Loan_df['Personal Loan'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Personal Loan Vs Mortgage')
sns.boxplot(Loan_df['Mortgage'], Loan_df['Personal Loan'], palette='Oranges')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Personal Loan Vs CC Avg.')
sns.boxplot(Loan_df['CCAvg'], Loan_df['Personal Loan'], palette='Blues')
# Relationship of Dependent Variable on Independent Attributes

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Personal Loan Vs Income')
sns.barplot(Loan_df['Personal Loan'], Loan_df['Income'], hue=Loan_df['Family'],palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Personal Loan Vs Age')
sns.barplot(Loan_df['Personal Loan'],Loan_df['Age'], hue=Loan_df['Education'], palette='Oranges')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Personal Loan Vs Experience')
sns.barplot(Loan_df['Personal Loan'], Loan_df['Experience'], hue=Loan_df['Education'], palette='Blues')
# Continued study of Dependent Variable.

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Personal Loan Vs Mortgage')
sns.barplot(Loan_df['Personal Loan'], Loan_df['Mortgage'], hue=Loan_df['Family'],palette='Greys')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Personal Loan Vs CC Usage')
sns.barplot(Loan_df['Personal Loan'],Loan_df['CCAvg'], hue=Loan_df['Education'], palette='YlOrRd')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Personal Loan Vs Income')
sns.barplot(Loan_df['CD Account'], Loan_df['Income'], hue=Loan_df['Personal Loan'], palette='Greys')
# Understand the Data distribution through Pair Plot.

sns.pairplot(Loan_df)
Loan_df.head()
# Defining Dependent & Independent Variables for model inputs

X = Loan_df.drop(['ID', 'Personal Loan'], axis=1) # ID is not having any influence on data set hence droped from Independent Variables
y = Loan_df['Personal Loan'] # Dependent Variable
X.head()
y.head()
X = pd.get_dummies(X, drop_first=True ) # #Convert categorical vriables to dummy variables
X.head()
# Import train test model to spilt the data in 70:30 ration

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("{0:0.2f}% data is in training set".format((len(X_train)/len(Loan_df.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_test)/len(Loan_df.index)) * 100))
print("Original Personal Loan True Values    :{0}({1:0.2f}%)".format(len(Loan_df.loc[Loan_df['Personal Loan']==1]), (len(Loan_df.loc[Loan_df['Personal Loan']==1])/len(Loan_df.index))*100))
print("Orinical Personal Loan False Values   :{0}({1:0.2f}%)".format(len(Loan_df.loc[Loan_df['Personal Loan']==0]), (len(Loan_df.loc[Loan_df['Personal Loan']==0])/len(Loan_df.index))*100))
print("")
print ("Training Personal Loan True Values   :{0}({1:0.2f}%)".format(len(y_train.loc[y_train[:]==1]),(len(y_train.loc[y_train[:]==1])/len(y_train.index))*100))
print ("Training Personal Loan False Values  :{0}({1:0.2f}%)".format(len(y_train.loc[y_train[:]==0]),(len(y_train.loc[y_train[:]==0])/len(y_train.index))*100))
print("")
print ("Testing Personal Loan True Values    :{0}({1:0.2f}%)".format(len(y_test.loc[y_test[:]==1]),(len(y_test.loc[y_test[:]==1])/len(y_test.index))*100))
print ("Testing Personal Loan False Values   :{0}({1:0.2f}%)".format(len(y_test.loc[y_test[:]==0]),(len(y_test.loc[y_test[:]==0])/len(y_test.index))*100))
# To model the Navie Bayes classifier imoprt Bernoulli, Multinomial & Gaussian classifiers

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
# To calculate the accuracy score of the model, report & build confusion metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, recall_score
from sklearn.metrics import confusion_matrix
# Implement Classifiers to build the Model using Bernoulli Classifier
Ber = BernoulliNB()
Ber.fit(X_train, y_train) # Model Trained using training Data
y_pred = Ber.predict(X_test) # Model is ready for predictions based on test Data
y_pred_Train1 = Ber.predict(X_train) # Prediction of Training Data
confusion_matrix(y_test, y_pred)
print("Accuracy of the Bernoulli NB is  :({:0.2f}%)".format(accuracy_score(y_pred, y_test)*100)) # Accuracy of our Bernoulli Naive Bayes model

cm1 = confusion_matrix(y_train,y_pred_Train1, labels=[0,1]) # Confusion metrix of Bernoulli NB Classifier on Test Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm1_df = pd.DataFrame(cm1, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm1_df, annot=True, fmt='.4g')

cm2 = confusion_matrix(y_test,y_pred, labels=[0,1]) # Confusion metrix of Bernoulli NB Classifier on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm2_df = pd.DataFrame(cm2, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm2_df, annot=True, fmt='.4g')
print(classification_report(y_test, y_pred)) # Classification Report of Bernoulli NB Model
print("")
print(classification_report(y_train,y_pred_Train1))
# Implement Classifiers to build the Model using Gaussian Classifier
Gau = GaussianNB()
Gau.fit(X_train, y_train) # Model Trained using training Data.
y_pred1 = Gau.predict(X_test) # Model is ready for predictions based on test Data
y_pred_Train2 = Gau.predict(X_train)
print("Accuracy of the Gaussian NB is  :({:0.2f}%)".format(accuracy_score(y_pred1, y_test)*100)) # Accuracy of our Gaussian Naive Bayes model
cm3 = confusion_matrix(y_train,y_pred_Train2, labels=[0,1]) # Confusion metrix of Gaussian NB Classifier on Test Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm3_df = pd.DataFrame(cm3, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm3_df, annot=True, fmt='.5g')

cm4 = confusion_matrix(y_test,y_pred1, labels=[0,1]) # Confusion metrix of Gaussian NB Classifier on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm4_df = pd.DataFrame(cm4, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm4_df, annot=True, fmt='.5g')
print (classification_report(y_test,y_pred1)) # Print Classification Report to study other important parameters of the model
print("")
print(classification_report(y_train,y_pred_Train2)) # Classification Report of Training Data
# Import supporting Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#Build the logistic regression model
logisticRegr = LogisticRegression(solver='liblinear')
logisticRegr.fit(X_train,y_train)
y_pred2 = logisticRegr.predict(X_test) # Model is ready to predict based on test Data
y_pred_Train3 = logisticRegr.predict(X_train) # Predictions for Training Data
print(logisticRegr.score(X_test, y_test))
print("Accuracy of the Logistic Regression Model is  :({:0.2f}%)".format(accuracy_score(y_pred2, y_test)*100))
import statsmodels.api as sm

logit = sm.Logit(y_train, sm.add_constant(X_train))
lg = logit.fit()
lg.summary2()
#Calculate Odds Ratio, probability
##create a data frame to collate Odds ratio, probability and p-value of the coef
lgcoef = pd.DataFrame(lg.params, columns=['coef'])
lgcoef.loc[:, "Odds_ratio"] = np.exp(lgcoef.coef)
lgcoef['probability'] = lgcoef['Odds_ratio']/(1+lgcoef['Odds_ratio'])
lgcoef['pval']=lg.pvalues
pd.options.display.float_format = '{:.2f}'.format
# FIlter by significant p-value (pval <0.1) and sort descending by Odds ratio
lgcoef = lgcoef.sort_values(by="Odds_ratio", ascending=False)
pval_filter = lgcoef['pval']<=0.1
lgcoef[pval_filter]
cm_LR1 = plt.cm.Greens_r # Color Scheme for confusion metrics
cm5 = confusion_matrix(y_train,y_pred_Train3, labels=[0,1]) # Confusion metrix for logistic Regression Classifier on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm5_df = pd.DataFrame(cm5, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm5_df, annot=True, fmt='.5g', cmap=cm_LR1, linecolor='Black', square=True)

cm_LR2 = plt.cm.Reds_r # Color Scheme for confusion metrics
cm6 = confusion_matrix(y_test,y_pred2, labels=[0,1]) # Confusion metrix of Logistic Regression Classifier on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm6_df = pd.DataFrame(cm6, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm6_df, annot=True, fmt='.5g', cmap=cm_LR2, linecolor='Black', square=True)
print(classification_report(y_pred2, y_test))
print("")
print(classification_report(y_pred_Train3, y_train))
# ROC AUC Curves Logistic Regression.

logit_roc_auc = roc_auc_score(y_test, y_pred2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Import libraries to build KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
# Call KNN algorithm

knn.fit(X_train,y_train)
# For every test data point, predict it's label based on 5 nearest neighbours in this model.

y_pred3 = knn.predict(X_test)
y_pred_Train4 = knn.predict(X_train)
print("Accuracy of the KNN Test Model is      :({:0.2f}%)".format(accuracy_score(y_pred3, y_test)*100))
print("Accuracy of the KNN Training Model is  :({:0.2f}%)".format(accuracy_score(y_pred_Train4, y_train)*100))
print(confusion_matrix(y_pred3, y_test)) # Confusion metrics for Test Data with K=5 Value
print("")
print(confusion_matrix(y_pred_Train4,y_train)) # Confusion metrics for Training Data with K=5 value
print("")
print("")
print(classification_report(y_pred3, y_test))
print("")
print(classification_report(y_pred_Train4,y_train))
# Building graphical metrics to find optimal value of K.
scores = []
for k in range (1,100):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test,y_test))
plt.plot(range(1,100), scores)
plt.xlabel("Number of K Neighbors")
plt.ylabel("Accuracy Score")
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))
# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 1,3,5....49
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred3 = knn.predict(X_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred3)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
#Plot misclassification error vs k (with k value on X-axis) using matplotlib.
# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
#Use K=23 as the final model for prediction
knnfinal = KNeighborsClassifier(n_neighbors = 23)

# fitting the model
knnfinal.fit(X_train, y_train)

# predict the response
y_pred4 = knnfinal.predict(X_test)
y_pred_Train5 = knnfinal.predict(X_train)

# evaluate accuracy

print("Accuracy of the KNN Test Model is      :({:0.2f}%)".format(accuracy_score(y_test, y_pred4)*100))
print("Recall of the KNN Test Model is        :({:0.2f}%)".format(recall_score(y_test, y_pred4)*100))
cm_knn1 = plt.cm.Greys_r # Color Scheme for confusion metrics
cm9 = confusion_matrix(y_train,y_pred_Train5, labels=[0,1]) # Confusion metrix for KNN Classifier on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm9_df = pd.DataFrame(cm9, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm9_df, annot=True, fmt='.5g', cmap=cm_knn1, linecolor='Black', square=True)

cm_knn2 = plt.cm.Oranges_r # Color Scheme for confusion metrics
cm10 = confusion_matrix(y_test,y_pred4, labels=[0,1]) # Confusion metrix of KNN Classifier on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm10_df = pd.DataFrame(cm10, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm10_df, annot=True, fmt='.5g', cmap=cm_knn2, linecolor='Black', square=True)
print(classification_report(y_test, y_pred4))
print("")
print(classification_report(y_train, y_pred_Train5))
# Import SVM library for the model building

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train) # Model training

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
# Scale the data points using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# Now fit the data on scaled points
svc.fit(X_train_scaled,y_train)
print('Accuracy on training set: {:.2f}'.format(svc.score(X_train_scaled,y_train)))
print('Accuracy on testting set: {:.2f}'.format(svc.score(X_test_scaled,y_test)))
# predict the response
y_pred5 = svc.predict(X_test_scaled)
y_pred_Train6 = svc.predict(X_train_scaled)
confusion_matrix(y_test,y_pred5)
cm_svm1 = plt.cm.Blues_r # Color Scheme for confusion metrics
cm11 = confusion_matrix(y_train,y_pred_Train6, labels=[0,1]) # Confusion metrix for SVM on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm11_df = pd.DataFrame(cm11, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm11_df, annot=True, fmt='.5g', cmap=cm_svm1, linecolor='Black', square=True)

cm_svm2 = plt.cm.Oranges_r # Color Scheme for confusion metrics
cm12 = confusion_matrix(y_test,y_pred5, labels=[0,1]) # Confusion metrix for SVM on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm12_df = pd.DataFrame(cm12, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm12_df, annot=True, fmt='.5g', cmap=cm_svm2, linecolor='Black', square=True)
print(classification_report(y_test,y_pred5))
print("")
print(classification_report(y_train,y_pred_Train6))
# Try improving the SVM model accuracy using C & gamma
svc1 = SVC(gamma=0.1, C=1000)
svc1.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.1, C=1000): {:.3f}".format(svc1.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.1, C=1000): {:.3f}".format(svc1.score(X_test_scaled, y_test)*100))
print("")

svc2 = SVC(gamma=0.1, C=100)
svc2.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.1, C=100): {:.3f}".format(svc2.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.1, C=100): {:.3f}".format(svc2.score(X_test_scaled, y_test)*100))
print("")

svc3 = SVC(gamma=0.01, C=1000)
svc3.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.01, C=100): {:.3f}".format(svc3.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.01, C=100): {:.3f}".format(svc3.score(X_test_scaled, y_test)*100))
print("")

svc4 = SVC(gamma=0.01, C=100)
svc4.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.01, C=100): {:.3f}".format(svc4.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.01, C=100): {:.3f}".format(svc4.score(X_test_scaled, y_test)*100))
print("")

svc5 = SVC(gamma=0.01, C=10)
svc5.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.01, C=10): {:.3f}".format(svc5.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.01, C=10): {:.3f}".format(svc5.score(X_test_scaled, y_test)*100))
print("")

svc6 = SVC(gamma=0.1, C=10)
svc6.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=0.1, C=10): {:.3f}".format(svc6.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=0.1, C=10): {:.3f}".format(svc6.score(X_test_scaled, y_test)*100))
print("")

svc7 = SVC(gamma=1, C=10)
svc7.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=1, C=10): {:.3f}".format(svc7.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=1, C=10): {:.3f}".format(svc7.score(X_test_scaled, y_test)*100))
print("")

svc8 = SVC(gamma=10, C=100)
svc8.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=10, C=100): {:.3f}".format(svc8.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=10, C=100): {:.3f}".format(svc8.score(X_test_scaled, y_test)*100))
print("")

svc9 = SVC(gamma=100, C=1000)
svc9.fit(X_train_scaled, y_train)

print("Accuracy on training set (g=100, C=1000): {:.3f}".format(svc9.score(X_train_scaled, y_train)*100))
print("Accuracy on test set (g=100, C=1000): {:.3f}".format(svc9.score(X_test_scaled, y_test)*100))
print("")


# predict the response with svm2 Model
y_pred6 = svc2.predict(X_test_scaled)
y_pred_Train7 = svc2.predict(X_train_scaled)

# predict the response with svm6 Model
y_pred7=svc6.predict(X_test_scaled)
y_pred_Train8 =svc6.predict(X_train_scaled)

print(confusion_matrix(y_test,y_pred6))
print("")
print(confusion_matrix(y_test,y_pred7))
cm_svm3 = plt.cm.Greys_r # Color Scheme for confusion metrics
cm13 = confusion_matrix(y_train,y_pred_Train7, labels=[0,1]) # Confusion metrix for SVM on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm13_df = pd.DataFrame(cm13, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm13_df, annot=True, fmt='.5g', cmap=cm_svm3, linecolor='Black', square=True)

cm_svm4 = plt.cm.Blues_r # Color Scheme for confusion metrics
cm14 = confusion_matrix(y_test,y_pred6, labels=[0,1]) # Confusion metrix for SVM on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm14_df = pd.DataFrame(cm14, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm14_df, annot=True, fmt='.5g', cmap=cm_svm4, linecolor='Black', square=True)
print(classification_report(y_test,y_pred6))
print("")
print(classification_report(y_train,y_pred_Train7))
# ROC AUC Curves SMV.

svm_roc_auc = roc_auc_score(y_test, y_pred6)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred6)
plt.figure()
plt.plot(fpr1, tpr1, label='SVM (area = %0.2f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def generate_model_report (y_actual, y_predicted):
    print ("Accuracy =", round(accuracy_score(y_actual,y_predicted)*100,2))
    print ("Precision =", round(precision_score(y_actual,y_predicted)*100,2))
    print ("Recall =", round(recall_score(y_actual,y_predicted)*100,2))
    print ("F1 Score =", round(f1_score(y_actual,y_predicted)*100,2))
    pass
print("Bernoulli NB Model Report")
print("")
generate_model_report(y_test,y_pred)
print("")

print("Gaussian NB Model Report")
print("")
generate_model_report(y_test,y_pred1)
print("")

print("Logistic Regression Model Report")
print("")
generate_model_report(y_test,y_pred2)
print("")

print("KNN Model Report K=5")
print("")
generate_model_report(y_test,y_pred3)
print("")

print("KNN Model Report K=23")
print("")
generate_model_report(y_test,y_pred4)
print("")

print("SVM Model Report with Default gamma & C ")
print("")
generate_model_report(y_test,y_pred5)
print("")

print("SVM Model Report with Optimized gamma & C ")
print("")
generate_model_report(y_test,y_pred6)
print("")