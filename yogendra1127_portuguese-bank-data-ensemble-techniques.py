# Import neccessary basic packages/libraries to start with

import numpy as np                          # used for numeric operations
import pandas as pd                         # used for data structuring/framing operations
import matplotlib.pyplot as plt             # Used for data visualization
%matplotlib inline
import seaborn as sns                       # Used for data visualization
import warnings                             # Used for filtering the warnings in code
warnings.filterwarnings('ignore')
bank_df = pd.read_csv('../input/ensemble-technique-data/Portuguese_Bank.csv') # Data set is converted into a pandas dataframe now
bank_df.head(10) # Dsiplays the first 10 rows of Dataframe
bank_df.tail(10) # Dsiplays the last 10 rows of Dataframe
# Check the shape of given data set

print(f"Shape of the Dataframe = {bank_df.shape}")
total_rows = bank_df.shape[0]
total_columns = bank_df.shape[1]
print(f"Total number of rows in given data set are = {total_rows}")
print(f"Total number of columns in given data set are = {total_columns}")
# Check the data type of attributes in data set
bank_df.info()
bank_df.dtypes
# Check the missing values in dataset
bank_df.isna()
bank_df.isnull().sum()
# Check the Value Counts of categorical attributes in data set.
print(f'''**Job Type** 
================================
{bank_df['job'].value_counts()}
================================''')

print(f'''**Marital Status**
================================
{bank_df['marital'].value_counts()}
================================''')
print(f'''**Education Summary**
================================
{bank_df['education'].value_counts()}
================================      ''')
print(f'''**Default Summary**
================================
{bank_df['default'].value_counts()}
================================''')
print(f'''**Housing Loan Summary**
================================
{bank_df['housing'].value_counts()}
================================''')
print(f'''**Loan Summary**
================================
{bank_df['loan'].value_counts()}
================================''')
print(f'''** Contact Type**
================================
{bank_df['contact'].value_counts()}
================================''')
print(f'''**Outcome of Previous Campaign**
================================
{bank_df['poutcome'].value_counts()}
================================''')
print(f'''**Summary of Target Variable**
================================
{bank_df['Target'].value_counts()}
================================''')
# Lets explore Ratio of "Yes" to "No" in dependent variable "Target"

print(f'''(% Distribution of Target Variable)
{round(bank_df['Target'].value_counts(normalize=True)*100,3)}''')
# Now Check the distribution of Data using five point summary statistics.

bank_df.describe(include='all').round(2)
### it seems typo error,lets handle negative values of columns "balance" & "pdays".
### Convert negative values to positives by using abs function.

bank_df['balance']=bank_df['balance'].abs()
bank_df['pdays']=bank_df['pdays'].abs()
## again look at five point summary to cross check the distribution.

bank_df.describe().round(2).T
## Corelation analysis between variables using heat map

corelation = plt.cm.viridis  # Color range used in heat map
plt.figure(figsize=(15,10));
plt.title('Corelation Between Attributes', y=1.02, size=20);
sns.heatmap(data=bank_df.corr().round(2), linewidths=0.1, vmax=1, square=True, cmap=corelation, linecolor='black',annot=True);
bank_df.columns
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Distribution of Age')
sns.distplot(bank_df['age'], color='r')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Distribution of Balance')
sns.distplot(bank_df['balance'], color='b')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Distribution of Day')
sns.distplot(bank_df['day'], color='g')

# Box plot distribution of Data
plt.figure(figsize=(20,6))
plt.subplot(1,4,1)
plt.title('Distribution of Age')
sns.boxplot(bank_df['age'], orient='vertical', color='r')

# Box Subplot 2
plt.subplot(1,4,2)
plt.title('Distribution of Balance')
sns.boxplot(bank_df['balance'], orient='vertical', color='b')

# Box Subplot 3
plt.subplot(1,4,3)
plt.title('Distribution of Days')
sns.boxplot(bank_df['day'], orient='vertical', color='g')
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title('Distribution of Duration')
sns.distplot(bank_df['duration'], color='r')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Distribution of Campaign')
sns.distplot(bank_df['campaign'], color='b')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Distribution of Campaign')
sns.distplot(bank_df['campaign'], color='g')

# Box plot distribution of Data
plt.figure(figsize=(20,6))
plt.subplot(1,4,1)
plt.title('Distribution of Duration')
sns.boxplot(bank_df['duration'], orient='vertical', color='r')

# Box Subplot 2
plt.subplot(1,4,2)
plt.title('Distribution of Campaign')
sns.boxplot(bank_df['campaign'], orient='vertical', color='b')

# Box Subplot 3
plt.subplot(1,4,3)
plt.title('Distribution of pdays')
sns.boxplot(bank_df['pdays'], orient='vertical', color='g')
bank_df.columns # for reference of column names in analysis.
plt.figure(figsize=(40,6))
plt.subplot(1,2,1)
plt.title('Distribution of Job')
sns.countplot(bank_df['job'], palette='Greens')

#Subplot 2
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.title('Distribution of Marital Status')
sns.countplot(bank_df['marital'], palette='Reds')

# Subplot 3
plt.subplot(1,3,2)
plt.title('Distribution of Education')
sns.countplot(bank_df['education'], palette='Greys')

# Subplot 4
plt.subplot(1,3,3)
plt.title('Distribution of Default Cust')
sns.countplot(bank_df['default'], palette='viridis')

#Subplot 5
plt.figure(figsize=(22,6))
plt.subplot(1,4,1)
plt.title('Distribution of Housing Loan')
sns.countplot(bank_df['housing'], palette='RdYlGn')

#Subplot 6
plt.subplot(1,4,2)
plt.title('Distribution of Loan')
sns.countplot(bank_df['loan'], palette='Accent')

# Subplot 7
plt.subplot(1,4,3)
plt.title('Distribution of Contact Type')
sns.countplot(bank_df['contact'], palette='Greens')

# Subplot 8
plt.figure(figsize=(50,6))
plt.subplot(1,5,1)
plt.title('Distribution of Month')
sns.countplot(bank_df['month'], palette='Reds')

#Subplot 9
plt.subplot(1,5,2)
plt.title('Distribution of poutcome')
sns.countplot(bank_df['poutcome'], palette='Greys')
# Univariate distribution of Target Variable

plt.title('Distribution of Target Variable')
sns.countplot(bank_df['Target'], palette='PiYG')
bank_df.columns
# Relationship of Dependent Variable on Independent numeric Attributes using box plot (bi-variate Analysis)

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title("Target Vs Age")
sns.boxplot(bank_df['age'], bank_df['Target'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title("Target Vs Balance")
sns.boxplot(bank_df['balance'], bank_df['Target'], palette='Reds')

# Subplot 3
plt.subplot(1,3,3)
plt.title("Target Vs Duration")
sns.boxplot(bank_df['duration'], bank_df['Target'], palette='Oranges')

# Subplot 4
plt.figure(figsize=(30,6))
plt.subplot(1,4,1)
plt.title("Target Vs Campaign")
sns.boxplot(bank_df['campaign'], bank_df['Target'], palette='Greys')

# Subplot 5
plt.subplot(1,4,2)
plt.title("Target Vs Pdays")
sns.boxplot(bank_df['pdays'], bank_df['Target'], palette='Blues')

# Subplot 6
plt.subplot(1,4,3)
plt.title("Target Vs Previous")
sns.boxplot(bank_df['previous'], bank_df['Target'], palette='Set1')
# Relationship of Dependent Variable on Independent categorical Attributes (bi-variate Analysis)

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('Target Vs Age')
sns.barplot(bank_df['Target'], bank_df['age'], hue=bank_df['marital'],palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Target Vs Balamce')
sns.barplot(bank_df['Target'],bank_df['balance'], hue=bank_df['education'], palette='Oranges')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Target Vs Duration')
sns.barplot(bank_df['Target'], bank_df['duration'], hue=bank_df['default'], palette='Blues')

# Bi-variate Analysis Continues

plt.figure(figsize=(60,6))
plt.subplot(1,3,1)
plt.title('Target Vs Duration (over Job)')
sns.barplot(bank_df['Target'], bank_df['duration'], hue=bank_df['job'],palette='Reds')


plt.figure(figsize=(80,6))
plt.subplot(1,4,1)
plt.title('Target Vs Duration (Over Month)')
sns.barplot(bank_df['Target'], bank_df['duration'], hue=bank_df['month'],palette='Greys')
# Calculate number of customers with Yes & No in Target variable

print(f'''Number of Customers subsribed for Term deposit:- 
{bank_df['Target'].value_counts()}''')
# Average contact duration for Customers with "Yes" & "No" tags in Target attribute

print(f'''Avg. contact duration of Customer with term deposit (Yes) = {round(bank_df[bank_df['Target']=='yes']['duration'].mean(),2)}''')
print("")
print(f'''Avg. contact duration of Customer without term deposit (No) = {round(bank_df[bank_df['Target']=='no']['duration'].mean(),2)}''')
# Standard Deviation for Customers with "Yes" & "No" tags in Target attribute

print(f'''Standard Deviation of Customer with term deposit (Yes) = {round(bank_df[bank_df['Target']=='yes']['duration'].std(),2)}''')
print("")
print(f'''Standard Deviation of Customer without term deposit (No) = {round(bank_df[bank_df['Target']=='no']['duration'].std(),2)}''')
duration_df = bank_df[['duration','Target']]
duration_df.head()
# Will use numpy for creating 2d array from above data frame

Group1 = np.array(duration_df)
Group1
# Seperating the data into two groups
Duration_Yes= Group1[:,1]=='yes'
Duration_Yes = Group1[Duration_Yes][:,0]
Duration_No = Group1[:,1]=='no'
Duration_No = Group1[Duration_No][:,0]
# Now we will use two sample t-test on these groups assuming alpha =0.05
# importing neccessary libraries for test
from scipy.stats import ttest_ind, shapiro,levene
from statsmodels.stats.power import ttest_power
t_stats, p_value = ttest_ind(Duration_Yes, Duration_No)

print(f'''t_stas & p_value are= {round(t_stats,2),round(p_value,4)}''')
print(shapiro(Duration_Yes))
print("")
print(shapiro(Duration_No))
bank_df.groupby(["Target"]).count()  #The data set is skewed in terms of target column.
plt.hist(bank_df['duration']) # High outliers in duration attribute
bank_df['duration_cbrt'] = np.cbrt(bank_df['duration']) # can be treated by square root/cube root
plt.hist(bank_df['duration_cbrt']) # Below histogram is much better distributed than the above distribution.
bank_df.head()
# Defining Dependent & Independent Variables for model inputs

X = bank_df.drop(['Target', 'duration_cbrt'], axis=1)  # Independent Variable
y = bank_df['Target']  # Dependent Variable
X.head()
y.head()
y.head()
bank_df['Target'].value_counts()
# Lets explore Ratio of "Yes" to "No" in dependent variable "Target"

print(f'''(% Distribution of Target Variable)
{round(bank_df['Target'].value_counts(normalize=True)*100,3)}''')
# Implementing oversampling for handling data imbalanced

from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(random_state=42) # Random oversampling using imblearn library
X_res, y_res=os.fit_sample(X,y)
print('Shape of X=',X.shape)
print("")
print('Shape of y=',y.shape)
print("")
print('Shape of X_res=',X_res.shape)
print("")
print('Shape of y_res=',y_res.shape)
# Lets Check new Ratio of "Yes" to "No" in dependent variable after oversampling

print(f'''(% Distribution of Target Variable)
{round(y_res.value_counts(normalize=True)*100,3)}''')
X = pd.get_dummies(X, drop_first=True) # Converting categorical independent attributes to numbers using dummy method.
y = pd.get_dummies(y, drop_first=True)  # Converting categorical dependent vriables to number format using dummy method.
X_res = pd.get_dummies(X_res, drop_first=True) # Converting resampled categorical independent attributes to numbers using dummy method.
y_res = pd.get_dummies(y_res, drop_first=True) # Converting resampled categorical dependent vriables to number format using dummy method.
print(X.head())
print("")
print(X_res.head())
print("")
print(y.head())
print("")
print(y_res.head())
# Import train-test model for spliting data into 70:30 ratio

from sklearn.model_selection import train_test_split

#Training & testing data with original Data set
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.30, random_state=42)

#Training & testing data after oversampling technique
X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(X_res, y_res, test_size=0.30, random_state=42)
print("{0:0.2f}% data is in training set".format((len(X_train)/len(bank_df.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_test)/len(bank_df.index)) * 100))
print("Original Data Set Distribution")
print("")

print("Original Target Column 'Yes' Values    :{0}({1:0.2f}%)".format(len(bank_df.loc[bank_df['Target']=='yes']), (len(bank_df.loc[bank_df['Target']=='yes'])/len(bank_df.index))*100))
print("Orinical Target Column 'No' Values   :{0}({1:0.2f}%)".format(len(bank_df.loc[bank_df['Target']=='no']), (len(bank_df.loc[bank_df['Target']=='no'])/len(bank_df.index))*100))
print("")
print ("Training Target Column 'Yes' Values   :{0}({1:0.2f}%)".format(len(y_train.loc[y_train['yes']== 1]),(len(y_train.loc[y_train['yes']==1])/len(y_train.index))*100))
print ("Training Target Column 'No' Values  :{0}({1:0.2f}%)".format(len(y_train.loc[y_train['yes']==0]),(len(y_train.loc[y_train['yes']==0])/len(y_train.index))*100))
print("")
print ("Testing Target Column 'Yes' Values    :{0}({1:0.2f}%)".format(len(y_test.loc[y_test['yes']==1]),(len(y_test.loc[y_test['yes']==1])/len(y_test.index))*100))
print ("Testing Target Column 'No' Values   :{0}({1:0.2f}%)".format(len(y_test.loc[y_test['yes']==0]),(len(y_test.loc[y_test['yes']==0])/len(y_test.index))*100))

print("")
print("Scaled Data Set Distribution after Oversampling")
print("")
print("Sacled Target Column 'Yes' Values    :{0}({1:0.2f}%)".format(len(y_res.loc[y_res['yes']==1]), (len(y_res.loc[y_res['yes']==1])/len(y_res.index))*100))
print("Scaled Target Column 'No' Values   :{0}({1:0.2f}%)".format(len(y_res.loc[y_res['yes']==0]), (len(y_res.loc[y_res['yes']==0])/len(y_res.index))*100))
print("")
print ("Scaled Training Target Column 'Yes' Values   :{0}({1:0.2f}%)".format(len(y_res_train.loc[y_res_train['yes']==1]),(len(y_res_train.loc[y_res_train['yes']==1])/len(y_res_train.index))*100))
print ("Scaled Training Target Column 'No' Values  :{0}({1:0.2f}%)".format(len(y_res_train.loc[y_res_train['yes']==0]),(len(y_res_train.loc[y_res_train['yes']==0])/len(y_res_train.index))*100))
print("")
print ("Scaled Testing Target Column 'Yes' Values    :{0}({1:0.2f}%)".format(len(y_res_test.loc[y_res_test['yes']==1]),(len(y_res_test.loc[y_res_test['yes']==1])/len(y_res_test.index))*100))
print ("Scaled Testing Target Column 'No' Values   :{0}({1:0.2f}%)".format(len(y_res_test.loc[y_res_test['yes']==0]),(len(y_res_test.loc[y_res_test['yes']==0])/len(y_res_test.index))*100))

# To model the Naive bayes classifier import Bernoulli, Multinomial & Gaussian classifiers

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


# To calculate the accuracy score of the model, report & build confusion metrics
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.metrics import confusion_matrix
# Implement Classifiers to build the Model using Bernoulli Classifier on Original Data set
Ber = BernoulliNB()
Ber.fit(X_train, y_train) # Model Trained using training Data
y_pred = Ber.predict(X_test) # Model is ready for predictions based on test Data
print("Accuracy of the Bernoulli NB is  :({:0.2f}%)".format(accuracy_score(y_pred, y_test)*100)) # Accuracy of our Bernoulli Naive Bayes model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
# Now we will use the same Model on Scaled Data Set to observe the difference.

Ber_Scaled = BernoulliNB()
Ber_Scaled.fit(X_res_train, y_res_train) # Model Trained using Scaled training Data
y_scaled_pred = Ber_Scaled.predict(X_res_test) # Model is ready for predictions
print("Accuracy of the Bernoulli NB is  :({:0.2f}%)".format(accuracy_score(y_scaled_pred, y_res_test)*100)) # Accuracy of our Bernoulli Naive Bayes model
print(confusion_matrix(y_res_test, y_scaled_pred)) # Confusion metrix of Model
print(classification_report(y_res_test, y_scaled_pred)) # Classification Report of Bernoulli NB Model
# Implement Classifiers to build the Model using Gaussian Classifier
Gau = GaussianNB()
Gau.fit(X_train, y_train) # Model Trained using training Data.
y_pred1 = Gau.predict(X_test) # Model is ready to predict
print(f'''Accuracy of Gaussian NB is = {round(accuracy_score(y_test,y_pred1)*100,2)}%''')
print (confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
# Import neccessary libraries for logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
# Build logistic regression model

logitR = LogisticRegression(solver='liblinear')

logitR.fit(X_train, y_train)
y_pred2 = logitR.predict(X_test) # Logistic model is ready for predictions
print(f'''Accuracy of Logistic Regression Model is = {round(accuracy_score(y_test,y_pred2),2)*100}%''')
# Interpretation of R^2 in logistic regression model

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
# Confusion metrics of logistic regression model

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
# AUC, ROC Curve for logistic Regression

logit_roc_auc = roc_auc_score(y_test,y_pred2)
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
# Import supporting libraries for the model

from sklearn.neighbors import KNeighborsClassifier
# Finding the optimal value of K is very important in this algorithm, hence lets find best value of K first.
scores = []
for k in range (1,100):
    knn=KNeighborsClassifier(n_neighbors=k, weights='distance')
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
#Use K=29 as the final model for prediction
knnfinal = KNeighborsClassifier(n_neighbors = 29)

# fitting the model
knnfinal.fit(X_train, y_train)

# predict the response
y_pred4 = knnfinal.predict(X_test)
print(f'''Accuracy of KNN Model is = {round(accuracy_score(y_test,y_pred4),2)*100}%''')
# Confusion metrics of KNN model
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test, y_pred4))
# Import SVM library for Model building

from sklearn.svm import SVC

svc =SVC()
svc.fit(X_train, y_train) # SVM Model Training

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
y_pred5 = svc.predict(X_test) # SVM model is ready to predict the results
print(f'''Accuracy of SVM Model is ={round(accuracy_score(y_test,y_pred5),2)*100}%''')
print(confusion_matrix(y_test, y_pred5))
print(classification_report(y_test, y_pred5))
# Let us check how SVM performs over scaled data

svc.fit(X_res_train, y_res_train) # SVM Model training on Scaled Data

print("Accuracy on training set: {:.2f}".format(svc.score(X_res_train, y_res_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_res_test, y_res_test)))
y_scaled_pred1=svc.predict(X_res_test)
print(classification_report(y_res_test, y_scaled_pred1))
# Import Tree library to build the model

from sklearn.tree import DecisionTreeClassifier
# Create decision Tree using "entropy Method" of finding the split columns.

dTree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dTree_entropy.fit(X_train, y_train)
# Create decision Tree using "gini Index" of finding the split columns.

dTree_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dTree_gini.fit(X_train, y_train)
# Performance check of Decision Tree model on traing & testing Data

print(f'''Accuracy of Decision Tree for Training Data Using Entropy Criterion = {round(dTree_entropy.score(X_train, y_train),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Testing Data Using Entropy Criterion = {round(dTree_entropy.score(X_test, y_test),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Training Data Using Gini Index = {round(dTree_gini.score(X_train, y_train),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Testing Data Using Gini Index = {round(dTree_gini.score(X_test, y_test),4)*100}%''')
# Pruned decision Tree with Entropy method

dTree_entropy_pruned = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=4, min_samples_leaf=5)
dTree_entropy_pruned.fit(X_train, y_train)
# Pruned decision Tree with Gini method

dTree_gini_pruned = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=4, min_samples_leaf=5)
dTree_gini_pruned.fit(X_train, y_train)
y_pred6 = dTree_entropy_pruned.predict(X_test) # Entropy based Decision Tree is ready to Predict now
y_pred7 = dTree_gini_pruned.predict(X_test) # Gini Criterion based Decision Tree is ready for prediction
# Performance check of Decision Tree model on traing & testing Data after Pruning

print(f'''Accuracy of Decision Tree for Training Data Using Entropy Criterion after Pruning = {round(dTree_entropy_pruned.score(X_train, y_train),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Testing Data Using Entropy Criterion after Pruning = {round(dTree_entropy_pruned.score(X_test, y_test),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Training Data Using Gini Index after Pruning = {round(dTree_gini_pruned.score(X_train, y_train),4)*100}%''')
print("")
print(f'''Accuracy of Decision Tree for Testing Data Using Gini Index after Pruning = {round(dTree_gini_pruned.score(X_test, y_test),4)*100}%''')
print(classification_report(y_test, y_pred7))
print(confusion_matrix(y_test, y_pred7))
# Import supporting packages for Tree visualization

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
import graphviz
from sklearn import tree
from os import system

dTree_Visual  = X.columns
train_char_label = ['No', 'Yes']
Visual_Tree_File = open('credit_tree_pruned.dot','w')
dot_data = tree.export_graphviz(dTree_gini_pruned, out_file=Visual_Tree_File, feature_names = dTree_Visual, 
                                class_names = list(train_char_label), special_characters=True, filled=True, rounded=True)
Visual_Tree_File.close()
retCode = system("dot -Tpng credit_tree_pruned.dot -o credit_tree_pruned.png")
if(retCode>0):
    print("system command returning error: "+str(retCode))
else:
    display(Image("credit_tree_pruned.png"))
# importance of features in the tree building ( The importance of a feature is computed as the (normalized) 
# total reduction of the criterion brought by that feature. It is also known as the Gini importance )

feat_imp_df = pd.DataFrame(dTree_gini_pruned.feature_importances_, columns = ['imp'], index=X_train.columns)
feat_imp_df.sort_values(by=['imp'], ascending=False)
# Import library to build random forest model

from sklearn.ensemble import RandomForestClassifier

Random_forest = RandomForestClassifier(criterion='gini', min_samples_leaf=10,n_estimators=100, random_state=42)
Random_forest= Random_forest.fit(X_train, y_train) # Model is trained using Random Forest Model


y_pred8 = Random_forest.predict(X_test)  # Model is ready for predictions.
print(f'''Accuracy of Random Forest Model on Training Data = {round(Random_forest.score(X_train, y_train),5)*100}%''')
print(f'''Accuracy of Random Forest Model on Testing Data = {round(accuracy_score(y_test, y_pred8),4)*100}%''')
print(classification_report(y_test, y_pred8))
print(confusion_matrix(y_test, y_pred8))
# Import supporting packages

from sklearn.ensemble import AdaBoostClassifier

Ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
Ada_boost = Ada_boost.fit(X_train, y_train) # Model is trained using Ada boost method


y_pred9 = Ada_boost.predict(X_test) # Model is ready for predictions
print(f'''Accuracy of Ada Boost Model on Training Data = {round(Ada_boost.score(X_train, y_train),5)*100}%''')
print(f'''Accuracy of Ada Boost Model on Testing Data = {round(accuracy_score(y_test, y_pred9),4)*100}%''')
print(classification_report(y_test, y_pred9))
print(confusion_matrix(y_test, y_pred9))
# Import Neccessary library for Gradient boost classifier

from sklearn.ensemble import GradientBoostingClassifier

Gra_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.50, random_state=42)
Gra_boost = Gra_boost.fit(X_train, y_train) # Gradient Boost classifier is trained now.

y_pred10 = Gra_boost.predict(X_test) # Model is ready to predict the values based on Gradient Boost classifier
print(f'''Accuracy of Gradient Boost Model on Training Data = {round(Gra_boost.score(X_train, y_train),3)*100}%''')
print(f'''Accuracy of Gradient Boost Model on Testing Data = {round(accuracy_score(y_test, y_pred10),4)*100}%''')
print(classification_report(y_test, y_pred10))
print(confusion_matrix(y_test, y_pred10))
# Import packages to load bagging classifier

from sklearn.ensemble import BaggingClassifier

Bagging = BaggingClassifier(n_estimators=50, bootstrap=True, oob_score=True, random_state=42, max_samples=.7)
Bagging = Bagging.fit(X_train, y_train) # Model is trained based on traing Data

y_pred11 = Bagging.predict(X_test) # Bagging classifier is ready for prediction now.
print(f'''Accuracy of Gradient Boost Model on Training Data = {round(Bagging.score(X_train, y_train),3)*100}%''')
print(f'''Accuracy of Gradient Boost Model on Testing Data = {round(accuracy_score(y_test, y_pred11),4)*100}%''')
print(classification_report(y_test, y_pred11))
print(confusion_matrix(y_test, y_pred11))
# Import Libraries

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
Performance_df1 = pd.DataFrame({'Model Name':['Bernoulli NB'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred)],
                              'Recall':[recall_score(y_test,y_pred)], 'Precision':[precision_score(y_test,y_pred)],
                              'F1-Score':[f1_score(y_test,y_pred)]})
# Will create dataframe for each model for evaluating performance

temp_df1 = pd.DataFrame({'Model Name':['Gaussian NB'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred1)],
                              'Recall':[recall_score(y_test,y_pred1)], 'Precision':[precision_score(y_test,y_pred1)],
                              'F1-Score':[f1_score(y_test,y_pred1)]})

temp_df2 = pd.DataFrame({'Model Name':['Logistic Regression'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred2)],
                              'Recall':[recall_score(y_test,y_pred2)], 'Precision':[precision_score(y_test,y_pred2)],
                              'F1-Score':[f1_score(y_test,y_pred2)]})

temp_df3 = pd.DataFrame({'Model Name':['KNN'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred4)],
                              'Recall':[recall_score(y_test,y_pred4)], 'Precision':[precision_score(y_test,y_pred4)],
                              'F1-Score':[f1_score(y_test,y_pred4)]})

temp_df4 = pd.DataFrame({'Model Name':['SVM'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred5)],
                              'Recall':[recall_score(y_test,y_pred5)], 'Precision':[precision_score(y_test,y_pred5)],
                              'F1-Score':[f1_score(y_test,y_pred5)]})

temp_df5 = pd.DataFrame({'Model Name':['Decision Tree'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred7)],
                              'Recall':[recall_score(y_test,y_pred7)], 'Precision':[precision_score(y_test,y_pred7)],
                              'F1-Score':[f1_score(y_test,y_pred7)]})

temp_df6 = pd.DataFrame({'Model Name':['Random Forest'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred8)],
                              'Recall':[recall_score(y_test,y_pred8)], 'Precision':[precision_score(y_test,y_pred8)],
                              'F1-Score':[f1_score(y_test,y_pred8)]})

temp_df7 = pd.DataFrame({'Model Name':['Ada Boost'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred9)],
                              'Recall':[recall_score(y_test,y_pred9)], 'Precision':[precision_score(y_test,y_pred9)],
                              'F1-Score':[f1_score(y_test,y_pred9)]})

temp_df8 = pd.DataFrame({'Model Name':['Gradient Boost'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred10)],
                              'Recall':[recall_score(y_test,y_pred10)], 'Precision':[precision_score(y_test,y_pred10)],
                              'F1-Score':[f1_score(y_test,y_pred10)]})

temp_df9 = pd.DataFrame({'Model Name':['Bagging Classifier'],'Testing Data Accuracy':[accuracy_score(y_test,y_pred11)],
                              'Recall':[recall_score(y_test,y_pred11)], 'Precision':[precision_score(y_test,y_pred11)],
                              'F1-Score':[f1_score(y_test,y_pred11)]})
# Now combine all data frames to one in order to compare the peformance

Final_df = pd.concat([Performance_df1, temp_df1, temp_df2, temp_df3, temp_df4, temp_df5, temp_df6, temp_df7, temp_df8, temp_df9])


Final_df # Print the final data frame for comparision

round(Final_df[['Testing Data Accuracy', 'Recall','Precision','F1-Score']],4)*100 # Rounded values of Performance
# Import Neccessary library for Gradient boost classifier

from sklearn.ensemble import GradientBoostingClassifier

Gra_boost_Scaled = GradientBoostingClassifier(n_estimators=100, learning_rate=0.50, random_state=42)
Gra_boost_Scaled = Gra_boost_Scaled.fit(X_res_train, y_res_train) # Gradient Boost classifier is trained now on Scaled Data Set.

y_scaled_train = Gra_boost_Scaled.predict(X_res_train)
y_scaled_pred2 = Gra_boost_Scaled.predict(X_res_test) # Model is ready to predict the values based on Gradient Boost classifier
print(f'''Accuracy of Gradient Boost Model on Scaled Training Data = {round(Gra_boost_Scaled.score(X_res_train, y_res_train),3)*100}%''')
print(f'''Accuracy of Gradient Boost Model on Scaled Testing Data = {round(accuracy_score(y_res_test, y_scaled_pred2),4)*100}%''')

print(f'''    ----------------------------------------------------
         ***Classification Report of Original Data****
    ----------------------------------------------------
  {classification_report(y_test, y_pred10)}''')


print(f'''    ----------------------------------------------------
         ***Classification Report of Scaled Data****
    ----------------------------------------------------
  {classification_report(y_res_test, y_scaled_pred2)}''')
cm_Gra = plt.cm.Greys_r # Color Scheme for confusion metrics
cm_train = confusion_matrix(y_res_train,y_scaled_train, labels=[0,1]) # Confusion metrix for Gradient Boost on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm_train_df = pd.DataFrame(cm_train, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm_train_df, annot=True, fmt='.5g', cmap=cm_Gra, linecolor='Black', square=True)

cm_Gra1 = plt.cm.Blues_r # Color Scheme for confusion metrics
cm_test = confusion_matrix(y_res_test,y_scaled_pred2, labels=[0,1]) # Confusion metrix for Gradient Boost on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm_test_df = pd.DataFrame(cm_test, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm_test_df, annot=True, fmt='.5g', cmap=cm_Gra1, linecolor='Black', square=True)
# ROC AUC Curves Gradient Boost.

Gra_roc_auc = roc_auc_score(y_res_test, y_scaled_pred2)
fpr1, tpr1, thresholds1 = roc_curve(y_res_test, y_scaled_pred2)
plt.figure()
plt.plot(fpr1, tpr1, label='Gradient (area = %0.2f)' % Gra_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve')
plt.legend(loc="lower right")
plt.show()
X_res.head()
# Let us drop the feature "Duration" from above data frame

X_res_update = X_res.drop(['duration'],axis=1)
#Build the Model using updated data & split it in 70:30 Ratio

X_res_train1, X_res_test1, y_res_train1, y_res_test1 = train_test_split(X_res_update, y_res, test_size=0.30, random_state=42)
# Gradient boost classifier for dropeed column in scaled data

Gra_boost_Scaled1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.50, random_state=42)
Gra_boost_Scaled1 = Gra_boost_Scaled1.fit(X_res_train1, y_res_train1) # Gradient Boost classifier is trained now on Scaled Data Set.

y_scaled_train_new = Gra_boost_Scaled1.predict(X_res_train1)
y_scaled_pred_new = Gra_boost_Scaled1.predict(X_res_test1) # Model is ready to predict the values based on Gradient Boost classifier
print(f'''Accuracy of Gradient Boost Model on Scaled Training Data after dropping "Duration" feature = {round(Gra_boost_Scaled1.score(X_res_train1, y_res_train1),3)*100}%''')
print(f'''Accuracy of Gradient Boost Model on Scaled Testing Data after dropping "Duration" feature  = {round(accuracy_score(y_res_test1, y_scaled_pred_new),4)*100}%''')
cm_Gra2 = plt.cm.Greens_r # Color Scheme for confusion metrics
cm_train1 = confusion_matrix(y_res_train1,y_scaled_train_new, labels=[0,1]) # Confusion metrix for Gradient boost on Training Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm_train_df1 = pd.DataFrame(cm_train1, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm_train_df1, annot=True, fmt='.5g', cmap=cm_Gra2, linecolor='Black', square=True)

cm_Gra3 = plt.cm.Reds_r # Color Scheme for confusion metrics
cm_test1 = confusion_matrix(y_res_test1,y_scaled_pred_new, labels=[0,1]) # Confusion metrix for Gradient Boost on Test Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm_test_df1 = pd.DataFrame(cm_test1, columns=[i for i in ["Actual 1", "Actual 0"]], index=[i for i in ["Predict 1","Predict 0"]])
sns.heatmap(data=cm_test_df1, annot=True, fmt='.5g', cmap=cm_Gra3, linecolor='Black', square=True)
print(f'''    ----------------------------------------------------
***Classification Report of Realistic Predictive Model****
    ----------------------------------------------------
  {classification_report(y_res_test1, y_scaled_pred_new)}''')