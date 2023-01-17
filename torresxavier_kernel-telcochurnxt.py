import pandas as pd

import numpy as np
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
df.isna().sum()
df.isnull().sum()
df.dtypes

df["TotalCharges"].describe()
df.loc[df["TotalCharges"] == " "]
df.loc[(df["tenure"] == 0) & (df["TotalCharges"] == " ")]
df.loc[df["TotalCharges"] == " ", "TotalCharges"] = "0"
df.loc[(df["tenure"] == 0) & (df["TotalCharges"] == "0")]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df.describe()
df["TotalCharges"].describe()
df.describe()   #describes numerical variables
df.describe(include='object')  #describes categorical variables using the 'include' property
df.corr()
for col in df:

    if (col != "customerID"):

        if df[col].dtype == "object":

            total_uniques= df[col].nunique()

            uniques_list = set(df[col])

            print (f"{col}, uniques= { total_uniques}, list = {uniques_list}")
def change_category(col):

    if (col != "customerID") & (col != "InternetService")  & (col != "Contract") & (col != "PaymentMethod"):

        if df[col].dtype == "object":

            df[col].replace(('Male','Female'), ("0", "1"), inplace=True)    # works for gender, assigning strings (we're still working with object type columns...)

            df[col].replace(('No','Yes','No internet service'), ("0", "1", "2"), inplace=True)  # works for other columns except MultipleLines

            df[col].replace(('No','Yes','No phone service'), ("0", "1", "2"), inplace=True)  # works for other columns

            df[col] = df[col].astype(int)    # change variable type from string to int

            

          
for col in df:

        change_category(col)

        
df.head(5)
df.dtypes
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(16,12))

ax = plt.subplot(111)

sns.heatmap(df.corr(), annot=True, cmap="Greens", ax=ax)

bottom, top = ax.get_ylim()             # this fixes a bug where+ first and last columns were not visualized correctly

ax.set_ylim(bottom + 0.5, top - 0.5)
for col in df:

        aggregate = df.groupby([col, "Churn"]).size()   # size of observation for each category

        print(aggregate)

        print("_________________________________________")
df.loc[df["InternetService"] == "No", "InternetService"] = "0"

df.loc[df["InternetService"] == "DSL", "InternetService"] = "1"

df.loc[df["InternetService"] == "Fiber optic", "InternetService"] = "2"



df.loc[df["Contract"] == "Month-to-month", "Contract"] = "0"

df.loc[df["Contract"] == "One year", "Contract"] = "1"

df.loc[df["Contract"] == "Two year", "Contract"] = "2"



df.loc[df["PaymentMethod"] == "Electronic check", "PaymentMethod"] = "0"

df.loc[df["PaymentMethod"] == "Credit card (automatic)", "PaymentMethod"] = "1"

df.loc[df["PaymentMethod"] == "Bank transfer (automatic)", "PaymentMethod"] = "2"

df.loc[df["PaymentMethod"] == "Mailed check", "PaymentMethod"] = "3"
df["InternetService"].head(15)
df["InternetService"] = df["InternetService"].astype('int32')

df["Contract"] = df["Contract"].astype('int32')

df["PaymentMethod"] = df["PaymentMethod"].astype('int32')
df.dtypes
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#split dataset in features and target variable

feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure','PhoneService','MultipleLines',

               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport',

                'StreamingTV','StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod',

                'MonthlyCharges','TotalCharges']

X = df[feature_cols] # Features

y = df["Churn"] # Target variable
print(X.head(10))

print("_____________________________")

print(y.head(10))
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

x=metrics.accuracy_score(y_test, y_pred)

print("Accuracy:"+"{:.2%}".format(x))
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth=4)  # We establish the tree to a maximum depth of 4 levels



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

x=metrics.accuracy_score(y_test, y_pred)

print("Accuracy:"+"{:.2%}".format(x))
from sklearn.metrics import confusion_matrix
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))  # true in rows and predicted in columns

print(conf_matrix)



precision_no_churn = conf_matrix.loc[0,0]/(conf_matrix.loc[0,0]+conf_matrix.loc[1,0])

print()

print("Precision No churn = "+"{:.1%}".format(precision_no_churn))



precision_churn = conf_matrix.loc[1,1]/(conf_matrix.loc[0,1]+conf_matrix.loc[1,1])

print()

print("Precision churn = "+"{:.1%}".format(precision_churn))
df.groupby("Churn").size()
from sklearn.utils import resample
# Separate majority and minority classes

df_majority = df[df.Churn == 0]

df_minority = df[df.Churn == 1]
# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=5174,    # to match majority class

                                 random_state=456) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

df_upsampled.Churn.value_counts()

# 1    576

# 0    576

# Name: balance, dtype: int64
X = df_upsampled[feature_cols] # Features

y = df_upsampled["Churn"] # Target variable# Split dataset into training set and test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth=4)   #Let's keep a 4 depth level tree



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

x=metrics.accuracy_score(y_test, y_pred)

print("Accuracy:"+"{:.2%}".format(x))
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))  # true in rows and predicted in columns

print(conf_matrix)



precision_no_churn = conf_matrix.loc[0,0]/(conf_matrix.loc[0,0]+conf_matrix.loc[1,0])

print()

print("Precision No churn = "+"{:.1%}".format(precision_no_churn))



precision_churn = conf_matrix.loc[1,1]/(conf_matrix.loc[0,1]+conf_matrix.loc[1,1])

print()

print("Precision churn = "+"{:.1%}".format(precision_churn))



recall_no_churn = conf_matrix.loc[0,0]/(conf_matrix.loc[0,0]+conf_matrix.loc[0,1])

print()

print("Recall No churn, % of observations found = "+"{:.1%}".format(recall_no_churn))



recall_churn = conf_matrix.loc[1,1]/(conf_matrix.loc[1,0]+conf_matrix.loc[1,1])

print()

print("Recall churn, % of observations found  = "+"{:.1%}".format(recall_churn))
from sklearn.tree.export import export_text
tree_rules = export_text(clf, feature_names=list(X_train))
print(tree_rules)