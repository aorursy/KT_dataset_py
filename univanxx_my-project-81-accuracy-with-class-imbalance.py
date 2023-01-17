import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting
import seaborn as sns 
import matplotlib.pyplot as plt

# data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as imb_pipeline

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',usecols = lambda column : column not in 
["customerID"])
df.head()
df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
empty_values = []
for i in range(len(df['TotalCharges'])):
    if df['TotalCharges'].iloc[i] == ' ':
        empty_values.append(i)
print("There are empty indexes found:", end=' ')
print(empty_values)
for i in range(len(empty_values)):
    print(df.iloc[empty_values[i]])
df["TotalCharges"] =  df["TotalCharges"].replace(r' ', '0')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.info()
for i in df.columns:
    if i not in ['tenure','MonthlyCharges','TotalCharges']:
        print(i,'column has',len(pd.unique(df[i])),'unique values or rather:')
        print(pd.unique(df[i]))
plt.figure(figsize=(10,6))

plt.title("Churn chart")

sns.countplot(df['Churn'])
churn_yes = df[df.Churn == "Yes"].shape[0]
churn_no = df[df.Churn == "No"].shape[0]

churn_yes_percent = round((churn_yes / (churn_yes + churn_no) * 100),2)
churn_no_percent = round((churn_no / (churn_yes + churn_no) * 100 ),2)

print('There are',churn_yes_percent,'percent of customers that will churn and',churn_no_percent,'percent of customers that will not churn')
categorial_columns = [cname for cname in df.columns if cname not in ['tenure','MonthlyCharges','TotalCharges','Churn']]

print("Our categorial columns:", categorial_columns)
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

# Creating train and test subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# One Hot Encoding
X_train = pd.get_dummies(X_train_full)
X_valid = pd.get_dummies(X_valid_full)

# For y-values we will use LabelEncoder

label_enc  = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_valid = label_enc.fit_transform(y_valid)
X_train.head()
from sklearn.preprocessing import RobustScaler

# I use RobustScaler because it's quite robust to outliers

rob_scaler = RobustScaler()

columns_to_scale = ['tenure','MonthlyCharges','TotalCharges']

X_train[columns_to_scale] = rob_scaler.fit_transform(X_train[columns_to_scale])
X_valid[columns_to_scale] = rob_scaler.fit_transform(X_valid[columns_to_scale])
X_valid.head()
# Use GridSearchCV to find the best parameters.
# from sklearn.model_selection import GridSearchCV

# Logistic Regression 
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_valid)
# Calculate accuracy
ACC = accuracy_score(y_valid, predictions)
print(ACC)
log_reg_cf = confusion_matrix(y_valid, predictions)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))

sns.heatmap(log_reg_cf, annot=True, cmap=plt.cm.Pastel1)
plt.title("Logistic Regression Confusion Matrix", fontsize=14)
plt.xlabel("Predicted classes")
plt.ylabel("Actual classes")

plt.show()
from imblearn.under_sampling import NearMiss

undersample_pipeline = make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
undersample_model = undersample_pipeline.fit(X_train, y_train)
undersample_predictions = undersample_model.predict(X_valid)
# Calculate accuracy
ACC = accuracy_score(y_valid, undersample_predictions)
print(ACC)
log_reg_cf = confusion_matrix(y_valid, undersample_predictions)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))

sns.heatmap(log_reg_cf, annot=True, cmap=plt.cm.Pastel1)
plt.title("Logistic Regression Confusion Matrix", fontsize=14)
plt.xlabel("Predicted classes")
plt.ylabel("Actual classes")

plt.show()
from imblearn.over_sampling import SMOTE

# I use other solver and increase numer of iterations because our dataset will become larger
oversample_pipeline = make_pipeline(SMOTE(sampling_strategy='minority'), LogisticRegression(solver = 'saga', max_iter=10000))
oversample_model = oversample_pipeline.fit(X_train, y_train)
oversample_predictions = oversample_model.predict(X_valid)
# Calculate accuracy
ACC = accuracy_score(y_valid, oversample_predictions)
print(ACC)
log_reg_cf = confusion_matrix(y_valid, oversample_predictions)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))

sns.heatmap(log_reg_cf, annot=True, cmap=plt.cm.Pastel1)
plt.title("Logistic Regression Confusion Matrix", fontsize=14)
plt.xlabel("Predicted classes")
plt.ylabel("Actual classes")

plt.show()
# I use Grid Search to find best parameters for our model
from sklearn.model_selection import GridSearchCV

#Creating pipeline with data augmentation and subsequent regression
pipeline = imb_pipeline(
                    [('nearmiss', SMOTE(sampling_strategy='minority')),
                     ('logreg', LogisticRegression(solver = 'saga', max_iter=10000))
                     
])

parameters = {}
parameters['logreg__penalty'] = ['l1', 'l2']
parameters['logreg__C'] = [i for i in range(80,420,40)]

CV = GridSearchCV(pipeline, parameters, scoring = 'accuracy', n_jobs= 1)
CV.fit(X_train, y_train)   

print('Best parameter combination for linear regression is:', CV.best_params_)
oversample_pipeline = make_pipeline(SMOTE(sampling_strategy='minority'), LogisticRegression(solver = 'saga', penalty = 'l1', C=120, max_iter=10000))
oversample_model = oversample_pipeline.fit(X_train, y_train)
oversample_predictions = oversample_model.predict(X_valid)

print('Accuracy on validation set: %s' % (accuracy_score(y_valid, oversample_predictions)))