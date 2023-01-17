# Import the data processing and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Read the dataset in pandas
df_liver = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
# Access the first 5 rows of df_liver
df_liver.head()
# Access the last 5 rows of df_liver
df_liver.tail()
# Retrieve the colunmn information
df_liver.columns.values
# Retrieve the full information of df_liver regarding the features and response, in order to verify 
# if the values are unique or are there any missing data.
df_liver.info()
# Find the shape of the datframe df_liver
df_liver.shape
# We can performing some simple statistical inferences to get a good feel of the data
df_liver.describe()
# I can quickly perform some additional statistics to include all
df_liver.describe(include ='all')
# Define a function that allows us to create a table of missing values in df_liver and their percentages in 
# descending order
def missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    percentage_final = (round(percentage, 2) * 100)
    total_percent = pd.concat(objs=[total, percentage_final], axis = 1, keys=['Total', '%'])
    return total_percent
# Find the total count and % of missing values 
missing_values(df_liver)
# Replace missing values with the mean of feature column Albumin_and_Globulin_Ratio, 
# then check to see that it has been successfull, where the sum of missig values should be 0
df_liver['Albumin_and_Globulin_Ratio'].fillna(df_liver['Albumin_and_Globulin_Ratio'].mean(), inplace = True)
df_liver['Albumin_and_Globulin_Ratio'].isnull().sum()
# Repeat to see what is the % of missing values
missing_values(df_liver)
# Correlation pairplot
sns.set()
sns.pairplot(df_liver, hue='Dataset', kind='reg')
# A more robust way of figuring out correlations other than observations as above is to generate a full correlation
# table with the ranging from -1 to 1
df_liver.corr().style.background_gradient(cmap='coolwarm')
# Change the current categorical feature Gender to a numerical feature of 0 or 1 (as ML algorithms prefer numerical 
# features)
df_liver['Gender'] = df_liver['Gender'].map({'Male': 1, 'Female': 0})
# Alternatively, you can use the apply and lambda function
# df_liver['Gender'] = df_liver['Gender'].apply(lambda x:1 if x == 'Male' else 0)

# Check to make sure that the gender has been correctly converted
df_liver.head()
# Create a table for Dataset (with and without liver disease) and gender
df_liver_Gender = round(df_liver[['Gender', 'Dataset']].groupby(['Gender'], as_index=False).agg(np.sum), 3)

# Generate plot to determine the effect of gender on the dataset (target feature)
# Figure configuration
plt.figure(figsize=(10,5))

sns.barplot(x="Gender", y="Dataset", data=df_liver_Gender, ci=None)
plt.title("Survival wrt Gender")
plt.ylim(0, 700)
# Create a table for Dataset (with and without liver disease) and Albumin
df_liver_Albumin = round(df_liver[['Gender', 'Albumin', 'Dataset']]
                         .groupby(['Albumin', 'Gender'], as_index=False).agg(np.sum), 1)

# Generate plot to determine the effect of gender on the dataset (target feature) and Albumin
# Figure configuration
plt.figure(figsize=(18,10))

sns.barplot(x="Albumin", y="Dataset", hue='Gender', data=df_liver_Albumin, ci=None)
plt.title("Survival wrt Albumin conc.")
# Create a table for Dataset (with and without liver disease) and Total Proteins
df_liver_TP = round(df_liver[['Gender', 'Total_Protiens', 'Dataset']]
                    .groupby(['Total_Protiens', 'Gender'], as_index=False).agg(np.sum), 2)

# Generate plots to determine the effect of gender on the dataset (target feature) and total proteins
# Figure configuration
plt.figure(figsize=(20,15))

sns.barplot(x="Total_Protiens", y="Dataset", hue='Gender', data=df_liver_TP, ci=None)
plt.title("Survival wrt Total Proteins conc.")
# Create a table for Dataset (with and without liver disease) and Alkaline Phosphatase
df_liver_ALP = round(df_liver[['Gender', 'Albumin_and_Globulin_Ratio', 'Dataset']]
                     .groupby(['Albumin_and_Globulin_Ratio', 'Gender'], as_index=False).agg(np.sum), 1)

# Generate plots to determine the effect of gender on the dataset (target feature) and AGR
# Figure configuration
plt.figure(figsize=(18,10))

sns.barplot(x="Albumin_and_Globulin_Ratio", y="Dataset", hue='Gender', data=df_liver_ALP, ci=None)
plt.title("Survival wrt Albumin Globulin Ratio")
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Age and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Age'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Age'], 
                  bins=40, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Age'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Age'], 
                  bins=40, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male')
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Total Bilirubin  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Total_Bilirubin'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Total_Bilirubin'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Total_Bilirubin'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Total_Bilirubin'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Direct Bilirubin  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Direct_Bilirubin'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Direct_Bilirubin'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Direct_Bilirubin'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Direct_Bilirubin'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male')
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both ALP  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Alkaline_Phosphotase'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Alkaline_Phosphotase'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Alkaline_Phosphotase'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Alkaline_Phosphotase'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both AAT and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Alamine_Aminotransferase'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Alamine_Aminotransferase'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Alamine_Aminotransferase'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Alamine_Aminotransferase'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both AAT  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Aspartate_Aminotransferase'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Aspartate_Aminotransferase'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Aspartate_Aminotransferase'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Aspartate_Aminotransferase'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Total Proteins  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Total_Protiens'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Total_Protiens'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Total_Protiens'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Total_Protiens'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Albumin  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Albumin'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Albumin'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Albumin'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Albumin'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male') 
# Figure configuration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Visualize the effect of the dataset(with or without disease) based on both Albumin and Globulin Ratio  and Gender.
ld = 'Liver Disease'
no_ld = 'No Liver Disease'
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 1]['Albumin_and_Globulin_Ratio'], 
                  bins=18, label=ld, ax=axes[0], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 0][df_liver[df_liver['Gender'] == 0]['Dataset'] == 2]['Albumin_and_Globulin_Ratio'], 
                  bins=20, label=no_ld, ax=axes[0], kde=False, color='red')
ax.legend()
ax.set_title('Female')
ax.set_ylabel('Counts')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 1]['Albumin_and_Globulin_Ratio'], 
                  bins=18, label=ld, ax=axes[1], kde=False, color='blue')
ax = sns.distplot(df_liver[df_liver['Gender'] == 1][df_liver[df_liver['Gender'] == 1]['Dataset'] == 2]['Albumin_and_Globulin_Ratio'], 
                  bins=20, label=no_ld, ax=axes[1], kde=False, color='red')
ax.legend()
ax.set_title('Male')
# Create a new dataframe for the simple hypothesis testing 
df_liver_hyp = df_liver
df_liver_hyp.head()
# Create a 'Hypothesis' column and set that equal to 0
df_liver_hyp['Hypothesis'] = 0

# Our hypothesis is that if the patients have liver disease then set the hypothesis column to 1
df_liver_hyp.loc[df_liver_hyp['Dataset'] == 1, 'Hypothesis'] = 1

# Next, to check if our hypothesis is correct I will create another column called Result and set that equal to 0
df_liver_hyp['Result'] = 0

# If the Dataset column agrees with our Hypothesis column, I am going to update the 'Result' column by 1.
df_liver_hyp.loc[df_liver_hyp['Dataset'] == df_liver_hyp['Hypothesis'], 'Result'] = 1

df_liver_hyp.head()
# I will now find the percentage of passengers that have liver disease
round(df_liver_hyp['Result'].value_counts(normalize=True) * 100, 3)
# Machine learning libraries in sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Cross validate model with Kfold stratified cross val
# K-fold cross validation: randomly splits the training set into (n_splits) 10 distinct subsets called folds, 
# then it trains and evaluates the models 10 times, picking a different fold for evaluation every time and 
# training on the other 9 folds.
K_fold = StratifiedKFold(n_splits=10)
# Separate train features and response
X = df_liver.drop(["Dataset", "Hypothesis", "Result"],axis = 1)
Y = df_liver["Dataset"]

# It turns out the I get the error message of reaching the total number of iterations reached the limit. 
# In this case I may need to scale the data

# Scale the data
scaler=MinMaxScaler()
scaled_values=scaler.fit_transform(X)
X.loc[:,:]=scaled_values

# Create the Train and Test sets
# Splitting the train and test into 70% training and 30% testing
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,stratify=Y, test_size=0.3,random_state=42)


# Find the shape of all sets
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
# Logistic Regression
model_logreg = LogisticRegression()
model_logreg.fit(X_train,Y_train)
y_pred = model_logreg.predict(X_test)

scores = cross_val_score(model_logreg, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_logreg = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_logreg))
acc_logreg = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_logreg))
# K-Neighbors Classifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, Y_train)
y_pred = model_knn.predict(X_test)

scores = cross_val_score(model_knn, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_knn = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_knn))
acc_knn = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_knn))
model_MLP = MLPClassifier()
model_MLP.fit(X_train, Y_train)
y_pred = model_MLP.predict(X_test)

scores = cross_val_score(model_MLP, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy' )

print(scores)
score_MLP = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_MLP))
acc_MLP = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_MLP))
# Decision Tree Classifer
model_dtc = DecisionTreeClassifier()
model_dtc.fit(X_train, Y_train)
y_pred = model_dtc.predict(X_test)

scores = cross_val_score(model_dtc, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_dtc = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_dtc))
acc_dtc = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_dtc))
# Random Forest Classifier
model_rfc = RandomForestClassifier(n_estimators=50)
model_rfc.fit(X_train, Y_train)
y_pred = model_rfc.predict(X_test)

scores = cross_val_score(model_rfc, X_train, Y_train, cv=K_fold, n_jobs=4, scoring ='accuracy')

print(scores)
score_rfc = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_rfc))
acc_rfc = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_rfc))
# Gaussian Naive Bayes
model_gaussNB = GaussianNB()
model_gaussNB.fit(X_train, Y_train)
y_pred = model_gaussNB.predict(X_test)

scores = cross_val_score(model_gaussNB, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_gaussNB = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_gaussNB))
acc_gaussNB = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_gaussNB))
# Support vector classification
model_SVC = SVC()
model_SVC.fit(X_train, Y_train)
y_pred = model_SVC.predict(X_test)

scores = cross_val_score(model_SVC, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_SVC = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_SVC))
acc_SVC = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_SVC))
# Gradient Boosting Classifier
model_GBC = GradientBoostingClassifier()
model_GBC.fit(X_train, Y_train)
y_pred = model_GBC.predict(X_test)

scores = cross_val_score(model_GBC, X_train, Y_train, cv=K_fold, n_jobs=4, scoring='accuracy')

print(scores)
score_GBC = round(np.mean(scores) * 100, 3)
print("Score: {}".format(score_GBC))
acc_GBC = round(np.mean(accuracy_score(Y_test, y_pred)) * 100, 3)
print("Accuracy: {}".format(acc_GBC))
results = pd.DataFrame({'Model': ['Logistic Regression','KNeighborsClassifer', 'MLP Classifier', 
                                  'Decision Tree Classifier', 'Random Forest Classifier', 'GaussianNB', 'SVC', 
                                  'GB Classifier'],
                        'Accuracy': [acc_logreg, acc_knn, acc_MLP, acc_dtc, acc_rfc, acc_gaussNB, 
                                  acc_SVC, acc_GBC], 
                        'Score': [score_logreg, score_knn, score_MLP, score_dtc, score_rfc, score_gaussNB, 
                                  score_SVC, score_GBC],})
df_results = results.sort_values(by='Score', ascending=False)
df_results = df_results.set_index('Score')
df_results
