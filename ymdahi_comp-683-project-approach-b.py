import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system interface
import matplotlib.pyplot as plt

# bring in data source
so_results_file_path = '../input/survey_results_public.csv'
#so_schema_file_path = '../input/survey_results_schema.csv'

# create dataframe to hold results data
df = pd.read_csv(so_results_file_path)

df = df.dropna(subset=['JobSatisfaction'])
df['JobSatisfaction'].value_counts().plot.pie()
# Create a pandas column from 'CareerSatisfaction' that converts the qualitative values to quantitative values
JobSatRating = []
for row in df['JobSatisfaction']:
    if row == 'Extremely dissatisfied':
        JobSatRating.append(1)
    elif row == 'Moderately dissatisfied':
        JobSatRating.append(2)
    elif row == 'Slightly dissatisfied':
        JobSatRating.append(3)
    elif row == 'Neither satisfied nor dissatisfied':
        JobSatRating.append(4)
    elif row == 'Slightly satisfied':
        JobSatRating.append(5)
    elif row == 'Moderately satisfied':
        JobSatRating.append(6)
    elif row == 'Extremely satisfied':
        JobSatRating.append(7)
    else:
        JobSatRating.append('Failed') # failed

df['JobSatRating'] = JobSatRating

df['JobSatRating'].describe()
# Columns that we are interested in observing.
columns_of_interest = ['AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5','AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10','JobSatisfaction','JobSatRating']

# Drop any rows that does not have complete data for the COI above.
clean_df = df[columns_of_interest].dropna()

# Rename the columns in our COI so they are easier to read.
clean_df.columns = ['Industry', 'FinancialStatus','Frameworks','Benefits','Culture','Remote','PD','Diversity','Impact','Salary','JobSatisfaction','JobSatRating']

# The column we want to predict
target_column = ['JobSatRating']

# The columns we will use to model and make prediction
prediction_columns = ['Industry', 'FinancialStatus','Frameworks','Benefits','Culture','Remote','PD','Diversity','Impact','Salary']

# Let's tale a look at our dataframe
clean_df.head()
# Let's take a look at our prediction columns
clean_df[prediction_columns].describe()
# Box Plot for Prediction Columns
clean_df[prediction_columns].boxplot(figsize=(18,10))
clean_df[prediction_columns].hist(figsize=(18,10))
# Count and plot predictors that ranked 3 or lower. i.e. higher importance.
print (clean_df[clean_df[prediction_columns]<=3].count())
print (clean_df[clean_df[prediction_columns]<=3].count().plot.bar())
# Count and plot predictors that ranked 7 or higher. i.e lower importance
print (clean_df[clean_df[prediction_columns]>=7].count())
print (clean_df[clean_df[prediction_columns]>=7].count().plot.bar())
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)

# Randomly pick some data to be training data.
clean_df['is_train'] = np.random.uniform(0, 1, len(clean_df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
train = clean_df[clean_df['is_train']==True]
test = clean_df[clean_df['is_train']==False]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
# Remembering our prediction and target columns
print ('What we want to predict: ')
print (target_column)
print ('Factors we will consider when predicting: ')
print (prediction_columns)
clean_df.head(10)
# Create new df to hold training data
y = train[target_column]
y.head()
# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the career satisfaction rating)
clf.fit(train[prediction_columns], y)
print (clf.score(train[prediction_columns], y))
# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
print(clf.predict(test[prediction_columns]))
# View the predicted probabilities of the first 10 observations
print(clf.predict_proba(test[prediction_columns])[0:10])
preds = clf.predict(test[prediction_columns])
print('Predictions for first 5 elements in test df:')
print(preds[0:5])
print('Actual values for first 5 elements in test df:')
print(test['JobSatRating'].head())
# Create confusion matrix
cm = pd.crosstab(test['JobSatRating'], preds, rownames=['Actual JobSatisfaction'], colnames=['Predicted JobSatisfaction'])
cm
cm.plot.bar(figsize=(18,10))
cm.plot(kind="bar", figsize=(8,8),stacked=True)
# View a list of the features and their importance scores
imp = list(zip(train[prediction_columns], clf.feature_importances_))
imp
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train[prediction_columns].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train[prediction_columns].shape[1]), importances[indices],
       color="grey", yerr=std[indices], align="center")
plt.xticks(range(train[prediction_columns].shape[1]), indices)
plt.xlim([-1, train[prediction_columns].shape[1]])
plt.show()