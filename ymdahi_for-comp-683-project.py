import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system interface
import matplotlib.pyplot as plt

# bring in data source
so_results_file_path = '../input/survey_results_public.csv'
#so_schema_file_path = '../input/survey_results_schema.csv'

# create dataframe to hold results data
df = pd.read_csv(so_results_file_path)

df = df.dropna(subset=['CareerSatisfaction'])
print ( 'Number of rows (respondants): ' , (df.shape[0]) )
print ( 'Number of columns (questions): ' , (df.shape[1]) )
df['CareerSatisfaction'].value_counts()
df['CareerSatisfaction'].value_counts().plot.pie()
# Create a pandas column from 'CareerSatisfaction' that converts the qualitative values to quantitative values
CareerSatRating = []
for row in df['CareerSatisfaction']:
    if row == 'Extremely dissatisfied':
        CareerSatRating.append(1)
    elif row == 'Moderately dissatisfied':
        CareerSatRating.append(2)
    elif row == 'Slightly dissatisfied':
        CareerSatRating.append(3)
    elif row == 'Neither satisfied nor dissatisfied':
        CareerSatRating.append(4)
    elif row == 'Slightly satisfied':
        CareerSatRating.append(5)
    elif row == 'Moderately satisfied':
        CareerSatRating.append(6)
    elif row == 'Extremely satisfied':
        CareerSatRating.append(7)
    else:
        CareerSatRating.append('Failed') # failed

df['CareerSatRating'] = CareerSatRating

df['CareerSatRating'].head(5)

columns_of_interest = ['AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5','AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10','CareerSatisfaction','CareerSatRating']
selected_columns = df[columns_of_interest].dropna()
selected_columns.columns = ['Industry', 'FinancialStatus','Frameworks','Benefits','Culture','Remote','PD','Diversity','Impact','Salary','CareerSatisfaction','CareerSatRating']
target_column = selected_columns['CareerSatRating']
prediction_columns = ['Industry', 'FinancialStatus','Frameworks','Benefits','Culture','Remote','PD','Diversity','Impact','Salary']
selected_columns[prediction_columns].describe()
selected_columns[prediction_columns].info()
selected_columns[prediction_columns].boxplot(figsize=(18,10))
selected_columns[prediction_columns].hist(figsize=(18,10))
selected_columns.groupby('CareerSatRating').hist(figsize=(18,10))
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)

selected_columns['is_train'] = np.random.uniform(0, 1, len(selected_columns)) <= .75
selected_columns.head(10)
# Create two new dataframes, one with the training rows, one with the test rows
train = selected_columns[selected_columns['is_train']==True]
test = selected_columns[selected_columns['is_train']==False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
# Create a list of the feature column's names
features = selected_columns.columns[:10]
features
# train['ConvCareerSatisfaction'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = train['CareerSatRating']
y.head()
# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the career satisfaction rating)
clf.fit(train[features], y)
print (clf.score(train[features], y))
# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
print(clf.predict(test[features]))
# View the predicted probabilities of the first 10 observations
print(clf.predict_proba(test[features])[0:10])
preds = clf.predict(test[features])
print('Predictions for first 5 elements in test df:')
print(preds[0:5])
print('Actual values for first 5 elements in test df:')
print(test['CareerSatRating'].head())
# Create confusion matrix
cm = pd.crosstab(test['CareerSatRating'], preds, rownames=['Actual CareerSatisfaction'], colnames=['Predicted CareerSatisfaction'])
cm
cm.plot(kind="bar", figsize=(8,8),stacked=True)
# View a list of the features and their importance scores
imp = list(zip(train[features], clf.feature_importances_))
imp
importances = clf.feature_importances_
importances
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train[features].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train[features].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train[features].shape[1]), indices)
plt.xlim([-1, train[features].shape[1]])
plt.show()