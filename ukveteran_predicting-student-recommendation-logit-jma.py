import pandas as pd
# turn of warning messages
pd.options.mode.chained_assignment = None  # default='warn'

# get data
df = pd.read_csv('../input/student-records/student_records.csv')
df
# get features and corresponding outcomes
feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
training_features = df[feature_names]

outcome_name = ['Recommend']
outcome_labels = df[outcome_name]
# view features
training_features
# view outcome labels
outcome_labels
# list down features based on type
numeric_feature_names = ['ResearchScore', 'ProjectScore']
categoricial_feature_names = ['OverallGrade', 'Obedient']
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# fit scaler on numeric features
ss.fit(training_features[numeric_feature_names])

# scale numeric features now
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])

# view updated featureset
training_features
training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)
# view newly engineering features
training_features
# get list of new categorical features
categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))
from sklearn.linear_model import LogisticRegression
import numpy as np

# fit the model
lr = LogisticRegression() 
model = lr.fit(training_features, np.array(outcome_labels['Recommend']))
# view model parameters
model
# simple evaluation on training data
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels['Recommend'])

# evaluate model performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print('Accuracy:', float(accuracy_score(actual_labels, pred_labels))*100, '%')
print('Classification Stats:')
print(classification_report(actual_labels, pred_labels))