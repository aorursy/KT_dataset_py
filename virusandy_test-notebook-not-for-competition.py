import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
# load the test data and seen on it
data_test = pd.DataFrame.from_csv(open('../input/test.csv', 'rb')) 
print(
    "Test data:\n", 
    data_test.head(), 
    "\n", "-" * 50
)

# load the train data
data_train = pd.DataFrame.from_csv(open('../input/train.csv', 'rb')) 
print(
    "Train data:\n", 
    data_train.head(), 
    "\n", "-" * 50
)
data_test['Survived'] = None
data_all = data_test.merge(data_train, how='outer')

print(
    "All data:\n", 
    data_all.head(), 
    "\n", "-" * 50
)
print(data_all.info())
numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
print(data_all[numeric_columns].describe(), "\n")
for col in numeric_columns:
    print(col, ' unique values: ', len(data_all[col].unique()))
