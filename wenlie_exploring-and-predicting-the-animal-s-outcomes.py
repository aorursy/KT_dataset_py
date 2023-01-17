import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
%matplotlib inline

sns.set(style="white", color_codes=True)
data = pd.read_csv('../input/aac_shelter_outcomes.csv')
data_cols = data.columns
data.info()
print(data.iloc[0])
data.nunique()
data.groupby(['outcome_type','outcome_subtype']).size()
# Lets start taking a loot at the possible outcomes
plt.figure(figsize=(12,4))
sns.countplot(y=data['outcome_type'], 
              palette='rainbow',
              order=data['outcome_type'].value_counts().index)
plt.show()
# plt.figure(figsize=(12,10))
# sns.countplot(y=data['sex_upon_outcome'], 
#                   palette='rainbow',
#                   hue=data['outcome_type'])
# plt.show()
#data['sex_upon_outcome'].value_counts()
g = sns.FacetGrid(data, row='sex_upon_outcome', aspect=5)
g.map(sns.countplot, 'outcome_type', palette='rainbow')
#x = data[['sex_upon_outcome', 'outcome_type']].groupby(by=['sex_upon_outcome', 'outcome_type']).head()
#x.value_counts()
plt.figure(figsize=(12,6))
sns.countplot(data=data,
              x='animal_type',
              hue='outcome_type')
plt.legend(loc='upper right')
plt.show()
# g = sns.FacetGrid(data, row='animal_type', aspect=4)
# g.map(sns.countplot, 'outcome_type', palette='rainbow')
age_types = data['age_upon_outcome'].dropna().unique().tolist()
print(age_types)
# Let's format our datetime and date_of_birth columns to datetime format
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], format='%Y-%m-%d')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')
data['days_of_age'] =  (data['datetime'] - data['date_of_birth']).dt.days
data['days_of_age'].describe()
data[data['days_of_age'] < 0] = 0 
plt.subplots(figsize=(12, 6))
g = sns.boxplot(data=data, x='outcome_type', y='days_of_age', hue='animal_type')
labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)
plt.show(g)
g = sns.FacetGrid(data, hue="animal_type", size=12)
g.map(sns.kdeplot, "days_of_age") 
g.add_legend()
g.set(xlim=(0,5000), xticks=range(0,5000,365))
plt.show(g)
g = sns.FacetGrid(data[(data['animal_type']=='Cat') & (data['days_of_age']<2000)], 
                  hue="outcome_type", 
                  size=12)
g.map(sns.kdeplot, "days_of_age")
g.add_legend()
g.set(xlim=(0,1200), xticks=range(0,1200,365))
plt.show(g)
data.head()
# Lets remove the age_upon_outcome because we haver our days_of_age column already
data.drop('age_upon_outcome', axis=1, inplace=True)
# Lets also drop the date of birth, datetime and monthyear of our dataset for the sake of simplicity
data.drop(['date_of_birth','datetime', 'monthyear'], axis=1, inplace=True)
# The animal_id column is made of unique values that add nothing to our model, lets drop it
data.drop('animal_id', axis=1, inplace=True)
# Lets transform all string columns into lowercase strings 
string_columns = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'outcome_type', 'sex_upon_outcome']
for col in string_columns:
    data[col] = data[col].str.lower()
data.head()
def text_process(text):
    '''
    takes in a string and return
    a string with no punctuations
    but puts spaces in slash place
    '''
    text = text.replace('/', ' ')
    return ''.join([char for char in text if char not in string.punctuation])
for col in string_columns:
    data[col] = data[col].apply(lambda x:text_process(str(x)))
data.head()
data2 = data # data2 = placeholder
# dropping outcome_subtype as we won't try to predict it (yet?)
data.drop('outcome_subtype', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
columns = ['animal_type', 'breed', 'color', 'name',
       'sex_upon_outcome', 'outcome_type']

def encoder(df):
    for col in columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])
    return df
# feature = 'outcome_type'
# label_encoder = LabelEncoder()
# label_encoder.fit(data[feature])
# data[feature] = label_encoder.transform(data[feature])
data = encoder(data)
data.head()
from sklearn.model_selection import train_test_split
X = data.drop('outcome_type', axis=1)
y = data['outcome_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
list(data['outcome_type'].value_counts() / data['outcome_type'].count())[0]
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()
rfclf.fit(X_train, y_train)
rfclf_predictions = rfclf.predict(X_test)
print(classification_report(y_test, rfclf_predictions))
feat_importance = pd.DataFrame({'Feature':data.columns[:-1],'Importance':rfclf.feature_importances_.tolist()})

plt.subplots(figsize=(8, 6))
g = sns.barplot(data=feat_importance, x='Feature', y='Importance')
labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)
plt.show(g)
