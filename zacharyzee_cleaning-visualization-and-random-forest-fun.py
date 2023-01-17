import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/survey.csv')
df.head()
df.tail()
df.info()
df['Age'].value_counts().plot(kind='bar', figsize=(14,8));
df.query('Age <= 10 | Age >= 65')
df = df[df.Age > 18]
df = df[df.Age < 65]
df.info()
df['Age'].mean()
df['Age'].value_counts().plot(kind='bar',figsize=(14,8))
df['Country'].value_counts().plot(kind='bar',figsize=(14,8));
fam_history_pos = df[df.family_history == 'Yes']

fam_history_neg = df[df.family_history == 'No']
ax = fam_history_pos['treatment'].value_counts().plot(kind='bar', figsize = (14,8), alpha = 0.6, label = 'Family History', width=0.15, position = 0);

ax = fam_history_neg['treatment'].value_counts().sort_values(ascending=True).plot(kind='bar', figsize = (14,8), alpha = 0.6, color = 'r', label = 'No Family History', width= 0.15, position = 1, title='Seeking Treatment Based on Family History')

ax.set_xlabel("Sought Treatment")

ax.set_ylabel("Count")

plt.tight_layout()

plt.legend();
df['Gender'].value_counts()
male = ['Male ', 'male', 'male ', 'M', 'm', 'make', 'man', 'cis man', 'malr', 'mail', 'mal', 'Make', 'Male', 'Cis Male', 'Cis Man', 'Male (CIS)', 'male (cis)', 'Mal', 'ostensibly male, unsure what that really means', 'cis male', 'Malr', 'maile', 'msle', 'Mail', 'Man']

female = ['Female', 'female', 'female ', 'femake', 'cis female', 'F', 'f', 'Woman', 'Female ', 'cis-female/femme', 'Femake', 'Cis Female', 'femail', 'woman', 'Female (cis)', 'female (cis)']

trans_other = ['Nah','non-binary','Male-ish','trans-female','queer','Guy (-ish) ^_^','enby','Androgyne','neuter','queer/she/they','nah','Agender','Genderqueer','male leaning androgynous','fluid','trans woman','Female (trans)',"Trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "Enby", "fluid", "genderqueer", "Androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "Trans woman", "Neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"] 
df.Gender = df.Gender = df.Gender.replace(male, 'male')

df.Gender = df.Gender.replace(female, 'female')

df.Gender = df.Gender.replace(trans_other, 'trans/other')
df['Gender'].value_counts()
gender_male = df[df.Gender == 'male']

gender_female = df[df.Gender == 'female']

gender_trans = df[df.Gender == 'trans/other']
gender_male['seek_help'].value_counts()
ax = gender_male['seek_help'].value_counts().plot(kind='bar', figsize = (14,8), alpha = 0.6, label = 'Male', width=0.1, position = 0);

ax = gender_female['seek_help'].value_counts().plot(kind='bar', figsize = (14,8), alpha = 0.6, label = 'Female', width=0.1, position = 1, color = 'g');

ax = gender_trans['seek_help'].value_counts().sort_values(ascending=True).plot(kind='bar', figsize = (14,8), alpha = 0.6, color = 'r', label = 'Trans & Others', width= 0.1, position = 2, title='Seeking Help Based on Gender')



ax.set_xlabel("Sought Help")

ax.set_ylabel("Count")

plt.tight_layout()

plt.legend();
ax = gender_trans['seek_help'].value_counts().sort_values(ascending=True).plot(kind='bar', figsize = (14,8), alpha = 0.6, color = 'r', label = 'Trans & Others', width= 0.15, position = 2, title='Seeking Help Based on Gender - Trans & Others')

ax.set_xlabel("Sought Help")

ax.set_ylabel("Count")

plt.tight_layout()

plt.legend();
df.drop('Timestamp',axis=1,inplace=True)

df.drop('self_employed',axis=1,inplace=True)

df.drop('work_interfere',axis=1,inplace=True)

df.drop('Country',axis=1,inplace=True)

df.drop('state',axis=1,inplace=True)
df.head()
comments = df['comments']

df.drop('comments',axis=1,inplace=True)
comments.dropna(inplace=True)

comments.head()
# new column names

col_name = df.columns.values
le = LabelEncoder()
df.family_history = le.fit_transform(df.family_history) 

df.mental_health_consequence = le.fit_transform(df.mental_health_consequence)

df.phys_health_consequence = le.fit_transform(df.phys_health_consequence)

df.coworkers = le.fit_transform(df.coworkers)

df.supervisor = le.fit_transform(df.supervisor)

df.mental_health_interview = le.fit_transform(df.mental_health_interview)

df.phys_health_interview = le.fit_transform(df.phys_health_interview)

df.mental_vs_physical = le.fit_transform(df.mental_vs_physical)

df.obs_consequence = le.fit_transform(df.obs_consequence)

df.remote_work = le.fit_transform(df.remote_work)

df.tech_company = le.fit_transform(df.tech_company)

df.benefits = le.fit_transform(df.benefits)

df.care_options = le.fit_transform(df.care_options)

df.wellness_program = le.fit_transform(df.wellness_program)

df.seek_help = le.fit_transform(df.seek_help)

df.anonymity = le.fit_transform(df.anonymity)
#preserve order in company size

df.loc[df['no_employees']=='1-5',['no_employees']]=1

df.loc[df['no_employees']=='6-25',['no_employees']]=2

df.loc[df['no_employees']=='26-100',['no_employees']]=3

df.loc[df['no_employees']=='100-500',['no_employees']]=4

df.loc[df['no_employees']=='500-1000',['no_employees']]=5

df.loc[df['no_employees']=='More than 1000',['no_employees']]=6
df['leave'].replace(['Very easy', 'Somewhat easy', "Don\'t know", 'Somewhat difficult', 'Very difficult'], [1, 2, 3, 4, 5],inplace=True) 
df.loc[df['Gender']=='male',['Gender']]=1

df.loc[df['Gender']=='female',['Gender']]=2

df.loc[df['Gender']=='trans/other',['Gender']]=3
df.head(5)
X = df.drop(['treatment'],axis=1)

y = df['treatment']

y = le.fit_transform(y) # yes:1 no:0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(max_depth=40,min_samples_split=10, n_estimators=50, random_state=1)

clf.fit(X_train,y_train)
clf.predict(X_test)
y_test
clf.predict_proba(X_train)[0:10]
list(zip(X_train,clf.feature_importances_))
accuracy_score(y_test,clf.predict(X_test))