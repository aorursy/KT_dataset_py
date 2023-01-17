import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")



df.head()
df["pdays"].describe()
df["pdays"].value_counts()
df["pdays"][df["pdays"]>0].describe()
sns.barplot(x="education",y="balance", data=df[df["education"]!="unknown"],estimator=np.median);
plt.figure(figsize=(25,8))

sns.boxplot("pdays", data= df[df["pdays"]>0]);
df.drop(df[df["pdays"]>600].index,inplace=True) # remove putliers
df.head()
df['deposit'].replace('no',0, inplace=True)

df['deposit'].replace('yes',1, inplace=True)
df[df["pdays"]<0]
sns.barplot(x="deposit", y="age", data=df);
sns.barplot(x="deposit", y="balance", data=df);
sns.barplot(x="deposit", y="duration", data=df);
sns.barplot(x="deposit", y="pdays", data=df[df['pdays']>0]);
sns.barplot(x="deposit", y="previous", data=df[df['previous']>0]);
df.corr()
numeric_features_names = ['age','balance', 'day', 'duration','campaign', 'pdays', 'previous']

categorical_features_name = ['job','marital','education','default','housing','loan','contact','month', 'poutcome']
from sklearn.preprocessing import StandardScaler

ss= StandardScaler()

ss.fit(df[numeric_features_names])

df[numeric_features_names] = ss.transform(df[numeric_features_names])

#print(df)
df = pd.get_dummies(df, columns=categorical_features_name,drop_first=True) #drop first = true to take care of dummy

#variable trap



#print(df)
df.columns
X = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',

        'job_blue-collar', 'job_housemaid',

       'job_management', 'job_retired', 'job_services',

       'job_student', 'job_technician', 'job_unemployed',

       'marital_married', 'marital_single', 'education_secondary',

       'education_tertiary', 'education_unknown', 'housing_yes',

       'loan_yes', 'contact_telephone', 'contact_unknown',

       'month_dec', 'month_feb', 'month_jan', 'month_jun',

       'month_mar', 'month_may', 'month_oct', 'month_sep',

       'poutcome_other', 'poutcome_success', 'poutcome_unknown'] ]

y = df["deposit"]
import statsmodels.api as sm

OLS= sm.OLS(y,X).fit()

OLS.summary()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

model = lr.fit(X_train,np.array(y_train))



print(model)
y_pred = model.predict(X_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)

np.mean(scores)
from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=45,random_state=0,max_depth=30)

classifier=classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, X_train, y_train, cv=5)

np.mean(scores)
from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))