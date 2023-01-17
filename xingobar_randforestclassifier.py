import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from scipy.stats import skew

from scipy.stats import boxcox

from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report,accuracy_score,roc_auc_score

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')
df.head()
df.describe()
df['Class'].value_counts()
df['Time_in_hour'] = df['Time'] / 3600

df['Log_Amount'] = np.log1p(df['Amount'])
sns.plt.hist(df['Log_Amount'])

plt.title('Log Amount')
df['Time_in_hour'].plot(kind='hist')

plt.title('Time in hour')
sns.countplot(df['Class'])

plt.title('Class Count')
sns.plt.hist(data=df[df['Class'] == 1] , x ='Time_in_hour')

plt.title('Time_in_hour of class 1' )
columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',

       'Class', 'Time_in_hour', 'Log_Amount']

fig,ax = plt.subplots(figsize=(8,6))

correlation = df[columns].corr(method='pearson')

sns.heatmap(correlation,square=True,ax=ax,vmax=1)

plt.title('Numeric Columns Correlation')
corr_dict = correlation['Class'].to_dict()

## descending 

for key,val in sorted(corr_dict.items(),key=lambda x:-abs(x[1])):

    print('{0} \t : {1}'.format(key,val))
df.drop(['Time','Amount'],axis=1,inplace=True)
columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',

       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',

       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time_in_hour',

       'Log_Amount']

skewed_list = []

for col in columns:

    skewed_list.append(skew(df[col]))

plt.plot(skewed_list,'bo-')

plt.plot([0.25 for i in range(len(columns))],'r--')

plt.text(12,2,'threshold 0.25')
skewed = df[columns].apply(lambda x:skew(x.dropna()))

skewed = skewed[skewed > 0.25].index

df[skewed] = np.log1p(df[skewed])

for col in columns:

    df.loc[df[col].isnull(),col] = df[col].median()
y = df['Class']

df.drop(['Class'],axis=1,inplace=True)

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=.2,random_state=42)
clf = RandomForestClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_valid)
print(classification_report(y_valid,y_pred))

print('Accuracy : %f' %(accuracy_score(y_valid,y_pred)))

print('Area under the curve : %f' %(roc_auc_score(y_valid,y_pred)))