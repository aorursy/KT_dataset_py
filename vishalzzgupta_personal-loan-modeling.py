# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
df=pd.read_csv('/kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv')

df.head()
df.shape
df.info()
df.isnull().any()
df.drop(['ID', 'ZIP Code'], axis = 1, inplace = True)
cols = set(df.columns)

cols_numeric = set(['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage'])

cols_categorical = list(cols - cols_numeric)

cols_categorical
for x in cols_categorical:

    df[x] = df[x].astype('category')



df.info()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
def summary_non_category(x):

    x_min = df[x].min()

    x_max = df[x].max()

    Q1 = df[x].quantile(0.25)

    Q2 = df[x].quantile(0.50)

    Q3 = df[x].quantile(0.75)

    print(f'Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min) : {x_min}\n'

          f'Q1                    : {Q1}\n'

          f'Q2(Median)            : {Q2}\n'

          f'Q3                    : {Q3}\n'

          f'{x.capitalize()}(max) : {x_max}')

# Plotting Graph

    sns.distplot(df[x])

    plt.title(f'{x.capitalize()} Density Distribution')

    plt.show()
for column in cols_numeric:

    summary_non_category(column)
def summary_category(category_column):

    count_category= []

    value_category = []

    category_loan = []

    category_no_loan =[]

    category = df[category_column].unique()

    for x in category:

        value_category.append(x)

        count_category.append(df[category_column][df[category_column] ==x].count())

    value_category = np.array(value_category)  

    for x in np.nditer(value_category):

        category_loan.append(df[category_column][df[category_column]==x][df["Personal Loan"] ==1].count())

        category_no_loan.append(df[category_column][df[category_column]==x][df["Personal Loan"] ==0].count())

# Plotting Graph

    fig, (ax1,ax2) = plt.subplots(1,2)

    ax1.pie(count_category,labels=value_category, autopct='%1.1f%%')

    ax2.bar(value_category-0.2,category_loan, width=0.4, label="Loan")

    ax2.bar(value_category+0.2,category_no_loan, width=0.4,label="No Loan")

    plt.title(category_column)

    plt.legend()

    plt.show()
for category_column in cols_categorical:

    summary_category(category_column)
X = df.drop('Personal Loan', axis = 1)

y = df['Personal Loan']

data_num = df.select_dtypes(include='number')

sns.pairplot(X ,diag_kind = 'kde', vars = list(data_num.columns))
X = df.drop('Personal Loan', axis = 1)

y = df['Personal Loan']
# thanks to Anirban Datta

corr = X.corr()

plt.figure(figsize=(10, 8))

g = sns.heatmap(corr, annot=True, cmap = 'summer_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})

g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')

bottom, top = g.get_ylim()

g.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
df.Experience.unique()
df["Experience"][df["Experience"]<0].count()
df["Experience"][df["Experience"]>=0].count()
df2 = df[df["Experience"]>=0]

df2.head()
df2.describe().transpose()
df2["Mortgage"][df2["Mortgage"]==0].count()
mortgage = {}

mortgage["Personal_loan_and_no_mortagage"]    = df2["Mortgage"][df2["Mortgage"]==0][df2["Personal Loan"]==1].count()

mortgage["no_Personal_loan_and_no_mortagage"] = df2["Mortgage"][df2["Mortgage"]==0][df2["Personal Loan"]==0].count()

mortgage["no_Personal_loan_and_mortagage"]    = df2["Mortgage"][df2["Mortgage"]>0][df2["Personal Loan"]==0].count()

mortgage["Personal_loan_and_mortagage"]       = df2["Mortgage"][df2["Mortgage"]>0][df2["Personal Loan"]==1].count()

mortgage
xpos = np.arange(len(mortgage))

value = [x for x in mortgage.values()]

keys = [x for x in mortgage.keys()]

plt.bar(xpos,value)

plt.xticks(xpos)

plt.ylabel("Count")

plt.title('Mortgage')

plt.show()
from sklearn.preprocessing import MinMaxScaler
df2.columns

scale = MinMaxScaler()
X = df2[['Income', 'CCAvg',"Mortgage","Age","Experience"]]

scaledX = scale.fit_transform(X)

df2['Income']     = (scaledX[:,0])

df2["CCAvg"]      = (scaledX[:,1])

df2["Mortgage"]   = (scaledX[:,2])

df2["Age"]        = (scaledX[:,3])

df2["Experience"] = (scaledX[:,4])

df2.head()
sns.distplot(df2["Income"])
sns.distplot(df2["CCAvg"])
upper_limit_income = df2["Income"].mean() + 3*df2["Income"].std()

upper_limit_income
upper_limit_ccavg = df2["CCAvg"].mean() + 2*df2["CCAvg"].std()

upper_limit_ccavg
df2.shape
df3 = df2[df2["Income"]<upper_limit_income][df2["CCAvg"]<upper_limit_ccavg]

df3.shape
sns.distplot(df3["Income"])
sns.distplot(df3["CCAvg"])
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

X = df3.drop(['Personal Loan'],axis='columns')

X.head(3)
y = df3["Personal Loan"]

y.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.model_selection import ShuffleSplit          # for random suffle rather than in order

from sklearn.model_selection import cross_val_score



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(RandomForestClassifier(), X, y, cv=cv)
from sklearn.model_selection import ShuffleSplit          # for random suffle rather than in order

from sklearn.model_selection import cross_val_score



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)



cross_val_score(DecisionTreeClassifier(), X, y, cv=cv)
model_params = {

    'svm': {

        'model': svm.SVC(gamma='auto'),

        'params' : {

            'C': [1,10,20],

            'kernel': ['rbf','linear']

        }  

    },

    'random_forest': {

        'model': RandomForestClassifier(),

        'params' : {

            'n_estimators': [1,5,10]

        }

    },

    'logistic_regression' : {

        'model': LogisticRegression(solver='liblinear',multi_class='auto'),

        'params': {

            'C': [1,5,10]

        }

    },

    'DecisionTree': {

        'model' : DecisionTreeClassifier(),

        'params' : {

            'criterion' : ["gini", "entropy"]

        }

    },

    'GaussianNB' : {

        'model' : GaussianNB(),

        'params' : {}

          

 },

    'MultinomialNB' : {

        'model' : MultinomialNB(),

        'params' : {}

            

            

        

    }

}
from sklearn.model_selection import GridSearchCV

scores = []

best_estimators = {}

for model_name, mp in model_params.items():

    clf =  GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)

    clf.fit(X_train, y_train)

    scores.append({

        'model': model_name,

        'best_score': clf.best_score_,

        'best_params': clf.best_params_

    })

    best_estimators[model_name] = clf.best_estimator_

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

df
best_clf = best_estimators["DecisionTree"]
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, best_clf.predict(X_test))

cm
import seaborn as sn

plt.figure(figsize = (10,7))

sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')
# Thanks to Anirban Datta

best_clf.fit(X_train, y_train)



features = list(X_train.columns)

importances = best_clf.feature_importances_

indices = np.argsort(importances)



fig, ax = plt.subplots(figsize=(10, 7))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=14)

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance', fontsize = 18)

plt.show()