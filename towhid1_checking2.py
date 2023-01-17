# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
sns.set_style('whitegrid')
%matplotlib inline


# get bnp & test csv files as a DataFrame
train_df   = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")
for f in train_df.columns:
    # fill NaN values withm -1
    if train_df[f].dtype == 'float64':
        train_df.loc[:,f][np.isnan(train_df[f])] = -1
        test_df[f][np.isnan(test_df[f])] = -1
        
    # fill NaN values with -1
    elif train_df[f].dtype == 'object':
        train_df[f][train_df[f] != train_df[f]] = -1
        test_df[f][test_df[f] != test_df[f]] = -1
        
for f in train_df.columns:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[f].values)  + list(test_df[f].values)))
        train_df[f]   = lbl.transform(list(train_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))
plt.rcParams['figure.max_open_warning']=300
colnames=list(train_df.columns.values)
for i in colnames[5:6]:
        facet = sns.FacetGrid(train_df, hue="target",aspect=2)
        facet.map(sns.kdeplot,i,shade= False)
        facet.add_legend()
from sklearn import preprocessing

for f in bnp_df.columns:
    if bnp_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(bnp_df[f].values) + list(test_df[f].values)))
        bnp_df[f]   = lbl.transform(list(bnp_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))
X_train = bnp_df.drop(["ID","target"],axis=1)
Y_train = bnp_df["target"]
X_test  = test_df.drop("ID",axis=1).copy()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)
