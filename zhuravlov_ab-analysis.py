# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read data
import datetime as dt
df = pd.read_csv('../input/TrainData.csv', sep=';', encoding='iso-8859-1')
# replace 0 for nein and 1 for ja
df.replace({'nein': 0, 'ja' : 1}, inplace=True) 
# replace Schulabschluß with enumerable
df['Schulabschluß'].replace({'Real-/Hauptschule': 0, 'Abitur' : 1, 'Studium' : 2, 'Unbekannt' : np.nan}, inplace=True)
# same for Geschlecht, as there are only 2 possible values
df['Geschlecht'].replace({'w' : 0, 'm' : 1}, inplace=True)
# convert Monat and Tag into number of days since 1 jan
df['Date'] = pd.to_datetime(df['Tag'].map(str) + '/' + df['Monat'] + '/2017').dt.dayofyear
# remove lines with Unbekannt Schulabschluß and Art der Anstellung
df = df[np.isfinite(df['Schulabschluß'])]
df = df[df['Art der Anstellung'] != 'Unbekannt']
# convert categorical variables to dummy
df = pd.concat([df, pd.get_dummies(df['Art der Anstellung'], prefix='job')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Familienstand'], prefix='family')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Ergebnis letzte Kampagne'], prefix='last_result')], axis=1)
# drop not-useful columns, Kontaktart has too many 'Unbekannt'
df.drop(columns=['Tag', 'Monat', 'Anruf-ID', 'Art der Anstellung', 
        'Familienstand', 'Kontaktart', 'Ergebnis letzte Kampagne'], inplace=True)
# sample with unknown results of last campaign: 
y = df[df['last_result_Unbekannt'] == 1].iloc[:, 1].values
X = df[df['last_result_Unbekannt'] == 1].drop(columns=['Tage seit letzter Kampagne', 
                                                       'Anzahl Kontakte letzte Kampagne', 
                                                       'last_result_Erfolg',
                                                       'last_result_Kein Erfolg', 
                                                       'last_result_Sonstiges', 
                                                       'last_result_Unbekannt']).iloc[:, 2:].values

df1 = df[df['last_result_Unbekannt'] == 1].drop(columns=['Tage seit letzter Kampagne', 
                                                       'Anzahl Kontakte letzte Kampagne', 
                                                       'last_result_Erfolg',
                                                       'last_result_Kein Erfolg', 
                                                       'last_result_Sonstiges', 
                                                       'last_result_Unbekannt'])
features = list(df1)[2:]
len(features)

# divide into training and validation sample
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=0)
# undersample negative events in train sample
print('True train sample efficiency: ',  len(y_train[y_train == 1]) / len(y_train))
rus = RandomUnderSampler(return_indices=True)
X_train, y_train, id_rus = rus.fit_sample(X_train, y_train)
print('Ballances train sample efficiency: ',  len(y_train[y_train == 1]) / len(y_train))

from matplotlib import pyplot as plt
for i in range(0, X.shape[1]):
    plt.hist(X_train[y_train == 0, i], histtype='step', bins=40)
    plt.hist(X_train[y_train == 1, i], histtype='step', bins=40)
    plt.title(str(i) + ' ' + features[i])
    plt.show()
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_validate_std = stdsc.transform(X_validate)
# select useful features: l1 regularisation
from sklearn.linear_model import LogisticRegression
coefs_ = []
exponents_ = []
for exponent in np.arange (-3.5, -1.5, 0.05):
    lr = LogisticRegression(penalty='l1', C= 10**exponent, solver='liblinear')
    lr.fit(X_train_std, y_train)
    coefs_.append(lr.coef_.ravel().copy())
    exponents_.append(exponent)
coefs_ = np.array(coefs_)
exponents_ = np.array(exponents_)

plt.plot(exponents_, coefs_)
plt.xlabel('Log(C)')
plt.ylabel('Parameters')
plt.title('L1 Regularisation (Last Result Unknown)')
plt.show()
lr = LogisticRegression(penalty='l1', C= 10**-2.5, solver='liblinear')
lr.fit(X_train_std, y_train)
idx = list(np.argwhere(lr.coef_ != 0)[:, 1])
for i in idx:
    print(features[i])
# select 2 featires: Dauer and Haus
X_train_std_sel = X_train_std[:, idx]
X_validate_std_sel = X_validate_std[:, idx]
# ROC curve
from sklearn.metrics import roc_curve, auc
lr = LogisticRegression(penalty='l1', C= 0.01, solver='liblinear')
lr.fit(X_train_std_sel, y_train)
y_pred_valid = lr.predict_proba(X_validate_std_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_validate, y_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Logistic Regression AUC (Last Result Unknown)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))
plt.hist(y_pred_valid[y_validate == 0], label='background',bins=40)
plt.hist(y_pred_valid[y_validate == 1], label='signal',bins=40)
plt.xlabel('LR probability')
plt.legend()
plt.show()
# AUC versus l1 regularisation strenth
train_auc = []
valid_auc = []
exponents_ = []
for exponent in np.arange (-5, 2, 0.1):
    lr = LogisticRegression(penalty='l1', C= 10**exponent, solver='liblinear')
    lr.fit(X_train_std_sel, y_train)
    y_pred_train = lr.predict_proba(X_train_std_sel)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    train_auc.append(auc(fpr_train, tpr_train))
    y_pred_valid = lr.decision_function(X_validate_std_sel)
    fpr_valid, tpr_valid, _ = roc_curve(y_validate, y_pred_valid)
    valid_auc.append(auc(fpr_valid, tpr_valid))
    exponents_.append(exponent)
exponents_ = np.array(exponents_)
plt.plot(exponents_, train_auc, label='Traning Sample')
plt.plot(exponents_, valid_auc, label='Validation Sample')
plt.title('Logistic Regression AUC (Last Result Unknown)')
plt.xlabel('log(C)')
plt.legend()
plt.show()
# decision tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train_std_sel, y_train)
# ROC curve
y_pred_valid = tree.predict_proba(X_validate_std_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_validate, y_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Decision Tree (Last Result Unknown)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))
plt.hist(y_pred_valid[y_validate == 0], label='background',bins=40)
plt.hist(y_pred_valid[y_validate == 1], label='signal',bins=40)
plt.xlabel('DT probability')
plt.legend()
plt.show()
# AUC versus max depth of the tree
train_auc = []
valid_auc = []
depths = np.arange(2, 10)
for d in np.nditer(depths):        
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=d, random_state=0)
    tree.fit(X_train_std_sel, y_train)
    y_pred_train = tree.predict_proba(X_train_std_sel)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    train_auc.append(auc(fpr_train, tpr_train))
    y_pred_valid = tree.predict_proba(X_validate_std_sel)[:, 1]
    fpr_valid, tpr_valid, _ = roc_curve(y_validate, y_pred_valid)
    valid_auc.append(auc(fpr_valid, tpr_valid))
plt.plot(depths, train_auc, label='Traning Sample')
plt.plot(depths, valid_auc, label='Validation Sample')
plt.title('Decision Tree AUC (Last Result Unknown)')
plt.xlabel('Tree Depth')
plt.legend()
plt.show()
# random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=4, random_state=1)
forest.fit(X_train_std_sel, y_train)
# ROC curve
y_pred_valid = forest.predict_proba(X_validate_std_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_validate, y_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Random Forest (Last Result Unknown)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))
plt.hist(y_pred_valid[y_validate == 0], label='background',bins=40)
plt.hist(y_pred_valid[y_validate == 1], label='signal',bins=40)
plt.xlabel('RF probability')
plt.legend()
plt.show()
# AUC versus number of trees in forest
train_auc = []
valid_auc = []
ntrees = np.arange(1, 30)
for nt in np.nditer(ntrees):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=int(nt), max_depth=4, random_state=1)
    forest.fit(X_train_std_sel, y_train)
    y_pred_train = forest.predict_proba(X_train_std_sel)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    train_auc.append(auc(fpr_train, tpr_train))
    y_pred_valid = forest.predict_proba(X_validate_std_sel)[:, 1]
    fpr_valid, tpr_valid, _ = roc_curve(y_validate, y_pred_valid)
    valid_auc.append(auc(fpr_valid, tpr_valid))
plt.plot(ntrees, train_auc, label='Traning Sample')
plt.plot(ntrees, valid_auc, label='Validation Sample')
plt.title('Random Forest AUC (Last Result Unknown)')
plt.xlabel('Number of trees')
plt.legend()
plt.show()
# golden sample with known results of last campaign: 
df2 = df[df['last_result_Unbekannt'] == 0].drop(columns=['last_result_Unbekannt'])
y_gold = df[df['last_result_Unbekannt'] == 0].iloc[:, 1].values
X_gold = df2.iloc[:, 2:].values
features = list(df2)[2:]
# divide into training and validation sample
X_gold_train, X_gold_validate, y_gold_train, y_gold_validate = train_test_split(X_gold, y_gold, test_size=0.3, random_state=0)
print('True train sample efficiency: ',  len(y_gold_train[y_gold_train == 1]) / len(y_gold_train))
X_gold_train, y_gold_train, id_rus = rus.fit_sample(X_gold_train, y_gold_train)
print('Ballances train sample efficiency: ',  len(y_gold_train[y_gold_train == 1]) / len(y_gold_train))
stdsc_gold = StandardScaler()
X_gold_train_std = stdsc_gold.fit_transform(X_gold_train)
X_gold_validate_std = stdsc_gold.transform(X_gold_validate)

# select useful features: l1 regularisation
coefs_ = []
exponents_ = []
for exponent in np.arange (-3.0, -1.0, 0.05):
    lr_gold = LogisticRegression(penalty='l1', C= 10**exponent, solver='liblinear')
    lr_gold.fit(X_gold_train_std, y_gold_train)
    coefs_.append(lr_gold.coef_.ravel().copy())
    exponents_.append(exponent)
coefs_ = np.array(coefs_)
exponents_ = np.array(exponents_)

plt.plot(exponents_, coefs_)
plt.xlabel('Log(C)')
plt.ylabel('Parameters')
plt.title('L1 Regularisation (Last Result Known)')
plt.show()
lr_gold = LogisticRegression(penalty='l1', C= 10**-2.25, solver='liblinear')
lr_gold.fit(X_gold_train_std, y_gold_train)
idx_gold = list(np.argwhere(lr_gold.coef_ != 0)[:, 1])
for i in idx_gold:
    print(features[i])
# select 3 featires: Dauer, Haus and last_result_Erfolg
X_gold_train_std_sel = X_gold_train_std[:, idx_gold]
X_gold_validate_std_sel = X_gold_validate_std[:, idx_gold]
# ROC curve
lr_gold = LogisticRegression(penalty='l1', C= 0.1, solver='liblinear')
lr_gold.fit(X_gold_train_std_sel, y_gold_train)
y_gold_pred_valid = lr_gold.decision_function(X_gold_validate_std_sel)
fpr, tpr, _ = roc_curve(y_gold_validate, y_gold_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Logistic Regression (Last Result Known)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))
plt.hist(y_pred_valid[y_validate == 0], label='background',bins=40)
plt.hist(y_pred_valid[y_validate == 1], label='signal',bins=40)
plt.xlabel('LR probability')
plt.legend()
plt.show()

# decision tree
tree_gold = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree_gold.fit(X_gold_train_std_sel, y_gold_train)
# ROC curve
y_pred_valid = tree_gold.predict_proba(X_gold_validate_std_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_gold_validate, y_gold_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Decision Tree (Last Result Known)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))

# AUC versus max depth of the tree
train_auc = []
valid_auc = []
depths = np.arange(2, 10)
for d in np.nditer(depths):        
    tree_gold = DecisionTreeClassifier(criterion='entropy', max_depth=d, random_state=0)
    tree_gold.fit(X_gold_train_std_sel, y_gold_train)
    y_pred_train = tree_gold.predict_proba(X_gold_train_std_sel)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_gold_train, y_pred_train)
    train_auc.append(auc(fpr_train, tpr_train))
    y_pred_valid = tree_gold.predict_proba(X_gold_validate_std_sel)[:, 1]
    fpr_valid, tpr_valid, _ = roc_curve(y_gold_validate, y_pred_valid)
    valid_auc.append(auc(fpr_valid, tpr_valid))
plt.plot(depths, train_auc, label='Traning Sample')
plt.plot(depths, valid_auc, label='Validation Sample')
plt.title('Decision Tree AUC (Last Result Known)')
plt.xlabel('Tree Depth')
plt.legend()
plt.show()
# random forest
forest_gold = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=4, random_state=1)
forest_gold.fit(X_gold_train_std_sel, y_gold_train)
# ROC curve
y_pred_valid = forest_gold.predict_proba(X_gold_validate_std_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_gold_validate, y_pred_valid)
plt.plot(fpr, tpr)
plt.title('ROC Randon Forest (Last Result Known)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('AUC = ', auc(fpr, tpr))
plt.hist(y_pred_valid[y_gold_validate == 0], label='background',bins=40)
plt.hist(y_pred_valid[y_gold_validate == 1], label='signal',bins=40)
plt.xlabel('DT probability')
plt.legend()
plt.show()
# AUC versus number of trees in forest
train_auc = []
valid_auc = []
ntrees = np.arange(1, 30)
for nt in np.nditer(ntrees):
    forest_gold = RandomForestClassifier(criterion='entropy', n_estimators=int(nt), max_depth=4, random_state=1)
    forest_gold.fit(X_gold_train_std_sel, y_gold_train)
    y_pred_train = forest_gold.predict_proba(X_gold_train_std_sel)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_gold_train, y_pred_train)
    train_auc.append(auc(fpr_train, tpr_train))
    y_pred_valid = forest_gold.predict_proba(X_gold_validate_std_sel)[:, 1]
    fpr_valid, tpr_valid, _ = roc_curve(y_gold_validate, y_pred_valid)
    valid_auc.append(auc(fpr_valid, tpr_valid))
plt.plot(ntrees, train_auc, label='Traning Sample')
plt.plot(ntrees, valid_auc, label='Validation Sample')
plt.title('Random Forest AUC (Last Result Known)')
plt.xlabel('Number of trees')
plt.legend()
plt.show()
df_test = pd.read_csv('../input/TestData.csv', sep=';', encoding='iso-8859-1')
# replace 0 for nein and 1 for ja
df_test.replace({'nein': 0, 'ja' : 1}, inplace=True) 
# replace Schulabschluß with enumerable
df_test['Schulabschluß'].replace({'Real-/Hauptschule': 0, 'Abitur' : 1, 'Studium' : 2, 'Unbekannt' : np.nan}, inplace=True)
# same for Geschlecht, as there are only 2 possible values
df_test['Geschlecht'].replace({'w' : 0, 'm' : 1}, inplace=True)
# convert Monat and Tag into number of days since 1 jan
df_test['Date'] = pd.to_datetime(df_test['Tag'].map(str) + '/' + df_test['Monat'] + '/2017').dt.dayofyear
# convert categorical variables to dummy
df_test = pd.concat([df_test, pd.get_dummies(df_test['Art der Anstellung'], prefix='job')], axis=1)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Familienstand'], prefix='family')], axis=1)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Ergebnis letzte Kampagne'], prefix='last_result')], axis=1)
# drop not-useful columns, Kontaktart has too many 'Unbekannt'
df_test.drop(columns=['Tag', 'Monat', 'Anruf-ID', 'Art der Anstellung', 'job_Unbekannt', 
        'Familienstand', 'Kontaktart', 'Ergebnis letzte Kampagne'], inplace=True)
df_test.fillna(value=0, inplace=True)
# sample with unknown results of last campaign: 
X_test = df_test[df_test['last_result_Unbekannt'] == 1].drop(columns=['Tage seit letzter Kampagne', 
                                                       'Anzahl Kontakte letzte Kampagne', 
                                                       'last_result_Erfolg',
                                                       'last_result_Kein Erfolg', 
                                                       'last_result_Sonstiges', 
                                                       'last_result_Unbekannt']).iloc[:, 2:].values
X_test_std = stdsc.transform(X_test)
X_test_std_sel = X_test_std[:, idx]
y_pred_test = forest.predict_proba(X_test_std_sel)[:, 1]
y_pred_valid = forest.predict_proba(X_validate_std_sel)[:, 1]
plt.hist(y_pred_test, label='Test Sample (Last Result Unknown)', bins=40)
plt.legend()
plt.show()

# sample with known last result
X_gold_test = df_test[df_test['last_result_Unbekannt'] == 0].drop(columns=['last_result_Unbekannt']).iloc[:, 2:].values
X_gold_test_std = stdsc_gold.transform(X_gold_test)
X_gold_test_std_sel = X_gold_test_std[:, idx_gold]
y_gold_pred_test = forest_gold.predict_proba(X_gold_test_std_sel)[:, 1]
y_pred_valid = forest_gold.predict_proba(X_gold_validate_std_sel)[:, 1]
plt.hist(y_gold_pred_test, label='Test Sample (Last Result Known)', bins=40)
plt.legend()
plt.show()

Stammnummer = df_test[df_test['last_result_Unbekannt'] == 1]['Stammnummer'].values
Stammnummer_gold = df_test[df_test['last_result_Unbekannt'] == 0]['Stammnummer'].values
Stammnummer_combined = np.concatenate((Stammnummer, Stammnummer_gold))
y_pred_combined = np.concatenate((y_pred_test, y_gold_pred_test))
df_solution = pd.DataFrame({'ID' : Stammnummer_combined, 'Expected' : y_pred_combined})
df_solution.to_csv('solution.csv', index=False)
