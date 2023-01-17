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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb
# Get the data.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.info())
print(test.info())
print(train.head())
# EDA
# As we can see we have three types of data and most of the integer data are booleans. So here we only 
# concentrate on exploring float data and objects data.

# Float data.
from collections import OrderedDict
plt.figure(figsize = (30, 26))

color_use = {1: 'r', 2: 'g', 3: 'c', 4: 'y'}
colors = OrderedDict(color_use)
poverty_level = {1: 'extreme poverty', 2: 'moderate poverty', 3: 'vulnerable households', 
                            4: 'nonvulnerable households'}
pl_marker = OrderedDict(poverty_level)

float_data = train.select_dtypes('float')
for index, columns in enumerate(float_data):
    axis = plt.subplot(4, 2, index + 1)
    for pl, cl in colors.items():
        sb.kdeplot(train.loc[train['Target'] == pl, columns], ax = axis, color = cl, 
                   label = pl_marker[pl])
        plt.xlabel('values of ' f'{columns}')
        plt.ylabel('proportion')
# From these float distributions we may find some features that can influence the poverty level 
# significantly and do further analysis.

# Object data.
object_data = train.select_dtypes('object')
print(object_data.info())
print(object_data.head())
# We can ignore the "Id" and "idhogar" because they are identifying features based on the problem. For the
# other three features we can replace "yes" to "1" and replace "no" to "0" according to the documentation 
# on the kaggle.
replace_yn = {'yes': 1, 'no': 0}

train['dependency'] = train['dependency'].replace(replace_yn).astype(np.float)
train['edjefe'] = train['edjefe'].replace(replace_yn).astype(np.float)
train['edjefa'] = train['edjefa'].replace(replace_yn).astype(np.float)

test['dependency'] = test['dependency'].replace(replace_yn).astype(np.float)
test['edjefe'] = test['edjefe'].replace(replace_yn).astype(np.float)
test['edjefa'] = test['edjefa'].replace(replace_yn).astype(np.float)
train['dependency'].head()
# Make plots if needed.
# Since some of individuals have different poverty level in the same household, we only need to consider 
# the poverty level of the head of the household according to the ducumentation on kaggle. Here we use the 
# poverty level of the head of the household as the true target variable.
train_num_househould = train.groupby('idhogar')['Target'].nunique()
print('Numbers of households: ', len(train_num_househould))
train_unique = train_num_househould[train_num_househould == 1]
print('Numbers of households whose poverty levels of family members are the same: ', len(train_unique))
train_not_unique = train_num_househould[train_num_househould != 1]
print('Numbers of households whose poverty levels of family members are not the same', len(train_not_unique))
# One example of these ununique targets within each household.
print(train[train['idhogar'] == train_not_unique.index[0]][['idhogar', 'parentesco1', 'Target']])
# Locate the head of household.
head = train[(train['idhogar'] == train_not_unique.index[0]) & (train['parentesco1'] == 1)]
print(head)
# Select the target variables that need to be changed.
not_unique_target = train.loc[train['idhogar'] == train_not_unique.index[0], 'Target']
print(not_unique_target)
# Change the target variables.
not_unique_target = int(head['Target'])
print(not_unique_target)
# Set the true label for members of households whose labels are not the same.
# Here we write a for loop to replace all the ununique target variables.
for unique_hhid in train_not_unique.index:
    hh_head = train[(train['idhogar'] == unique_hhid) & (train['parentesco1'] == 1)]
    train.loc[train['idhogar'] == unique_hhid, 'Target'] = hh_head['Target']

# Let's check if it works.
train_num_househould = train.groupby('idhogar')['Target'].nunique()
print('Numbers of households: ', len(train_num_househould))
train_unique = train_num_househould[train_num_househould == 1]
print('Numbers of households whose poverty levels of family members are the same: ', len(train_unique))
train_not_unique = train_num_househould[train_num_househould != 1]
print('Numbers of households whose poverty levels of family members are not the same', len(train_not_unique))
# Now let's deal with the missing values.
# Let's check the missiong values for both train and test data.
trainms = pd.DataFrame(train.isnull().sum())
trainms['counts'] = trainms
trainms['ratio'] = trainms['counts']/len(train)
print(trainms.sort_values(by = 'ratio', ascending = False).head(10))


testms = pd.DataFrame(test.isnull().sum())
testms['counts'] = testms
testms['ratio'] = testms['counts']/len(test)
print(testms.sort_values(by = 'ratio', ascending = False).head(10))
# Now we know what features have the most missing values, then we can analyze them based on their own 
# meanings and decide whether we should delete the features or replace the missing values.
head_train = train.loc[train['parentesco1'] == 1]
head_test = test.loc[test['parentesco1'] == 1]

# rez_esc: years behind in school.
# Actually I don't really know the mean of "years behind in school", but what I do know is that it may be 
# related to the age feature. So let's check what's the relationship between this feature and age.
print(train.loc[train['rez_esc'].isnull()]['age'].describe())
print(train.loc[train['rez_esc'].notnull()]['age'].describe())
print(test.loc[test['rez_esc'].isnull()]['age'].describe())
print(test.loc[test['rez_esc'].notnull()]['age'].describe())
# Now we find that all the defined "rez_esc" values are between the age 7 and 17. Those who are younger 
# than 7 or older than 17 have missing values, which means we should set "rez_esc" values of these people 
# to 0 instead of null. And if there are some other situations, we can leave the values to be imputed and 
# add a boolean flag.
train.loc[(train['age'] > 17) & (train['rez_esc'].isnull()), 'rez_esc'] = 0
train.loc[(train['age'] < 7) & (train['rez_esc'].isnull()), 'rez_esc'] = 0

test.loc[(test['age'] > 17) & (test['rez_esc'].isnull()), 'rez_esc'] = 0
test.loc[(test['age'] < 7) & (test['rez_esc'].isnull()), 'rez_esc'] = 0
# v18q1: number of tablets household owns.
# This could be compared with the feature "v18q: owns a tablet". Maybe the missing values in "v18q1" 
# indicates that the households don't even own a tablet, which means the value is 0 in "v18q".
# First let's check the values of "v18q1".
print('Numbers of null values of "v18q1" in train: ', head_train['v18q1'].isnull().sum())
print('Numbers of "0" of "v18q" in train: ', head_train[head_train['v18q'] == 0]['v18q'].count())
print('Numbers of null values of "v18q1" in test: ', head_test['v18q1'].isnull().sum())
print('Numbers of "0" of "v18q" in test: ', head_test[head_test['v18q'] == 0]['v18q'].count())
# Here we can see that every household has missing value in "v18q1" doesn't have any tablet. Then we can 
# replace the missing values to 0.
train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)

train['v18q1'].value_counts().plot.bar(color = 'red')
plt.xlabel('values of "v18q1"')
plt.ylabel('counts of individuals')
plt.title('"v18q1" in train')
plt.show()

train['v18q'].value_counts().plot.bar(color = 'red')
plt.xlabel('values of "v18q"')
plt.ylabel('counts of individuals')
plt.title('"v18q" in train')
plt.show()
test['v18q1'].value_counts().plot.bar(color = 'blue')
plt.xlabel('values of "v18q1"')
plt.ylabel('counts of individuals')
plt.title('"v18q1" in test')
plt.show()

test['v18q'].value_counts().plot.bar(color = 'blue')
plt.xlabel('values of "v18q"')
plt.ylabel('counts of individuals')
plt.title('"v18q" in test')
plt.show()
# v2a1: monthly rent payment.
# According to the explanations of the documentation on the kaggle, this feature may be related to the 
# "tipovivi", which represents the status of the house. For example, if people already own this house, then 
# they don't need to pay the rent anymore.
status = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']

train.loc[train['v2a1'].isnull(), status].sum().plot.bar(color = 'green')
plt.xticks([0, 1, 2, 3, 4], ['own and fully paid house', 'own and paying in installments', 
                            'rented', 'precarious', 'other'], rotation = 40)
plt.title('status of the house for missing values of "v2a1" in train')
plt.show()

test.loc[test['v2a1'].isnull(), status].sum().plot.bar(color = 'cyan')
plt.xticks([0, 1, 2, 3, 4], ['own and fully paid house', 'own and paying in installments', 
                            'rented', 'precarious', 'other'], rotation = 40)
plt.title('status of the house for missing values of "v2a1" in test')
plt.show()
# We can see that most of the households who don't have monthly rent payment generally own their house. But 
# we don't know the reason of missing values for the other situations.
train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0
test.loc[(test['tipovivi1'] == 1), 'v2a1'] = 0

train['v2a1'].isnull().value_counts()
test['v2a1'].isnull().value_counts()
# Since the rest of the missing values belong to "rez_esc" and "v2a1", which only contain integer values. 
# So we simply replace the remaining missing values to the median of the column values.
train.fillna(train.median(), inplace = True)
test.fillna(test.median(), inplace = True)

print(train.isnull().sum().sort_values(ascending = False).head())
print(test.isnull().sum().sort_values(ascending = False).head())
# ID_variables.
# We will keep these in the data since we need them for identification.
idv = ['Id', 'idhogar', 'Target']

# Individual variables.
ib = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
      'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2',  'parentesco3', 
      'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 
      'parentesco10', 'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
      'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'mobilephone']

io = ['rez_esc', 'escolari', 'age']

# Househould variables.
hb = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 'paredpreb','pisocemento', 
      'pareddes', 'paredmad','paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
      'pisonatur', 'pisonotiene', 'pisomadera','techozinc', 'techoentrepiso', 'techocane', 'techootro', 
      'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 
      'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6', 'energcocinar1', 
      'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
      'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 
      'eviv1', 'eviv2', 'eviv3', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
      'computer', 'television', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 
      'area2']

ho = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 'r4t3', 'v18q1', 
      'tamhog','tamviv','hhsize','hogar_nin', 'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 
      'qmobilephone']

hcn = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']

# Squared Variables.
sv = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 
      'SQBdependency', 'SQBmeaned', 'agesq']
# Sometimes variables are squared or transformed as part of feature engineering because it can help linear
# relationships that are non-linear. However, here we are going to use models that are more complex, these 
# squared features may be redundant since they are highly correlated with the non-squared features. This 
# means these squared will hurt the models badly by adding irrelevant information.
# We can take a look at the relationshop between the squared features and non-squared features.
sb.lmplot('edjefe', 'SQBedjefe', data = train)
plt.title('"edjefe" vs "SQBedjefe"')
plt.show()

sb.lmplot('dependency', 'SQBdependency', data = train)
plt.title('"dependency" vs "SQBdependency"')
plt.show()
# These features are highly correlated, we don't need to keep them both in the data. So I decide to delete 
# all of the squared variables.
train = train.drop(columns = sv)
test = test.drop(columns = sv)
# Household variables.
hhtrain = train.loc[train['parentesco1'] == 1, :]
hhtrain = hhtrain[idv + hb + ho + hcn]

hhtest = test.loc[test['parentesco1'] == 1, :]
hhtest = hhtest[['Id', 'idhogar'] + hb + ho + hcn]
# Then we need to check the correlations between these features. If there are some features highly 
# correlated, we have to delete one of the pair so that it will not cause data redundant.
cor_hh = hhtrain.corr()
uptr = np.triu(np.ones(cor_hh.shape), k = 1).astype(np.bool)
up = cor_hh.where(uptr)

a = [column for column in up.columns if any(abs(up[column]) > 0.9)]
print(a)

cor_hh.loc[cor_hh['tamviv'] > 0.9, cor_hh['tamviv'] > 0.9]
sb.heatmap(cor_hh.loc[cor_hh['tamviv'].abs() > 0.9, cor_hh['tamviv'].abs() > 0.9], annot = True)
# Based on the hearmap, we drop some of the features.
hhtrain = hhtrain.drop(columns = ['tamhog', 'r4t3', 'hogar_total', 'hhsize'])
hhtest = hhtest.drop(columns = ['tamhog', 'r4t3', 'hogar_total', 'hhsize'])
# Individual vriables.
itrain = train[idv + ib + io]

itest = test[['Id', 'idhogar'] + ib + io]
# Identify redundant features.
cor_i = itrain.corr()
uptr = np.triu(np.ones(cor_i.shape), k = 1).astype(np.bool)
up = cor_i.where(uptr)

b = [column for column in up.columns if any(abs(up[column]) > 0.9)]
print(b)
# This is related to the "male" feature, so we can move one of them. Here we remove the "male".
itrain = itrain.drop(columns = 'male')
itest = itest.drop(columns = 'male')
# Finally, we just need to aggregate the individual and household data, and we can run the model.
# Since the boolean aggregations can be the same, but this will create many redundant columns that need 
# to be deleted. We will do the aggregations and then go back to drop the redundant columns.
aggf = lambda x: x.max() - x.min()
aggf.__name__ = 'aggregation'

iagg_train = itrain.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', aggf])
iagg_test = itest.groupby('idhogar').agg(['min', 'max', aggf])
iagg_test.head()
# Rename the columns so that we can know their exact meaning.
rntrain = []
for realname in iagg_train.columns.levels[0]:
    for aggval in iagg_train.columns.levels[1]:
        rntrain.append(f'{realname}-{aggval}')
iagg_train.columns = rntrain

rntest = []
for realname in iagg_test.columns.levels[0]:
    for aggval in iagg_test.columns.levels[1]:
        rntest.append(f'{realname}-{aggval}')
iagg_test.columns = rntest
iagg_test.head()
# Check the correlations so that we can delete the redundant data.
cor_aggtrain = iagg_train.corr()

uptr_agg = cor_aggtrain.where(np.triu(np.ones(cor_aggtrain.shape), k = 1).astype(np.bool))
deletrain = [column for column in uptr_agg.columns if any(abs(uptr_agg[column])> 0.95)]
print(len(deletrain))
cor_aggtest = iagg_test.corr()

uptr_agg = cor_aggtest.where(np.triu(np.ones(cor_aggtest.shape), k = 1).astype(np.bool))
deletest = [column for column in uptr_agg.columns if any(abs(uptr_agg[column])> 0.95)]
print(len(deletest))
# Merge the data.
iagg_train = iagg_train.drop(columns = deletrain)
cleaned_train = hhtrain.merge(iagg_train, on = 'idhogar', how = 'left')
cleaned_train.shape
iagg_test = iagg_test.drop(columns = deletest)
cleaned_test = hhtest.merge(iagg_test, on = 'idhogar', how = 'left')
cleaned_test.shape
# First we use a simple method to establish a baseline accuracy based on kfold cross validation.
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
# Define the target labels.
y = np.array(cleaned_train['Target'])

# Since we are going to use different models to compare their performances, scaling the features would be 
# very necessary. Some distance metric-based models such as kNN and SVM may not be applied to the data if 
# the data is not scaled.
# Extract the training data.
train_set = cleaned_train.drop(columns = ['Id', 'idhogar', 'Target'])
test_set = cleaned_test.drop(columns = ['Id', 'idhogar'])
# Scale the data.
pipeline = Pipeline([('scaler', MinMaxScaler())])

train_set = pipeline.fit_transform(train_set)
test_set = pipeline.fit_transform(test_set)
# kNN.
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
acc_sco = make_scorer(accuracy_score)
f1_sco = make_scorer(f1_score, average = 'macro')

acc_knn = cross_val_score(knn_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_knn = cross_val_score(knn_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_knn))
print(np.mean(f1_knn))
# The accuracy increases with the increasing of n_neighbors.
accknn = []
f1knn = []
for i in range(4, 11):
    knn_classifier = KNeighborsClassifier(n_neighbors = i)
    acc_knn = cross_val_score(knn_classifier, train_set, y, cv = 10, scoring = acc_sco)
    f1_knn = cross_val_score(knn_classifier, train_set, y, cv = 10, scoring = f1_sco)
    a = np.mean(acc_knn)
    b = np.mean(f1_knn)
    accknn.append(a)
    f1knn.append(b)
    
print(accknn)
print(f1knn)
# After we get the baseline accuracy, we will use some of the advanced models to do further machine 
# learning.
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# Random Forest.
rf_classifier = RandomForestClassifier()
acc_rf = cross_val_score(rf_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_rf = cross_val_score(rf_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_rf))
print(np.mean(f1_rf))
accrf = []
f1rf = []
for i in range(1, 6):
    rf_classifier = RandomForestClassifier(max_depth = i)
    acc_rf = cross_val_score(rf_classifier, train_set, y, cv = 10, scoring = acc_sco)
    f1_rf = cross_val_score(rf_classifier, train_set, y, cv = 10, scoring = f1_sco)
    c = np.mean(acc_rf)
    d = np.mean(f1_rf)
    accrf.append(c)
    f1rf.append(d)
    
print(accrf)
print(f1rf)
# SVM.
svm_classifier = SVC(kernel = 'rbf')
acc_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_svm))
print(np.mean(f1_svm))
svm_classifier = SVC(kernel = 'linear')
acc_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_svm))
print(np.mean(f1_svm))
svm_classifier = SVC(kernel = 'poly')
acc_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_svm = cross_val_score(svm_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_svm))
print(np.mean(f1_svm))
# Naive Bayes.
nb_classifier = GaussianNB()
acc_nb = cross_val_score(nb_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_nb = cross_val_score(nb_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_nb))
print(np.mean(f1_nb))
# LDA.
lda_classifier = LinearDiscriminantAnalysis(n_components = 1)
acc_lda = cross_val_score(lda_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_lda = cross_val_score(lda_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_lda))
print(np.mean(f1_lda))
lda_classifier = LinearDiscriminantAnalysis(n_components = 2)
acc_lda2 = cross_val_score(lda_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_lda2 = cross_val_score(lda_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_lda2))
print(np.mean(f1_lda2))
# Logistic Regression.
lr_classifier = LogisticRegression()
acc_lr = cross_val_score(lr_classifier, train_set, y, cv = 10, scoring = acc_sco)
f1_lr = cross_val_score(lr_classifier, train_set, y, cv = 10, scoring = f1_sco)

print(np.mean(acc_lr))
print(np.mean(f1_lr))
all_model = pd.DataFrame(columns = ['model', 'avg_acc', 'avg_f1'])
all_model = all_model.append(pd.DataFrame({'model': 'kNN', 'avg_acc': np.mean(acc_knn), 
                                          'avg_f1': np.mean(f1_knn)}, index = [0]))
all_model = all_model.append(pd.DataFrame({'model': 'Random Forest', 'avg_acc': np.mean(acc_rf), 
                                          'avg_f1': np.mean(f1_rf)}, index = [0]))
all_model = all_model.append(pd.DataFrame({'model': 'SVM', 'avg_acc': np.mean(acc_svm), 
                                          'avg_f1': np.mean(f1_svm)}, index = [0]))
all_model = all_model.append(pd.DataFrame({'model': 'Naive Bayes', 'avg_acc': np.mean(acc_nb), 
                                          'avg_f1': np.mean(f1_nb)}, index = [0]))
all_model = all_model.append(pd.DataFrame({'model': 'LDA', 'avg_acc': np.mean(acc_lda), 
                                          'avg_f1': np.mean(f1_lda)}, index = [0]))
all_model = all_model.append(pd.DataFrame({'model': 'Logistic Regression', 'avg_acc': np.mean(acc_lr), 
                                          'avg_f1': np.mean(f1_lr)}, index = [0]))
print(all_model)
all_model.set_index('model', inplace = True)
all_model['avg_acc'].plot.bar(color = 'orange', figsize = (8, 6), yerr = list(all_model['avg_acc']))
plt.title('Model Average Accuracy Comparision')
plt.ylabel('Average Accuracy with F1 Score')
all_model.reset_index(inplace = True)


