import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

import math

import seaborn as sns



%matplotlib inline
def combine_train_test(train_raw, test_raw):

    train_ds = train_raw[["Pclass", "Name", "Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]];

    full_ds = pandas.concat([train_ds, test_raw])

    

    return full_ds;



def get_train_ds(full_ds, train_raw): 

    return full_ds[full_ds.index.isin(train_raw.index)]

    

def get_test_ds(full_ds, test_raw):

    return full_ds[full_ds.index.isin(test_raw.index)]

#load training data

train_raw = pandas.read_csv('../input/train.csv', sep=',', index_col=0)



#load test data

test_raw = pandas.read_csv('../input/test.csv', sep=',', index_col=0)



full_ds = combine_train_test(train_raw, test_raw)

train_label = train_raw["Survived"]; 



print(full_ds[0:10])

print(full_ds.dtypes)



print("\nNull value summary:")

print(full_ds.isnull().sum())

print("\nEmpty age sample:")

print(full_ds[pandas.isnull(full_ds["Age"])][0:10])

print("\nEmpty fare sample:")

print(full_ds[pandas.isnull(full_ds["Fare"])][0:10])

print("\nEmpty embarked sample:")

print(full_ds[pandas.isnull(full_ds["Embarked"])][0:10])
# Enhance the data-set - train set

def convert_name_to_salutation(name):

    name_t = name.lower()

    salutations = ['mrs.', 'mr.', 'ms.', 'mlle.', 'miss.', 'sir.', 'rev.', 'mme.', 'master.', 'major.',

                  'lady.', 'jonkheer.', 'dr.', 'don.', 'col.', 'capt.', 'countess.']

    

    for sal in salutations:

        if sal in name_t:

            return sal

        

    return 'none'

    

def convert_cabin_to_area(cabin):

    areas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

    cabin_t = str(cabin).upper().strip()

    

    for area in areas:

        if cabin_t.startswith(area):

            return area

    

    return 'none'



def discrete_age(age):

    if math.isnan(age):

        return 0

    for i in range(0, 101, 10):

        if(age < i):

            return i

        

def discrete_fare(fare):

    if math.isnan(fare):

        return 0

    for i in range(0, 601, 50):

        if(fare < i):

            return i



def add_salutation_col(df):

    df['Salutation'] = df['Name'].map(lambda x: convert_name_to_salutation(x))

    

def add_carbin_area_col(df):

    df['CabinArea'] = df['Cabin'].map(lambda x: convert_cabin_to_area(x))

    

def add_family_member_col(df):

    df['FamilyMember'] = df['SibSp'] + df['Parch']

    

add_salutation_col(full_ds)

add_carbin_area_col(full_ds)

add_family_member_col(full_ds)



print(full_ds[0:10])
from sklearn import preprocessing



saluation_enc = preprocessing.LabelEncoder()

saluation_enc.fit(full_ds["Salutation"])

full_ds["Salutation_enc"] = saluation_enc.transform(full_ds["Salutation"])



carbin_enc = preprocessing.LabelEncoder()

carbin_enc.fit(full_ds["CabinArea"])

full_ds["CabinArea_enc"] = carbin_enc.transform(full_ds["CabinArea"])



sex_enc = preprocessing.LabelEncoder()

sex_enc.fit(full_ds["Sex"])

full_ds["Sex_enc"] = sex_enc.transform(full_ds["Sex"])



print(full_ds[0:10])
def impute_missing (data_1_group, target):

    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean')

    imp=imp.fit(data_1_group[target].values.reshape(-1, 1))

    data_1_group[target]=imp.transform(data_1_group[target].values.reshape(-1, 1)).reshape((-1, 1))

    return(data_1_group)



def impute_by_group(dataset, groupby, target):

    result = pandas.DataFrame(columns=dataset.columns)

    result.index.name = dataset.index.name

    for grp_num, data_1_group in full_ds.groupby(groupby):

        imputed_group = impute_missing(data_1_group, target)

        result = pandas.concat([result, imputed_group])

    result = result.convert_objects()

    return result
# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(full_ds.corr(),  vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#Missing age: the most correlated field is salutation.

full_ds[["Salutation", "Age"]].boxplot(by="Salutation", figsize=(12, 4))

full_ds = impute_by_group(full_ds, "Salutation", "Age")

print(full_ds[full_ds.index.isin([6,18,20,27,29,30,32,33,37,43])])



#Missing Fare: the most correlated fields is cabin

full_ds[["CabinArea", "Fare"]].boxplot(by="CabinArea", figsize=(12, 4))

full_ds = impute_by_group(full_ds, "CabinArea", "Fare")

print(full_ds[full_ds.index.isin([1044])])



#Missing embark

full_ds[["Embarked", "Fare"]].boxplot(by="Embarked", figsize=(12, 4))

full_ds.set_value(62, 'Embarked', 'C')

full_ds.set_value(830, 'Embarked', 'C')

print(full_ds[full_ds.index.isin([62,830])])



embarked_enc = preprocessing.LabelEncoder()

embarked_enc.fit(full_ds["Embarked"])

full_ds["Embarked_enc"] = embarked_enc.transform(full_ds["Embarked"])

train_ds = get_train_ds(full_ds, train_raw)

train_ds["Survived"] = train_label



fig = plt.figure(figsize=(12,4))

train_ds["Survived"].value_counts().sort_index().plot.bar()



# Visualize data and the corresponding labels

fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["Pclass"].value_counts().sort_index().plot.bar()

ax1.set_title('P class - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["Pclass"].value_counts().sort_index().plot.bar()

ax2.set_title('P class - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["Pclass"].value_counts().sort_index().plot.bar()

ax3.set_title('P class - Dead')

plt.suptitle('P class statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["Sex"].value_counts().sort_index().plot.bar()

ax1.set_title('Sex - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["Sex"].value_counts().sort_index().plot.bar()

ax2.set_title('Sex - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["Sex"].value_counts().sort_index().plot.bar()

ax3.set_title('Sex - Dead')

plt.suptitle('Sex statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(141)

train_ds["Age"].value_counts().sort_index().plot.line()

ax1.set_title('Age - All')



ax2 = fig.add_subplot(142, sharey=ax1)

train_ds[train_ds.Survived==1]["Age"].value_counts().sort_index().plot.line()

ax2.set_title('Age - Survive')



ax3 = fig.add_subplot(143, sharey=ax1)

train_ds[train_ds.Survived==0]["Age"].value_counts().sort_index().plot.line()

ax3.set_title('Age - Dead')



ax4 = fig.add_subplot(144)

train_ds[["Age","Survived"]].boxplot(by='Survived', ax=ax4)

ax4.set_title('Age - boxplot')



plt.suptitle('Age statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["SibSp"].value_counts().sort_index().plot.bar()

ax1.set_title('SibSp - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["SibSp"].value_counts().sort_index().plot.bar()

ax2.set_title('SibSp - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["SibSp"].value_counts().sort_index().plot.bar()

ax3.set_title('SibSp - Dead')



plt.suptitle('SibSp statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["Parch"].value_counts().sort_index().plot.bar()

ax1.set_title('Parch - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["Parch"].value_counts().sort_index().plot.bar()

ax2.set_title('Parch - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["Parch"].value_counts().sort_index().plot.bar()

ax3.set_title('Parch - Dead')



plt.suptitle('Parch statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["FamilyMember"].value_counts().sort_index().plot.bar()

ax1.set_title('FamilyMember - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["FamilyMember"].value_counts().sort_index().plot.bar()

ax2.set_title('FamilyMember - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["FamilyMember"].value_counts().sort_index().plot.bar()

ax3.set_title('FamilyMember - Dead')



plt.suptitle('FamilyMember statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(141)

train_ds["Fare"].value_counts().sort_index().plot.line()

ax1.set_title('Fare - All')



ax2 = fig.add_subplot(142, sharey=ax1)

train_ds[train_ds.Survived==1]["Fare"].value_counts().sort_index().plot.line()

ax2.set_title('Fare - Survive')



ax3 = fig.add_subplot(143, sharey=ax1)

train_ds[train_ds.Survived==0]["Fare"].value_counts().sort_index().plot.line()

ax3.set_title('Fare - Dead')



ax4 = fig.add_subplot(144)

train_ds[["Fare","Survived"]].boxplot(by='Survived', ax=ax4)

ax4.set_title('Fare - boxplot')



plt.suptitle('Fare statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["Embarked"].value_counts().sort_index().plot.bar()

ax1.set_title('Embarked - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["Embarked"].value_counts().sort_index().plot.bar()

ax2.set_title('Embarked - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["Embarked"].value_counts().sort_index().plot.bar()

ax3.set_title('Embarked - Dead')



plt.suptitle('Embarked statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["Salutation"].value_counts().sort_index().plot.bar()

ax1.set_title('Salutation - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["Salutation"].value_counts().sort_index().plot.bar()

ax2.set_title('Salutation - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["Salutation"].value_counts().sort_index().plot.bar()

ax3.set_title('Salutation - Dead')



plt.suptitle('Salutation statistic')



fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(131)

train_ds["CabinArea"].value_counts().sort_index().plot.bar()

ax1.set_title('CabinArea - All')



ax2 = fig.add_subplot(132, sharey=ax1)

train_ds[train_ds.Survived==1]["CabinArea"].value_counts().sort_index().plot.bar()

ax2.set_title('CabinArea - Survive')



ax3 = fig.add_subplot(133, sharey=ax1)

train_ds[train_ds.Survived==0]["CabinArea"].value_counts().sort_index().plot.bar()

ax3.set_title('CabinArea - Dead')



plt.suptitle('CabinArea statistic')
sns.set(style="ticks")

train_raw_pair = train_ds[["Survived", "Pclass", "Sex_enc", "SibSp", \

                        "Parch", "FamilyMember", "Embarked_enc", "Salutation_enc", "CabinArea_enc", "Age", "Fare"]]



print(train_raw_pair[0:10])



print("\nNull value summary:")

print(train_raw_pair.isnull().sum())



colors = ["red", "blue"]

sns.pairplot(train_raw_pair, hue="Survived", 

             vars=["Pclass", "Sex_enc", "SibSp", "Parch", "FamilyMember", "Embarked_enc", "Salutation_enc", "CabinArea_enc", "Age", "Fare"], 

             plot_kws=dict(alpha=.2), palette=sns.xkcd_palette(colors))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from six.moves import cPickle as pickle



#create train and test dataset

train_dataset = get_train_ds(full_ds, train_raw)

test_dataset = get_train_ds(full_ds, test_raw)

train_dataset.sort_index(inplace=True)

test_dataset.sort_index(inplace=True)



#create train label

train_label = train_raw["Survived"]



''' Uncomment it if you want to save

try:

    set_filename = "../input/train_dataset.pickle"

    with open(set_filename, 'wb') as f:

        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)

except Exception as e:

    print('Unable to save data to', set_filename, ':', e)

    

try:

    set_filename = "../input/test_dataset.pickle"

    with open(set_filename, 'wb') as f:

        pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)

except Exception as e:

    print('Unable to save data to', set_filename, ':', e)

    

try:

    set_filename = "../input/train_label.pickle"

    with open(set_filename, 'wb') as f:

        pickle.dump(train_label, f, pickle.HIGHEST_PROTOCOL)

except Exception as e:

    print('Unable to save data to', set_filename, ':', e)

'''
columns = ["Pclass", "Embarked_enc", "Salutation_enc", "CabinArea_enc"]

train_ds_onehot = train_dataset[["Pclass", "Sex_enc", "SibSp", "Parch", "Fare", "CabinArea_enc",\

                                   "Embarked_enc", "Salutation_enc", "FamilyMember"]]

train_ds_onehot = pandas.get_dummies(train_ds_onehot, sparse=True, columns=columns)

print(train_ds_onehot[0:10])



scaler = preprocessing.StandardScaler().fit(train_ds_onehot)

train_ds_onehot_scaled = scaler.transform(train_ds_onehot) 



print(pandas.DataFrame(train_ds_onehot_scaled[0:10]))
def transform_ds_to_input(dataset):

    columns = ["Pclass", "Embarked_enc", "Salutation_enc", "CabinArea_enc"]

    ds_onehot = dataset[["Pclass", "Sex_enc", "SibSp", "Parch", "Fare", "CabinArea_enc",\

                                       "Embarked_enc", "Salutation_enc", "FamilyMember"]]

    ds_onehot = pandas.get_dummies(ds_onehot, sparse=True, columns=columns)

    scaler = preprocessing.StandardScaler().fit(ds_onehot)

    ds_onehot_scaled = scaler.transform(ds_onehot) 

    return ds_onehot_scaled



train_ds_onehot_scaled = transform_ds_to_input(train_dataset)

test_ds_onehot_scaled = transform_ds_to_input(test_dataset)
seed = 10

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(train_ds_onehot_scaled, train_label, test_size=test_size)



#clf = RandomForestClassifier(n_estimators=5, max_features=3, max_depth=3, min_samples_split=2, random_state=0)

clf = RandomForestClassifier()     

clf.fit(X_train, y_train)



# make predictions for test data

y_pred = clf.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.decomposition import PCA

#study pca 



pca = PCA()

pca.fit(train_ds_onehot_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)



plt.plot([i for i in range(len(cumsum))],cumsum,'b-')

plt.axhline(y=0.95, xmin=0, xmax=1, hold=None, c='k', linestyle='--')

plt.axvline(x=26, ymin=0, ymax=0.9, hold=None, c='k', linestyle='--')
import sklearn.manifold



pca = PCA(n_components=26)

train_ds_reduce = pca.fit_transform(train_ds_onehot_scaled)



tsne = sklearn.manifold.TSNE()

train_tsne = pandas.DataFrame(tsne.fit_transform(train_ds_reduce))

train_tsne["Survived"] = train_label
groups = train_tsne.groupby('Survived')



# Plot

fig, ax = plt.subplots()

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in groups:

    ax.plot(group[0], group[1], marker='o', linestyle='', ms=5, label=name)

ax.legend()



plt.show()