!pip install -U -q fancyimpute
# import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

import os

import fancyimpute

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/telcocustchurniykra/data.csv')

df.head()
# df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# df.head()
df.describe()
df.info()
df.isnull().sum()
for col in df.columns:

    print("%s unique count: %d" % (col, df[col].nunique()))

    print(df[col].unique(), '\n')
# drop customerID

df = df.drop(['customerID'], axis=1)
# replace 'No phone service' menjadi 'No' pada MultipleLines

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})



# replace 'No internet service' menjadi 'No'

cols_redundant = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 

                  'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in cols_redundant:

    df[col] = df[col].replace({'No internet service': 'No'})
cols_cat = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',

            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',

            'PaperlessBilling', 'Churn']



# replace 'Yes' menjadi 1 dan 'No' menjadi 0

for col in cols_cat:

    df[col] = df[col].replace({'Yes': 1, 'No': 0})
# replace 'female' menjadi 1 dan 'male' menjadi 0

df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
# Pada atribut InternetService

## Replace 'DSL' menjadi 1

## Replace 'Fiber optic' menjadi 2

## Replace 'No' menjadi 3

df['InternetService'] = df['InternetService'].replace({'DSL': 1,

                                                       'Fiber optic': 2,

                                                       'No': 3})
# Pada atribut Contract

##abs Replace 'Month-to-month' menjadi 1

## Replace 'One year' menjadi 2

## Replace 'Two year' menjadi 3

df['Contract'] = df['Contract'].replace({'Month-to-month': 1,

                                         'One year': 2,

                                         'Two year': 3})
# Pada atribut PaymentMethod

## Replace 'Electronic check' menjadi 1

## Replace 'Mailed check' menjadi 2

## Replace 'Bank transfer (automatic)' menjadi 3

## Replace 'Credit card (automatic)' menjadi 4

df['PaymentMethod'] = df['PaymentMethod'].replace({'Electronic check': 1,

                                                   'Mailed check': 2,

                                                   'Bank transfer (automatic)': 3,

                                                   'Credit card (automatic)': 4})
# df['TotalCharges'] = df['TotalCharges'].astype('float64')
df.TotalCharges.isnull().sum()
# cek data berisi spasi pada TotalCharges

temp = df.sort_values('TotalCharges') 

tcharges = temp.TotalCharges[temp['TotalCharges'] == ' ']

print("Jumlah data kosong: ", len(tcharges))

tcharges
# terdapat karakter spasi pada data

temp = df.sort_values('TotalCharges')['TotalCharges'][:15][936] 

temp, len(temp)
# convert space menjadi NaN

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# setelah dirubah ke NaN, data kosong bisa diketahui dengan fungsi isnull()

print("Data kosong pada TotalCharges:", df.TotalCharges.isnull().sum())



# isi data kosong pada TotalCharges dengan formula di atas

df['TotalCharges'].fillna(value=df['tenure'] * df['MonthlyCharges'], inplace=True)

print("Data kosong pada TotalCharges:", df.TotalCharges.isnull().sum()) # cek lagi data kosong
df.info()
# initialize

imputer = fancyimpute.KNN()



# impute data and convert 

encode_data = pd.DataFrame(np.round(imputer.fit_transform(df)),columns = df.columns)
cols_impute = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 

               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 

               'StreamingTV', 'StreamingMovies']



for col in cols_impute:

    data_before = df[col].value_counts(normalize=True)

    data_after = encode_data[col].value_counts(normalize=True)

    

    combine_data = pd.DataFrame()

    combine_data['percentage_before'] = data_before

    combine_data['percentage_after'] = data_after

    print(col)

    print(combine_data)

    print()
# cek lagi null data

encode_data.isnull().any().sum()
cols_float = ['MonthlyCharges', 'TotalCharges']

cols_int = [x for x in list(encode_data.columns) if not x in cols_float]



encode_data[cols_float] = encode_data[cols_float].astype('float64')

encode_data[cols_int] = encode_data[cols_int].astype('int64')

encode_data.info()
df_viz = encode_data.copy()
cols_cat = ['SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 

            'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 

            'StreamingMovies','PaperlessBilling', 'Churn']



for col in cols_cat:

    df_viz[col] = df_viz[col].replace({1:'Yes', 0:'No'})



# gender

df_viz['gender'] = df_viz['gender'].replace({1:'Male', 0:'Female'})



# InternetService

df_viz['InternetService'] = df_viz['InternetService'].replace({1:'DSL',

                                                               2:'Fiber optic',

                                                               3:'No'})

# Contract

df_viz['Contract'] = df_viz['Contract'].replace({1:'Month-to-month',

                                                 2:'One year',

                                                 3:'Two year'})

# PaymentMethod

df_viz['PaymentMethod'] = df_viz['PaymentMethod'].replace({1:'Electronic check',

                                                           2:'Mailed check',

                                                           3:'Bank transfer (automatic)',

                                                           4:'Credit card (automatic)'})
df_viz.head()
# statistical summary

df_viz.describe().round(3)
df_viz[df_viz['Churn'] == 'Yes']['MonthlyCharges'].mean()
df_viz[df_viz['Churn'] == 'No']['TotalCharges'].mean()
df_num = df_viz[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
plt.figure(figsize=(12,8))

sns.lmplot(x='tenure', y='TotalCharges', col='Churn', hue='Churn', data=df_num)

# plt.title('tenure vs TotalCharges')
plt.figure(figsize=(12,8))

sns.lmplot(x='MonthlyCharges', y='TotalCharges', col='Churn', hue='Churn', data=df_num)

# plt.title('MonthlyCharges vs TotalCharges')
plt.figure(figsize=(12,8))

sns.lmplot(x='tenure', y='MonthlyCharges', col='Churn', hue='Churn', data=df_num)

# plt.title('MonthlyCharges vs TotalCharges')
plt.figure(figsize = (15,10))

sns.pairplot(df_num, hue='Churn')
plt.figure(figsize=(17,10))



sns.boxplot(x='tenure', y='TotalCharges', data=df_num)



plt.title('Box Plot of Tenure X TotalCharges')
plt.figure(figsize = (17,10))

sns.countplot(df_viz['tenure'])
df_viz.describe(include='object')
# pie chart

cols_cat = df_viz.describe(include='object').columns



fig = plt.figure(figsize=(16,35), dpi=80)

plt.subplots_adjust(hspace=.35)



for idx, item in enumerate(cols_cat, 1):

    data = df_viz[item].value_counts().to_frame().T

    labels = data.columns

    title = df_viz[item].name

    

    fig.add_subplot(6,3,idx)

    plt.title(title, fontsize=14)

    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})

    plt.axis('equal')
cat_services = ['PhoneService','MultipleLines','InternetService',

                'OnlineSecurity', 'OnlineBackup','DeviceProtection',

                'TechSupport','StreamingTV','StreamingMovies']



fig = plt.figure(figsize=(18,18), dpi=80)

plt.subplots_adjust(hspace=.35)



for idx, col in enumerate(cat_services, 1):

    fig.add_subplot(3,3,idx)

    plt.title(f'{col} Counts', fontsize=16)

    countplot_sc = sns.countplot(df_viz[col], hue=df_viz.Churn)

    

    # text annotation

    for p in countplot_sc.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        countplot_sc.annotate('{:.2f}%'.format(100.*y/len(df_viz[col])), 

                              (x.mean(), y),

                              ha='center',

                              va='bottom') # set the alignment of the text
cat_custinfo = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']



fig = plt.figure(figsize=(12,10), dpi=80)

plt.subplots_adjust(hspace=.35)



for idx, col in enumerate(cat_custinfo, 1):

    fig.add_subplot(2,2,idx)

    plt.title(f'{col} Counts', fontsize=16)

    countplot_sc = sns.countplot(df_viz[col], hue=df_viz.Churn)

    

    # text annotation

    for p in countplot_sc.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        countplot_sc.annotate('{:.2f}%'.format(100.*y/len(df_viz[col])), 

                              (x.mean(), y),

                              ha='center',

                              va='bottom') # set the alignment of the text
cat_other = ['Contract', 'PaperlessBilling', 'PaymentMethod']



fig = plt.figure(figsize=(12,10), dpi=80)

plt.subplots_adjust(hspace=.35)



for idx, col in enumerate(cat_other, 1):

    fig.add_subplot(2,2,idx)

    plt.title(f'{col} Counts', fontsize=16)

    countplot_sc = sns.countplot(df_viz[col], hue=df_viz.Churn)

    

    if idx == 3:

        countplot_sc.set_xticklabels(countplot_sc.get_xticklabels(), rotation=20)

    

    # text annotation

    for p in countplot_sc.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        countplot_sc.annotate('{:.2f}%'.format(100.*y/len(df_viz[col])), 

                              (x.mean(), y),

                              ha='center',

                              va='bottom') # set the alignment of the text
data = df_viz.copy()

data.head()
# binary encoding

gender_map = {"Female" : 0, "Male": 1}

yes_no_map = {"Yes" : 1, "No" : 0}



data["gender"] = data["gender"].map(gender_map)



binary_encode_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 

                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 

                      'StreamingMovies', 'PaperlessBilling', 'Churn']



for col in binary_encode_cols:

    data[col] = data[col].map(yes_no_map)
data.head()
# one hot encoding

onehot_encode_cols = ['InternetService', 'Contract', 'PaymentMethod']

data_onehot_encode = pd.get_dummies(data[onehot_encode_cols])



# join

data = data.drop(onehot_encode_cols, axis=1)

data = pd.concat([data, data_onehot_encode], axis=1)
# change data type from category to uint8

for col in data.columns:

    if str(data[col].dtypes) == 'category':

        data[col] = data[col].astype('uint8')
plt.figure(figsize = (15, 15))

sns.heatmap(data.corr(), cmap="RdYlBu", annot=True, fmt=".1f")

plt.show()
data.corr()['Churn'].sort_values(ascending=False)