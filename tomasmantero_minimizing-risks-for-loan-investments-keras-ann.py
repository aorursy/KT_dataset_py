# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# scaling and train test split

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# creating a model

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.constraints import max_norm

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model



# evaluation on test data

from sklearn.metrics import classification_report,confusion_matrix
data_info = pd.read_csv('../input/subset-lending-club-loan/lending_club_info.csv',index_col='LoanStatNew')



def feat_info(col_name):

    print(data_info.loc[col_name]['Description'])



# example

feat_info('mort_acc')
df = pd.read_csv('../input/subset-lending-club-loan/lending_club_loan_two.csv')
print(df.info())
df.head()
df.describe().transpose()
sns.set(style="whitegrid", font_scale=1)



plt.figure(figsize=(12,12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',

            annot=True, annot_kws={"size":10}, cbar_kws={"shrink": .7})
f, axes = plt.subplots(1, 2, figsize=(15,5))

sns.countplot(x='loan_status', data=df, ax=axes[0])

sns.distplot(df['loan_amnt'], kde=False, bins=40, ax=axes[1])

sns.despine()

axes[0].set(xlabel='Status', ylabel='')

axes[0].set_title('Count of Loan Status', size=20)

axes[1].set(xlabel='Loan Amount', ylabel='')

axes[1].set_title('Loan Amount Distribution', size=20)
f, axes = plt.subplots(1, 2, figsize=(15,5))

sns.scatterplot(x='installment', y='loan_amnt', data=df, ax=axes[0])

sns.boxplot(x='loan_status', y='loan_amnt', data=df, ax=axes[1])

sns.despine()

axes[0].set(xlabel='Installment', ylabel='Loan Amount')

axes[0].set_title('Scatterplot between Loan Amount and Installment', size=15)

axes[1].set(xlabel='Loan Status', ylabel='Loan Amount')

axes[1].set_title('Boxplot between Loan Amount and Loan Status', size=15)
df.groupby('loan_status')['loan_amnt'].describe()
f, axes = plt.subplots(1, 2, figsize=(15,5), gridspec_kw={'width_ratios': [1, 2]})

sns.countplot(x='grade', hue='loan_status', data=df, order=sorted(df['grade'].unique()), palette='seismic', ax=axes[0])

sns.countplot(x='sub_grade', data=df, palette='seismic', order=sorted(df['sub_grade'].unique()), ax=axes[1])

sns.despine()

axes[0].set(xlabel='Grade', ylabel='Count')

axes[0].set_title('Count of Loan Status per Grade', size=20)

axes[1].set(xlabel='Sub Grade', ylabel='Count')

axes[1].set_title('Count of Loan Status per Sub Grade', size=20)

plt.tight_layout()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df[['loan_repaid','loan_status']].head()
df.corr()['loan_repaid'].sort_values(ascending=True).drop('loan_repaid').plot.bar(color='green')
print(len(df))
df.isnull().sum()
feat_info('emp_title')

print('\n')

feat_info('emp_length')

print('\n')

feat_info('title')

print('\n')

feat_info('revol_util')

print('\n')

feat_info('mort_acc')

print('\n')

feat_info('pub_rec_bankruptcies')
plt.figure(figsize=(10,5))

((df.isnull().sum())/len(df)*100).plot.bar(title='Percentage of missing values per column', color='green')
print(df['emp_title'].nunique())

df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)
per_charge_off = df[df['loan_repaid'] == 0]['emp_length'].value_counts() / df[df['loan_repaid'] == 1]['emp_length'].value_counts()

per_charge_off.plot.bar(color='green')
df = df.drop('emp_length', axis=1)
df[['title', 'purpose']].head(10)
df = df.drop('title', axis=1)
print("Mean of mort_acc column per total_acc")

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

print(total_acc_avg)
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']



def fill_mort_acc(total_acc,mort_acc):

    '''

    Accepts the total_acc and mort_acc values for the row.

    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value

    for the corresponding total_acc value for that row.

    

    total_acc_avg here should be a Series or dictionary containing the mapping of the

    groupby averages of mort_acc per total_acc values.

    '''

    if np.isnan(mort_acc):

        return total_acc_avg[total_acc]

    else:

        return mort_acc

    

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df = df.dropna()
# check for missing values

df.isnull().sum()
print(df['term'].value_counts())

print('\n')

print('\n')



df['term'] = df['term'].apply(lambda term: int(term[:3]))



print(df['term'].value_counts())
df = df.drop('grade', axis=1)
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)



df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)



df = pd.concat([df,dummies],axis=1)
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)

df = df.drop('home_ownership',axis=1)

df = pd.concat([df,dummies],axis=1)
df['zip_code'] = df['address'].apply(lambda address:address[-5:])



dummies = pd.get_dummies(df['zip_code'],drop_first=True)

df = df.drop(['zip_code','address'],axis=1)

df = pd.concat([df,dummies],axis=1)
df = df.drop('issue_d', axis=1)
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))

df = df.drop('earliest_cr_line', axis=1)



df.select_dtypes(['object']).columns
df = df.drop('loan_status',axis=1)
# Features

X = df.drop('loan_repaid',axis=1).values



# Label

y = df['loan_repaid'].values



# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
scaler = MinMaxScaler()



# fit and transfrom

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# everything has been scaled between 1 and 0

print('Max: ',X_train.max())

print('Min: ', X_train.min())
model = Sequential()



# input layer

model.add(Dense(78,activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(39,activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(19,activation='relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(1, activation='sigmoid'))



# compile model

model.compile(optimizer="adam", loss='binary_crossentropy')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 

          y=y_train, 

          epochs=400,

          verbose = 2,

          batch_size=256,

          validation_data=(X_test, y_test),

          callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)



plt.figure(figsize=(15,5))

sns.lineplot(data=losses,lw=3)

plt.xlabel('Epochs')

plt.ylabel('')

plt.title('Training Loss per Epoch')

sns.despine()
predictions = model.predict_classes(X_test)



print('Classification Report:')

print(classification_report(y_test, predictions))

print('\n')

print('Confusion Matirx:')

print(confusion_matrix(y_test, predictions))
rnd.seed(101)

random_ind = rnd.randint(0,len(df))



new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer
# we need to reshape this to be in the same shape of the training data that the model was trained on

model.predict_classes(new_customer.values.reshape(1,78))
# the prediction was right

df.iloc[random_ind]['loan_repaid']