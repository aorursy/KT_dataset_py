# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#For statistics

from scipy import stats

#For DataProcessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder



#For Machine Learning

from sklearn.model_selection import train_test_split

from sklearn import linear_model



#For Neural Network with TensorFlow



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from __future__ import absolute_import, division, print_function



import pathlib

import os

print(os.listdir("../input"))



print('TensorFlow version is:',tf.__version__)



# Any results you write to the current directory are saved as output.
#Read the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#print the dtypes in train to see how we can separate the quantitative from the qualitative

print("All dtypes in train are",train.dtypes.unique())



#separate the data into quantitative and qualitative

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

target = 'SalePrice'

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']



#For categorical data, we assign a value 'MISSING' to it to make it a new category

for c in qualitative:

    if train[c].isnull().any():

        train[c] = train[c].fillna('MISSING')

    if test[c].isnull().any():

        test[c] = test[c].fillna('MISSING')





def rank_qual(frame, feature):

    ordering = pd.DataFrame()

    ordering['val'] = frame[feature].unique()

    ordering.index = ordering['val'].values

    ordering.pop('val')

    ordering['spmean'] = frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']

    ordering = ordering.sort_values(by = 'spmean')

    ordering['rank'] = range(1,ordering.shape[0]+1)

    return ordering



def encode_rank(df,feature,ordering):

    """encode a categoraical feature based on rank

        the encoded feature becomes a new column featire+'_E' added to df

    df: DataFrame, the DataFrame with categorical feature 

    feature: str, the column name to encode

    ordering: DataFrame, a dictionary in df form that maps categoraical values to ranks

    

    """

    ordering = ordering['rank'].to_dict()

    for f_val, rank in ordering.items():

        df.loc[df[feature] == f_val, feature+'_E'] = rank

        

qual_encoded=[]        

for feature in qualitative:

    

    ordering = rank_qual(train,feature)



    encode_rank(train,feature,ordering)

    encode_rank(test,feature,ordering)

    qual_encoded.append(feature +'_E') 

    













# def encode(frame, feature):

#     ordering = pd.DataFrame()

#     ordering['val'] = frame[feature].unique()

#     ordering.index = ordering['val']

#     ordering['spmean'] = frame[[feature,target]].groupby(feature).mean()[target]

#     ordering = ordering.sort_values(by = 'spmean')

#     ordering['rank'] = range(1,ordering.shape[0]+1)

#     ordering = ordering['rank'].to_dict()

#     for f_val, rank in ordering.items():

#         frame.loc[frame[feature] == f_val, feature+'_E'] = rank

# qual_encoded=[]

# for f in qualitative:

#     encode(train,f)

#     qual_encoded.append(f+'_E') 

    

#the error used by this competition is the root-mean-squared error of log

def error(actual, hat):

    return np.sqrt(np.sum(np.square(np.log(actual)-np.log(hat)))/len(actual))





#Check the missing

def barplot_missing(df):

    #barplot and list the total number and percentage

    total = df.isnull().sum(axis=0)

    percent = 100*total /df.isnull().count()

    missing = pd.concat([total, percent], axis =1, keys = ['Total','Percent(%)'])

    missing = missing[missing['Total']>0]

    missing = missing.sort_values(by='Total', ascending=False)

    missing['Percent(%)'].plot.bar()

    return missing

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

missing = barplot_missing(train)

print('All missing data in training set are quantitative. Well done!' if(missing.index).all() in quantitative else 'Some qual_encoded in train is still missing. Check it!')

plt.subplot(1,2,2)

missing = barplot_missing(test)

print('All missing data in testing set are quantitative. Well done!' if (missing.index).all() in quantitative else 'Some qual_encoded in test is still missing. Check it!')
#Check the qual_encoded columns in testing data

missing_qual_E = [ c for c in missing.index if c in qual_encoded]

missing.loc[missing_qual_E]
#Check MSZoning column to see if the unique values in training and testing datasets are the same

print('Unique values in training data for MSZoning:',train['MSZoning'].unique())

print('Unique values in test data for MSZoning:',test['MSZoning'].unique())

#check to see if there's any NaN in train and test

print(train['MSZoning'].isnull().any(),test['MSZoning'].isnull().any())
#Assign avg rank to each missing encoded feature

mean = train[missing_qual_E].mean()

for c in missing_qual_E:

    avg = mean[c]

    test.loc[:,c]=test[c].fillna(avg)

ax1 = plt.subplot(1,2,1)

missing_num_train=barplot_missing(train)

plt.ylabel('Percent(%)')



plt.subplot(1,2,2,sharey=ax1)

missing_num_test= barplot_missing(test)

#check if any LotArea is zero

print(np.where(train['LotArea']==0))

print(np.where(test['LotArea']==0))



#check the min value of LotArea, LotFrontage

print('The min of LotArea/LotFrontage in train is:' ,train['LotArea'].min(),'/', train['LotFrontage'].min(),'\n',

'The min of LotAreaLot/Frontage in test is:',test['LotArea'].min(),'/', test['LotFrontage'].min())

#train.loc[np.where(train['LotArea']==0)[0],['LotArea','LotFrontage']]
def intersection(s1,s2):

    return list(set(s1)&set(s2))



LotArea_Positive = np.where(train['LotArea']>0)

LotFrontage_missing = np.where(train['LotFrontage'].isnull())



condition = intersection(LotArea_Positive[0],LotFrontage_missing[0])

print('The number of rows that  LotArea>0 but LotFrontage is missing is:',len(condition) )

#train.loc[condition, ['LotArea','LotFrontage']]
plt.figure(figsize=(15,5))

ax1 = plt.subplot(1,3,1)

sns.scatterplot(train['LotFrontage'],train['SalePrice'])



ax2 =plt.subplot(1,3,2,sharey=ax1)

sns.scatterplot(train['LotArea'],train['SalePrice'])





ax3= plt.subplot(1,3,3)

sns.scatterplot(train['LotFrontage'],train['LotArea'])



plt.figure(figsize=(15,5))

ax4 = plt.subplot(1,3,1)

sns.scatterplot(np.log(train['LotFrontage']),np.log(train['SalePrice']))



ax5 =plt.subplot(1,3,2)

sns.scatterplot(np.log(train['LotArea']),np.log(train['SalePrice']))





ax6= plt.subplot(1,3,3)

sns.scatterplot(np.log(train['LotFrontage']),np.log(train['LotArea']))

fr = pd.concat([train['LotArea'], train['LotFrontage'],

                train['SalePrice'],

                train['LotConfig_E'],train['LotShape_E'],

                train['LotConfig'], train['LotShape']],axis=1)

fr['LotArea_log'] = np.log(train['LotArea'])

fr['LotFrontage_log'] = np.log(train['LotFrontage'])

fr['SalePrice_log'] = np.log(train['SalePrice'])

numeric = ['LotArea','LotFrontage','SalePrice','LotConfig_E','LotShape_E','LotArea_log','LotFrontage_log','SalePrice_log']

fr[numeric].head()

corr = fr[numeric].corr()

sns.heatmap(corr, cmap='RdBu_r',vmin=-1,vmax=1)

df1=train[['LotArea','LotFrontage','LotConfig','LotShape']].dropna()

df2=test[['LotArea','LotFrontage','LotConfig','LotShape']].dropna()



df1['LotArea_log'] = np.log(df1['LotArea'])

df2['LotArea_log'] = np.log(df2['LotArea'])



df1['LotFrontage_log'] = np.log(df1['LotFrontage'])

df2['LotFrontage_log'] = np.log(df2['LotFrontage'])

df = pd.concat(

[df1,df2]

,axis=0,ignore_index=True )



df.tail()


regr = linear_model.LinearRegression()

X = df['LotArea_log'].values.reshape(-1,1)

Y = df['LotFrontage_log'].values.reshape(-1,1)

# print(X.reshape(-1,1).shape)

# print(Y)

regr.fit(X,Y)

print('Coefficients:',regr.coef_)

print('R^2:',regr.score(X,Y))



sns.lmplot(data=df,  x='LotFrontage_log', y='LotArea_log',

           truncate=True,fit_reg=True)



def fill_with_lm(regr,df,missing_feature, expl_vars,if_log_log=True):

    """

    missing_feature: str, the name of the missing feature to fill with linear model

    expl_vars: list of str, the 

    """

    #in training set

    missing_rows = np.where(df[missing_feature].isnull())[0]

    x = df.loc[missing_rows,expl_vars]

    if if_log_log:

        x = np.log(x).values



    #assign the values inplace

    try:

        y = regr.predict(x.reshape(-1,1))

        if if_log_log:

            y=np.exp(y)

        df.loc[missing_rows,missing_feature] = y

        print('Missing data in',missing_feature,'is fileed with linear model.')

    except:

        print('No filling is done.')



#fill the data with linear model

fill_with_lm(regr,train,'LotFrontage',['LotArea'])

fill_with_lm(regr,test,'LotFrontage',['LotArea'])





#fill_with_lm(regr,train,'LotFrontage',['LotArea'])



sns.distplot(train['GarageArea'])

print('training set has no missing GarageArea'

      if np.array_equal(np.where(train['GarageYrBlt'].isnull()),  np.where(train['GarageArea']==0))

     else 'training set has missing GarageArea')

print('testing set has no missing GarageArea'

      if np.array_equal(np.where(test['GarageYrBlt'].isnull()),  np.where(test['GarageArea']==0))

          else'training set has missing GarageArea')

print('In testing set:')

yr_is_missing=np.where(test['GarageYrBlt'].isnull())

area_is_zero = np.where(test['GarageArea']==0)

area_is_not_zero = np.where(test['GarageArea']!=0)



print(len(yr_is_missing[0]),'rows have GarageYrBlt is NaN.')

print(len(area_is_zero[0]),'rows have GarageArea ==0.')

#the index in testing data where they do have a garage but GarageYrBlt is unknown

yr_unknown_index=[i for i  in yr_is_missing[0] if i in area_is_not_zero[0]]

Yr_avg = train['GarageYrBlt'].mean()

test.loc[yr_unknown_index, 'GarageYrBlt'] = Yr_avg



#train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)

# test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
print(train['MasVnrArea'].describe())

plt.subplot(1,2,1)

sns.distplot(train['MasVnrArea'].dropna(),kde=False)

print(test['MasVnrArea'].describe())

plt.subplot(1,2,2)

sns.distplot(test['MasVnrArea'].dropna(),kde=False)
_ = barplot_missing(train)

plt.figure()

_ = barplot_missing(test)
missing_quant = [c for c in quantitative if test[c].isnull().any()]

handled = ['LotFrontage','GarageYrBlt','MasVnrArea']

#features in testing set that are still missing

unhandled = [f for f in missing_quant if f not in handled]

unhandled_mean = train[unhandled].mean()

unhandled_mean

for f in unhandled:

     test[f] = test[f].fillna(unhandled_mean[f])

        

test[unhandled].isnull().sum()

#we assign the average value to each of them 


#Choose quantitative and encoded qualitative data for training and testing

features = quantitative + qual_encoded

train_data = train[features]

test_data = test[features]





_=barplot_missing(test_data)

plt.figure()

_=barplot_missing(train_data)

#fill zeros to all 

train_data= train_data.fillna(0)

test_data= test_data.fillna(0)
#Normalize the data

train_stats = train_data.describe()

train_stats = train_stats.transpose()







def norm(x):

  return (x - train_stats.loc[:,'mean']) / train_stats.loc[:,'std']



normed_train_data = norm(train_data)

normed_test_data = norm(test_data)    
#check the normed data before training the model and giving predicition

print(normed_test_data.isnull().any().any(),

normed_train_data.isnull().any().any())
train_dataset = normed_train_data

train_labels = train['SalePrice'] 

def build_model():

  model = keras.Sequential([

    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)



  model.compile(loss='mean_squared_logarithmic_error',

                optimizer=optimizer,

                metrics=['mean_squared_logarithmic_error'])

  return model



#buili the model

model = build_model()







# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')



EPOCHS = 1500;



#set earlystop when val_loss is not improving, patienece is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)



history = model.fit(

  normed_train_data.values, train_labels.values,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

  callbacks=[

              early_stop,

              PrintDot()])



hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch

  



  

  plt.figure(figsize=(12,8))

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$MPG^2$]')

  plt.plot(hist['epoch'], hist['mean_squared_logarithmic_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_logarithmic_error'],

           label = 'Val Error')

  plt.ylim([0,.1])

  plt.legend()





plot_history(history)



plt.hlines(.028, 0, 1500,'b')

plt.show()

hist.tail()
train_predictions = model.predict(normed_train_data).flatten()





plt.scatter(train_labels, train_predictions)

plt.xlabel('True Values [SalePrice]')

plt.ylabel('Predictions [SalePrice]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

plt.plot([0, 10**6], [0, 10**6])

plt.plot(train['SalePrice'].mean(),train['SalePrice'].mean(),'ko')
train_predictions = model.predict(normed_train_data).flatten()

print('the error on the training data is',error(train_predictions,train['SalePrice']))
test_predictions = model.predict(normed_test_data).flatten()

sns.distplot(test_predictions)

sns.distplot(train['SalePrice'])
print(np.isnan(test_predictions).any())

len(test_predictions)
submission = pd.read_csv("../input/sample_submission.csv")

submission['SalePrice'] = test_predictions

submission.isnull().any().any()

submission.to_csv("submission_1_030419.csv",index=False)