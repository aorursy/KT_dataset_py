import pandas as pd

import numpy as np

import gc

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Plots look better and clearer in svg format

%config InlineBackend.figure_format = 'svg' 



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



# Some aesthetic settings

plt.style.use('bmh')

sns.set(style = 'whitegrid', font_scale = 0.8, rc={"grid.linewidth": 0.5, "lines.linewidth": 1})
# Loading datasets

id_train = pd.read_csv('../input/train_identity.csv', index_col = 'TransactionID')

tr_train = pd.read_csv('../input/train_transaction.csv', index_col = 'TransactionID')



id_test = pd.read_csv('../input/test_identity.csv', index_col = 'TransactionID')

tr_test = pd.read_csv('../input/test_transaction.csv', index_col = 'TransactionID')
# Join train and test datasets

train_df = tr_train.join(id_train)

test_df = tr_test.join(id_test)



# Removing datasets that we don't need anymore

del id_train

del tr_train

del id_test

del tr_test



gc.collect()
# Shape of datasets

print('Train dataset shape: ',  train_df.shape)

print('Test dataset shape: ',  test_df.shape)
train_df.head()



#     TransactionDT: timedelta from a given reference datetime (not an actual timestamp)

#     TransactionAMT: transaction payment amount in USD

#     ProductCD: product code, the product for each transaction

#     card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

#     addr: address

#     dist: distance

#     P_ and (R__) emaildomain: purchaser and recipient email domain

#     C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.

#     D1-D15: timedelta, such as days between previous transaction, etc.

#     M1-M9: match, such as names on card and address, etc.

#     Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
# Function block

def null_table(dataset):

    

    '''Create table with ammount of null values for dataset'''

    

    return pd.DataFrame({'Null values': dataset.isnull().sum(), 

                         '% of nulls': round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2)}).T



def plot_hist(col, hist=True, kde = True, bins = 50, log = False, fillna = np.nan):

    

    '''Creates distribution plot for train/test datasets and fraud

        col - column name

        hist - Whether to plot a histogram (True/False)

        kde - Whether to plot a gaussian kernel density estimate (True/False)

        bins - number of bins

        log - apply np.log1p transform to values (True/False)

        fillna - fill nulls with specified value'''

    

    if log:

        sns.distplot(np.log1p(train_df[col].fillna(fillna)), label = 'Train', hist = hist, bins = bins, kde = kde)

        sns.distplot(np.log1p(test_df[col].fillna(fillna)), label = 'Test', hist = hist, bins = bins, kde = kde)

        sns.distplot(np.log1p(train_df[train_df['isFraud'] == 1][col].fillna(fillna)), label = 'Fraud', hist = hist, bins = bins, kde = kde)

        plt.title(f'Distribution of {col} feature with log transform')

        

    else:

        sns.distplot(train_df[col].fillna(fillna), label = 'Train', hist = hist, bins = bins, kde = kde)

        sns.distplot(test_df[col].fillna(fillna), label = 'Test', hist = hist, bins = bins, kde = kde)

        sns.distplot(train_df[train_df['isFraud'] == 1][col].fillna(fillna), label = 'Fraud', hist = hist, bins = bins, kde = kde, color = 'black')

        plt.title(f'Distribution of {col} feature')

        

    plt.legend()



def plot_cat(col, test = True, rot = 0, n = False, fillna = np.nan, annot = False, title = True):

    

    '''Creates countplot for train/test datasets and fraud

       col - column name

       test - makes 2 plots if True - countplot for train and test values, second - countplot fraud and non fraud in train dataset

       rot - rotation of xticks

       n - plot only n top values

       fillna - fill nulls with specified values

       annot - whether to plot % of transactions

       '''

    

    if n:

        order = train_df[col].fillna(fillna).value_counts().sort_values(ascending = False).iloc[:n].index

    else:

        order = train_df[col].fillna(fillna).value_counts().sort_values(ascending = False).index

    

    if test:

        t = pd.DataFrame({col: train_df[col], 'Label': 'Train'})

        tst = pd.DataFrame({col: test_df[col], 'Label': 'Test'})

        t_tr = pd.concat([t, tst])        

               

        plt.subplot('211')        

        sns.countplot(t_tr[col].fillna(fillna), hue = t_tr['Label'], 

                      order = order).set_title('Train/Test countplot')

        plt.xticks(rotation = rot)

        

        plt.subplot('212')        

        g = sns.countplot(train_df[col].fillna(fillna), hue = train_df['isFraud'],

                     order = order)

        if annot:

            for p in g.patches:

                g.annotate('{:.1f}%'.format((p.get_height() / train_df.shape[0]) * 100, 2), 

                           (p.get_x() + 0.1, p.get_height()+5000))

        plt.title('Fraud/non fraud countplot')

        plt.xticks(rotation = rot)

        plt.tight_layout()

        

    else:

        g = sns.countplot(train_df[col].fillna(fillna), hue = train_df['isFraud'], 

                      order = order)

        if annot:

            for p in g.patches:

                g.annotate('{:.1f}%'.format((p.get_height() / train_df.shape[0]) * 100, 2), 

                           (p.get_x() + 0.1, p.get_height()+5000))

        if title:

            plt.title('Fraud/non fraud countplot')

        plt.xticks(rotation = rot)

        

def plot_rate(col, figsize = (8, 4), head = False, n = 10, df = False):

    

    ''' Plot fraud rate for categorical features'''

    

    if df:

        if head:

            return pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                                  train_df[col].value_counts()).sort_values(by = col, ascending = False).head(n)

        else:

            return pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                                  train_df[col].value_counts()).sort_values(by = col, ascending = False)

    else:

        

        if head:

            fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                                      train_df[col].value_counts()).sort_values(by = col, ascending = False).head(n)

        else:

            fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() / 

                                      train_df[col].value_counts()).sort_values(by = col, ascending = False)



        fraud_rate.plot(kind = 'bar', figsize = figsize)

        

def plot_grid(cols_list, rows, cols, start = 0, end = -1, figsize = (11, 10), rot = 45):

    fig = plt.figure(figsize = figsize)

    for i, col in enumerate(cols_list[start:end]):

        plt.subplot(rows, cols, i+1)

        if train_df[col].dtype == 'O':

            plot_cat(col, test = False, n = 15, rot = rot)



        else:

            plot_hist(col, hist = False)

    plt.tight_layout()
train_null = null_table(train_df)

test_null = null_table(test_df)



train_null
test_null
print('Number of train_df features with more than 50% null values: ', train_null.T[train_null.T['% of nulls'] > 50].shape[0])

print('Number of test_df features with more than 50% null values: ', test_null.T[test_null.T['% of nulls'] > 50].shape[0])
# isFraud feature

fig = plt.figure(figsize = (5, 4))

g = sns.countplot('isFraud', data = train_df)

for p in g.patches:

    g.annotate('{} ({:.2f})%'.format(p.get_height(), (p.get_height() / train_df.shape[0]) * 100, 2), (p.get_x() + 0.1, p.get_height()+5000))
# TransactionDT: timedelta from a given reference datetime (not an actual timestamp)

fig = plt.figure(figsize = (13, 6))

plot_hist('TransactionDT')
train_df['Tr_day'] = np.floor((train_df['TransactionDT'] / (3600 * 24) - 1) % 7)

train_df['Tr_hour'] = np.floor(train_df['TransactionDT'] / 3600) % 24
# Day of transaction plots

fig, ax = plt.subplots(1, 2, figsize = (13, 4))

sns.barplot(x = 'Tr_day', y = 'TransactionAmt', data = train_df, hue = 'isFraud', ax = ax[0])

sns.countplot(x = 'Tr_day', data = train_df, hue = 'isFraud', ax = ax[1])
# Hour of transaction plots

fig, ax = plt.subplots(2, 1, figsize = (13, 6))

sns.barplot(x = 'Tr_hour', y = 'TransactionAmt', data = train_df, hue = 'isFraud', ax = ax[0])

sns.countplot(x = 'Tr_hour', data = train_df, hue = 'isFraud', ax = ax[1])
#     TransactionAMT: transaction payment amount in USD

for i, j in enumerate([False, True]):

    fig = plt.figure(figsize = (13, 7))

    plt.subplot(f'21{i+1}')

    plot_hist('TransactionAmt', hist = False, log = j)
train_df['TransactionAmt'].describe()
# ProductCD: product code, the product for each transaction

fig, ax = plt.subplots(2, 1, figsize = (11, 7))

plot_cat('ProductCD')
plot_rate('ProductCD')
# card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

cols = ['card{}'.format(i) for i in range(1, 7)]

fig = plt.figure(figsize = (11, 7))

for i, col in enumerate(cols):

    plt.subplot('23{}'.format(i + 1))

    

    if col not in ['card4', 'card6']:    

        plot_hist(col, hist = False)

        

    else:

        plot_cat(col, test = False, rot = 90, annot = False)

        

plt.tight_layout()
plot_rate('card4')
#   addr: address

#   dist: distance

fig = plt.figure(figsize = (11, 7))

for i, col in enumerate(['addr1', 'addr2', 'dist1', 'dist2']):    

    plt.subplot(f'22{i+1}')

    plot_hist(col, hist = False)

plt.tight_layout()
fig = plt.figure(figsize = (11, 7))

for i, col in enumerate(['addr1', 'addr2', 'dist1', 'dist2']):    

    plt.subplot(f'22{i+1}')

    plot_cat(col, test = False, annot = False, n = 10, title = False)

plt.tight_layout()
fig, ax = plt.subplots(1, 2, figsize = (13, 4))

for i, col in enumerate(['addr1', 'addr2']):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(20)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

    
train_df[train_df['addr1'].isin(plot_rate('addr1', df = True, head = True, n = 6).index.values)]['addr1'].value_counts()
train_df[train_df['addr2'].isin(plot_rate('addr2', df = True, head = True, n = 6).index.values)]['addr2'].value_counts()
fig, ax = plt.subplots(2, 1, figsize = (11, 7))

for i, col in enumerate(['dist1', 'dist2']):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(40)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
train_df[train_df['dist1'].isin(plot_rate('dist1', df = True, head = True, n = 16).index.values)]['dist1'].value_counts()
train_df[train_df['dist2'].isin(plot_rate('dist2', df = True, head = True, n = 28).index.values)]['dist2'].value_counts()
#     P_ and (R__) emaildomain: purchaser and recipient email domain

# P_emaildomain feature

fig = plt.figure(figsize = (11, 8))

plot_cat('P_emaildomain', rot = 90, fillna = 'Null', n = 20, annot = False)
# R_emaildomain feature

fig = plt.figure(figsize = (11, 8))

plot_cat('R_emaildomain', rot = 90, n = 20, annot = False)
fig, ax = plt.subplots(2, 1, figsize = (11, 7))

for i, col in enumerate(['P_emaildomain', 'R_emaildomain']):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(40)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
print(train_df[train_df['P_emaildomain'] == 'protonmail.com']['P_emaildomain'].value_counts())

print(train_df[train_df['R_emaildomain'] == 'protonmail.com']['R_emaildomain'].value_counts())
#     C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.

c_cols = [f'C{i}' for i in range(1, 15)]

train_df[c_cols].describe()
# C1-C6 features

cols = [f'C{i}' for i in range(1, 7)]

fig = plt.figure(figsize = (11, 18))

for i, col in enumerate(cols):    

    plt.subplot(f'72{i+1}')

    plot_hist(col, hist = False)

plt.tight_layout()
fig = plt.figure(figsize = (11, 7))

for i, col in enumerate(cols):    

    plt.subplot(f'32{i+1}')

    plot_cat(col, test = False, annot = False, n = 10, title = False)

plt.tight_layout()
fig, ax = plt.subplots(6, 1, figsize = (11, 10))

for i, col in enumerate(cols):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(60)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
# C7-C14 features

cols = [f'C{i}' for i in range(7, 15)]

fig = plt.figure(figsize = (11, 20))

for i, col in enumerate(cols):    

    plt.subplot(f'72{i+1}')

    plot_hist(col, hist = False)

plt.tight_layout()
fig = plt.figure(figsize = (11, 8))

for i, col in enumerate(cols):    

    plt.subplot(f'24{i+1}')

    plot_cat(col, test = False, annot = False, n = 10, title = False, rot = 90)

plt.tight_layout()
fig, ax = plt.subplots(8, 1, figsize = (11, 12))

for i, col in enumerate(cols):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(60)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
#     D1-D15: timedelta, such as days between previous transaction, etc.

d_cols = [f'D{i}' for i in range(1, 16)]

train_df[d_cols].describe()
cols = [f'D{i}' for i in range(1, 9)]

fig = plt.figure(figsize = (11, 20))

for i, col in enumerate(cols):    

    plt.subplot(f'72{i+1}')

    plot_hist(col, hist = False)

plt.tight_layout()
fig = plt.figure(figsize = (11, 8))

for i, col in enumerate(cols):    

    plt.subplot(f'24{i+1}')

    plot_cat(col, test = False, annot = False, n = 10, title = False, rot = 90)

plt.tight_layout()
fig, ax = plt.subplots(8, 1, figsize = (11, 12))

for i, col in enumerate(cols):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(60)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
cols = [f'D{i}' for i in range(9, 16)]

fig = plt.figure(figsize = (11, 20))

for i, col in enumerate(cols):    

    plt.subplot(f'72{i+1}')

    plot_hist(col, hist = False)

plt.tight_layout()
fig = plt.figure(figsize = (11, 8))

for i, col in enumerate(cols):    

    plt.subplot(f'24{i+1}')

    plot_cat(col, test = False, annot = False, n = 10, title = False, rot = 90)

plt.tight_layout()
fig, ax = plt.subplots(7, 1, figsize = (11, 20))

for i, col in enumerate(cols):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False).head(60)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
#     M1-M9: match, such as names on card and address, etc.

m_cols = [f'M{i}' for i in range(1, 10)]

train_df[m_cols].describe()
fig = plt.figure(figsize = (11, 10))

for i, col in enumerate(m_cols):    

    plt.subplot(f'33{i+1}')

    plot_cat(col, test = False, fillna = 'Null')

plt.tight_layout()
fig, ax = plt.subplots(1, 9, figsize = (15, 3))

for i, col in enumerate(m_cols):

    fraud_rate = pd.DataFrame(train_df[train_df['isFraud'] == 1][col].value_counts() /

                              train_df[col].value_counts()).sort_values(by = col, ascending = False)

    fraud_rate.plot(kind = 'bar', ax = ax[i])

plt.tight_layout()
# DeviceType

fig = plt.figure(figsize = (8, 8))

plot_cat('DeviceType', rot = 90, fillna = 'null')
plot_rate('DeviceType')
# DeviceInfo

fig = plt.figure(figsize = (12, 4))

plot_cat('DeviceInfo', rot = 90, test = False, n = 20, annot = False)
plot_rate('DeviceInfo', head = True, figsize = (13, 4), n = 60)
train_df[train_df['DeviceInfo'].isin(plot_rate('DeviceInfo', df = True, head = True, n = 46).index.values)]['DeviceInfo'].value_counts()
# id features

id_cols = [f'id_0{i}' for i in range(1, 10)] + [f'id_{i}' for i in range(10, 39)]

train_df[id_cols].describe(include = 'all')
plot_grid(id_cols, 3, 3, 0, 9)
plot_grid(id_cols, 3, 3, 9, 18)
plot_grid(id_cols, 3, 3, 18, 27)
plot_grid(id_cols, 3, 3, 27, 36)
plot_grid(id_cols, 1, 2, 36, 39, figsize = (8, 3))