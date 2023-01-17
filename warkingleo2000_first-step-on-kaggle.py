import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

import scipy

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_curve

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from tqdm import tqdm_notebook
raw_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

raw_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')



print(raw_train.shape, raw_test.shape)
raw_test.head()
raw_train.head()
def plot_missing_values(df):



    cols = df.columns

    count = [df[col].isnull().sum() for col in cols]

    percent = [i/len(df) for i in count]

    missing = pd.DataFrame({'number':count, 'proportion': percent}, index=cols)

    

    fig, ax = plt.subplots(1,2, figsize=(20,7))

    for i, col in enumerate(missing.columns):



        plt.subplot(1,2,i+1)

        plt.title(f'Missing values on each columns({col})')

        sns.barplot(missing[col], missing.index)

        mean = np.mean(missing[col])

        std = np.std(missing[col])

        plt.ylabel('Columns')

        plt.plot([], [], ' ', label=f'Average {col} of missing values: {mean:.2f} \u00B1 {std:.2f}')

        plt.legend()

    plt.show()

    return missing.sort_values(by='number', ascending=False)
missing_train = plot_missing_values(raw_train)

missing_train.head()
missing_test = plot_missing_values(raw_test)

missing_test.head()
plt.figure(figsize=(6,6))

ax = sns.countplot(raw_train.target)



height = sum([p.get_height() for p in ax.patches])

for p in ax.patches:

        ax.annotate(f'{100*p.get_height()/height:.2f} %', (p.get_x()+0.3, p.get_height()+5000),animated=True)
plt.figure(figsize=(10,7))

num_cols = raw_train.select_dtypes(exclude=['object']).columns

corr = raw_train[num_cols].corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
plt.figure(figsize=(10,7))



corr_0 = raw_train[num_cols][raw_train.target==0].corr()

sns.heatmap(corr_0, 

            xticklabels=corr_0.columns.values,

            yticklabels=corr_0.columns.values)
plt.figure(figsize=(10,7))



corr_1 = raw_train[num_cols][raw_train.target==1].corr()

sns.heatmap(corr_1, 

            xticklabels=corr_1.columns.values,

            yticklabels=corr_1.columns.values)
num_cols = raw_test.select_dtypes(exclude=['object']).columns

fig, ax = plt.subplots(2,3,figsize=(22,7))

for i, col in enumerate(num_cols):

    plt.subplot(2,3,i+1)

    plt.xlabel(col, fontsize=9)

    sns.kdeplot(raw_train[col].values, bw=0.5,label='Train')

    sns.kdeplot(raw_test[col].values, bw=0.5,label='Test')

   

plt.show() 
target0 = raw_train.loc[raw_train['target'] == 0]

target1 = raw_train.loc[raw_train['target'] == 1]



fig, ax = plt.subplots(2,3,figsize=(22,7))

for i, col in enumerate(num_cols):

    plt.subplot(2,3,i+1)

    plt.xlabel(col, fontsize=9)

    sns.kdeplot(target0[col].values, bw=0.5,label='Target: 0')

    sns.kdeplot(target1[col].values, bw=0.5,label='Target: 1')

    sns.kdeplot(raw_test[col].values, bw=0.5,label='Test')

    

plt.show() 
bin_cols = [f'bin_{i}' for i in range(5)]



fig, ax = plt.subplots(1,5, figsize=(22, 5))



for i, col in enumerate(bin_cols):

     ax0 = plt.subplot(1,5,i+1)

     raw_train[col].value_counts().plot.bar(color='pink')

     height = sum([p.get_height() for p in ax0.patches])



     for p in ax0.patches:

         ax0.text(p.get_x()+p.get_width()/2., p.get_height()+4000, f'{100*p.get_height()/height:.2f} %', ha='center')

     plt.xlabel(f'{col}')

plt.suptitle('Distribution over binary feature of train data')

fig, ax = plt.subplots(1,5, figsize=(22, 5))



for i, col in enumerate(bin_cols):

     ax0 = plt.subplot(1,5,i+1)

     raw_test[col].value_counts().plot.bar(color='lime')

     height = sum([p.get_height() for p in ax0.patches])



     for p in ax0.patches:

         ax0.text(p.get_x()+p.get_width()/2., p.get_height()+4000, f'{100*p.get_height()/height:.2f} %', ha='center')

     plt.xlabel(f'{col}')

plt.suptitle('Distribution over binary feature of test data')
plt.figure(figsize=(22,6))

plt.title('Day distribution')

ax = sns.countplot(raw_train.day, hue=raw_train.target)

for p in ax.patches:

    ax.text(p.get_x()+p.get_width()/2., p.get_height()+1000, f'{100*p.get_height()/height:.2f} %',ha='center')

plt.show()
plt.figure(figsize=(22,6))

plt.title('Month distribution')

ax = sns.countplot(raw_train.month, hue=raw_train.target)

for p in ax.patches:

    ax.text(p.get_x()+p.get_width()/2., p.get_height()+1000, f'{100*p.get_height()/height:.2f} %', ha='center')

plt.show()
df_train = raw_train.dropna(subset=['month', 'day'])[['day', 'month', 'target']]

df_test = raw_test.dropna(subset=['month', 'day'])[['day', 'month']]

df0 = df_train[df_train.target == 0]

df1 = df_train[df_train.target == 1]



def number2datetime(df):

    time_col = '2019/' + df.month.astype(int).astype(str) + '/' + df.day.astype(int).astype(str)

    df['time'] = pd.to_datetime(time_col , format = '%Y/%m/%d')

    df = df.drop(columns=['day', 'month'])

    return df

df0 = number2datetime(df0)

df1 = number2datetime(df1)

df_test = number2datetime(df_test)
count0 = df0.time.value_counts()/len(df0)

count0 = count0.sort_index()

count1 = df1.time.value_counts()/len(df1)

count1 = count1.sort_index()

count_test = df_test.time.value_counts(normalize=True)
plt.figure(figsize=(20,8))

sns.lineplot(count0.index, count0.values, label='Target:0')

sns.lineplot(count1.index, count1.values, label='Target:1')

sns.lineplot(count_test.index, count_test.values, label='Test')

plt.legend(loc='upper left')
nom_cols = [f'nom_{i}' for i in range(5)]

fig, ax = plt.subplots(1,5, figsize=(22, 6))

for i, col in enumerate(nom_cols):

    plt.subplot(1,5,i+1)

    sns.countplot(f'nom_{i}', hue='target', data= raw_train)



plt.show()
plt.figure(figsize=(17, 35)) 

fig, ax = plt.subplots(2,3,figsize=(22,10))



for i, col in enumerate(raw_train[nom_cols]): 

    tmp = pd.crosstab(raw_train[col], raw_train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.subplot(2,3,i+1)

    sns.countplot(x=col, data=raw_train, order=list(tmp[col].values) , palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



    # twinX - to build a second yaxis

    gt = ax.twinx()

    gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

    gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

    gt.set_ylabel("Target %True(1)", fontsize=16)

    sizes=[] # Get highest values in y

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.subplots_adjust(hspace = 0.5, wspace=.3)

plt.show()
for col in nom_cols:

    fig, ax = plt.subplots(1,3,figsize=(22,6))

    ax[0].set_title(f'Target 0 data {col}')

    ax[1].set_title(f'Target 1 data {col}')

    ax[2].set_title(f'Test data {col}')



    explode = np.zeros(raw_train[col].nunique()+1)

    explode[1] = 0.05   

    target0_count = target0[col].value_counts(dropna=False)

    target1_count = target1[col].value_counts(dropna=False)    

    test_count = raw_test[col].value_counts(dropna=False)



    ax[0].pie(target0_count, labels=target0_count.index, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90)

    ax[0].legend(labels=target0_count.index,loc=3)

    ax[1].pie(target1_count, labels=target1_count.index, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90)

    ax[1].legend(labels=target1_count.index,loc=3)    

    ax[2].pie(test_count, labels=test_count.index, autopct='%1.1f%%', explode=explode, shadow=False, startangle=90)

    ax[2].legend(labels=test_count.index,loc=3)
nom_cols = [f'nom_{i}' for i in range(5,10)]

fig, ax = plt.subplots(5,1, figsize=(22,17))

for i,col in enumerate(nom_cols):

    plt.subplot(5,1,i+1)

    sns.countplot(raw_train[col])

plt.show()
ord_cols = [f'ord_{i}' for i in range(3)]

plt.figure(figsize=(17, 35)) 

fig, ax = plt.subplots(3,1,figsize=(15,15))



for i, col in enumerate(raw_train[ord_cols]): 

    tmp = pd.crosstab(raw_train[col], raw_train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.subplot(3,1,i+1)

    sns.countplot(x=col, data=raw_train, order=list(tmp[col].values) , palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



    gt = ax.twinx()

    gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

    gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

    gt.set_ylabel("Target %True(1)", fontsize=16)

    sizes=[] # Get highest values in y

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights



plt.subplots_adjust(hspace = 0.5, wspace=.3)

plt.show()
ord_cols = ['ord_3', 'ord_4']

plt.figure(figsize=(17, 35)) 

fig, ax = plt.subplots(2,1,figsize=(22,10))



for i, col in enumerate(raw_train[ord_cols]): 

    tmp = pd.crosstab(raw_train[col], raw_train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.subplot(2,1,i+1)

    sns.countplot(x=col, data=raw_train, order=list(tmp[col].values) , palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



    # twinX - to build a second yaxis

    gt = ax.twinx()

    gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

    gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

    gt.set_ylabel("Target %True(1)", fontsize=16)

    sizes=[] # Get highest values in y

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.subplots_adjust(hspace = 0.5, wspace=.3)

plt.show()
tmp = pd.crosstab(raw_train['ord_5'], raw_train['target'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

plt.figure(figsize=(18,7))



plt.subplot()

ax = sns.countplot(x='ord_5', data=raw_train, order=list(tmp['ord_5'].values) , color='chocolate') 

ax.set_ylabel('Count', fontsize=17) # y axis label

ax.set_title('ord_5 Distribution', fontsize=20) # title label

ax.set_xlabel('ord_5 values', fontsize=17) # x axis label
full_data = pd.concat([raw_train, raw_test], sort=False).drop(columns='target')

full_data.shape
# Replace values which doesnt appear in both train and test set with another special value ('xor')

cate_columns = full_data.select_dtypes(include=['object']).columns

for col in cate_columns:

    train_values = set(raw_train[col].unique())

    test_values = set(raw_test[col].unique())



    xor_values = test_values - train_values 

    if xor_values:

        print(f'Replace {len(xor_values)} in {col} column')

        print('They are: ', xor_values)

        print()

        full_data.loc[full_data[col].isin(xor_values), col] = 'xor'
map_ord1 = {'Novice':1, 

            'Contributor':2, 

            'Expert':4, 

            'Master':5, 

            'Grandmaster':6}

full_data.ord_1 = full_data.ord_1.map(map_ord1)
map_ord2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

full_data.ord_2 = full_data.ord_2.map(map_ord2)
# Replace a character with its ASCII value

full_data['ord_3_by_ord'] = full_data.ord_3.map(ord, na_action='ignore')

map_ord3 = {key:value for value,key in enumerate(sorted(full_data.ord_3.dropna().unique()))}

full_data.ord_3 = full_data.ord_3.map(map_ord3)
full_data['ord_4_by_ord'] = full_data.ord_4.map(ord, na_action='ignore')

map_ord4 = {key:value for value,key in enumerate(sorted(full_data.ord_4.dropna().unique()))}

full_data.ord_4 = full_data.ord_4.map(map_ord4)
# ord_5 is a little bit more special(2-characters-string)

# We divide it into 2 pieces of character but also keep the orgin string and convert to categorical features by Label Encoder



full_data['ord_5_1'] = full_data.ord_5.map(lambda string: ord(string[0]), na_action='ignore')

full_data['ord_5_2'] = full_data.ord_5.map(lambda string: ord(string[1]), na_action='ignore')



map_ord5 = {key:value for value,key in enumerate(sorted(full_data.ord_5.dropna().unique()))} 

full_data.ord_5 = full_data.ord_5.map(map_ord5)
num_columns = full_data.select_dtypes(exclude=['object']).columns.drop(['bin_0', 'bin_1', 'bin_2'])

cate_columns = full_data.columns.drop(num_columns)
missing_num_columns = [col for col in num_columns if any(full_data[col].isnull())]

for col in missing_num_columns:

    full_data[col+'_is_missing'] = full_data[col].isnull().astype(int)
time_cols = ['day', 'month']



for col in time_cols:

    full_data[col+'_sin'] = np.sin(2*np.pi*full_data[col]/7)

    full_data[col+'_cos'] = np.cos(2*np.pi*full_data[col]/12)

full_data = full_data.drop(columns=time_cols)
retain_cols = [f'ord_{i}' for i in range(6)] + ['day_sin', 'day_cos', 'month_sin', 'month_cos']

OH_cols = full_data.columns#.drop(retain_cols)
print(f"One-Hot encoding {len(OH_cols)} columns")



OH_full = pd.get_dummies(

    full_data,

    columns=OH_cols,

    drop_first=True,

    dummy_na=True,

    sparse=True,

).sparse.to_coo()
# Impute numeric features with mean value and normalize afterward 

imputer = SimpleImputer(strategy='mean')

retain_full  = pd.DataFrame(imputer.fit_transform(full_data[retain_cols]), columns=retain_cols)

retain_full = retain_full/retain_full.max()
encoded_full = scipy.sparse.hstack([OH_full, retain_full, retain_full**2]).tocsr()

print(encoded_full.shape)



encoded_train = encoded_full[:len(raw_train)]

encoded_test = encoded_full[len(raw_train):]
model = LogisticRegression(C=0.03, max_iter=300)
fig, ax = plt.subplots(figsize=(8,5))

aucs = []

cv = StratifiedKFold(n_splits=5, shuffle=True)



for i, (train,valid) in tqdm_notebook(enumerate(cv.split(encoded_train, raw_train.target))):

    

    model.fit(encoded_train[train], raw_train.target[train])

    valid_pred = model.predict_proba(encoded_train[valid])[:, 1]

    

    fpr, tpr, threshold = roc_curve(raw_train.target[valid], valid_pred)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = f'Folf number {i+1} (AUC = {roc_auc:.4f})')

    aucs.append(roc_auc)



ax.plot([0,1], [0,1], label='Luck', linestyle='--', color='r')  

mean_auc = np.mean(aucs)

std_auc = np.std(aucs)

ax.plot(mean_auc, label=f'Average AUC score: {mean_auc:.4f} $\pm$ {std_auc:.4f}')

ax.legend(loc="lower right")

ax.set(xlim=[-.1, 1.1], ylim=[-.1, 1.1], title='Logistic Regression')

plt.show()
"""

model = LogisticRegression()

param_grid = {'C' : np.logspace(-4, 4, 20), 'penalty' : ['l1', 'l2']}



# Create grid search object



clf = GridSearchCV(LogisticRegression(), scoring='roc_auc', param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# Fit on data



clf.fit(encoded_train, raw_train.target)



print("tuned hpyerparameters :(best parameters) ",clf.best_params_)

print("Accuracy :",clf.best_score_)"""
"""

model = LogisticRegression(**clf.best_params_)

fig, ax = plt.subplots(figsize=(8,5))

aucs = []

cv = StratifiedKFold(n_splits=5, shuffle=True)



for i, (train,valid) in tqdm_notebook(enumerate(cv.split(encoded_train, raw_train.target))):

    

    model.fit(encoded_train[train], raw_train.target[train])

    valid_pred = model.predict_proba(encoded_train[valid])[:, 1]

    

    fpr, tpr, threshold = roc_curve(raw_train.target[valid], valid_pred)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = f'Folf number {i} (AUC = {roc_auc:.4f})')

    aucs.append(roc_auc)



ax.plot([0,1], [0,1], label='Luck', linestyle='--', color='r')  

mean_auc = np.mean(aucs)

std_auc = np.std(aucs)

ax.plot(mean_auc, label=f'Average AUC score: {mean_auc:.4f} $\pm$ {std_auc:.4f}')

ax.legend(loc="lower right")

ax.set(xlim=[-.1, 1.1], ylim=[-.1, 1.1], title='Logistic Regression')

plt.show()"""
%%time

model = LogisticRegression(C=0.03, max_iter=300)

model.fit(encoded_train, raw_train.target)

test_pred = model.predict_proba(encoded_test)[:, 1]
%%time

submiss = pd.DataFrame({"id": raw_test.index, "target": test_pred})

submiss.to_csv('Phan_Viet_Hoang.csv', index=False)