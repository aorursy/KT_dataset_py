import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree
metadata = {

    'int': ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'delinq_2yrs',

       'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',

       'policy_code', 'acc_now_delinq', 'default_ind'],

    'float': ['funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti',

       'mths_since_last_delinq', 'mths_since_last_record', 'revol_util',

       'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',

       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',

       'collection_recovery_fee', 'last_pymnt_amnt',

       'collections_12_mths_ex_med', 'mths_since_last_major_derog',

       'annual_inc_joint', 'dti_joint', 'tot_coll_amt', 'tot_cur_bal',

       'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m',

       'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',

       'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',

       'total_cu_tl', 'inq_last_12m'],

    'object': ['term', 'grade', 'sub_grade', 'emp_title', 'emp_length',

       'home_ownership', 'verification_status', 'issue_d', 'pymnt_plan',

       'desc', 'purpose', 'title', 'zip_code', 'addr_state',

       'earliest_cr_line', 'initial_list_status', 'last_pymnt_d',

       'next_pymnt_d', 'last_credit_pull_d', 'application_type',

       'verification_status_joint']

}



target = 'default_ind'
cred = pd.read_csv(

    '../input/Data File.csv',

    dtype = {col:t for t in metadata for col in metadata[t]},

    encoding='latin')

print('Dataset Dimensions:',cred.shape)
nrows = len(cred)

#cred.info()
cred.head()
for col in [col for col in metadata['object'] if col[-2:]=='_d']:

    cred[col] = pd.to_datetime(cred[col],format='%b-%y')
col_levels = dict()
for col in metadata['object']:

    sm = cred[col].describe()

    col_levels[col] = sm['unique']

    print(sm)

    print(col,' Missing Rate = {:.2%}'.format(1-cred[col].count()/nrows))

    if col_levels[col] < 8:

        cred[col].value_counts().plot.bar(title=col)

    else:

        cred[col].value_counts().plot(logy=True, title=col)

    plt.show()
for col in metadata['int']:

    vc = cred[col].value_counts()

    col_levels[col] = len(vc)

    print(col,' Missing Rate = {:.2%}'.format(1-cred[col].count()/nrows))

    if col_levels[col] < 8:

        cred[col].value_counts().plot.bar(title=col)

    else:

        ax = cred[col].plot.hist(title=col)

        ax.set_yscale('log')

    plt.show()
for col in metadata['float']:

    vc = cred[col].value_counts()

    col_levels[col] = len(vc)

    print(col,' Missing Rate = {:.2%}'.format(1-cred[col].count()/nrows))

    if col_levels[col] < 8:

        cred[col].value_counts().plot.bar(title=col)

    else:

        ax = cred[col].plot.hist(title=col)

        ax.set_yscale('log')

    plt.show()
tmp = cred.iloc[np.random.choice(len(cred),10000)].copy()

for col in tmp.columns:

    if tmp[col].isna().sum() < 1:

        del tmp[col]

        continue

    tmp[col] = np.where(tmp[col].isna(),1,0)

tmp.head()
corr = tmp.corr()

ax = sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

ax.get_figure().set_size_inches(15,10)

ax.set_title('Missingness Correlation Matrix');
clf = tree.DecisionTreeClassifier(max_leaf_nodes=7,min_samples_leaf=10)

np.random.seed(19)

smp = np.random.choice(len(cred),1000)

y = cred.iloc[smp][target]
alphas = np.linspace(0.1, 1, 1001)

rgba_colors = np.zeros((1001,4))

# for red the first column needs to be one

rgba_colors[:,2] = 1.0

# the fourth column needs to be your alphas

rgba_colors[:, 3] = alphas
for col in metadata['object']:

    if len(cred.iloc[smp][col].value_counts()) <= 1:

        continue

    x_ = cred.iloc[smp][col].fillna('Missing')

    x = pd.get_dummies(x_)

    clf = clf.fit(x, y)

    if col_levels[col] < 8:

        y.groupby(x_).mean().plot.bar(title=col+' versus default')

    else:

        x_tree = clf.apply(x)

        ax = y.groupby(x_tree).mean().plot.bar(title=col+' versus default')

        plt.show()

        print(col.upper(),' Groupings / Most Common Within Group / Count In Sample of 1,000')

        print(x_.groupby(x_tree).apply(lambda x: x.value_counts().head()))

    plt.show()
for col in metadata['int']:

    if col == target:

        continue

    if len(cred.iloc[smp][col].value_counts()) <= 1:

        continue

    x_ = cred.iloc[smp][col].fillna(-99)

    x = x_.values.reshape(-1,1)

    clf = clf.fit(x, y)

    if col_levels[col] < 8:

        y.groupby(x_).mean().plot.bar(title=col+' versus default')

        plt.show()

    else:

        x_tree = clf.apply(x)

        tbl = x_.groupby(x_tree).apply(lambda x: pd.Series([x.min(),x.max(),x.count()],index=['min','max','N'])).unstack()

        tbl['PD'] = y.groupby(x_tree).mean()

        tbl = tbl.sort_values('min')

        plt.bar(x=tbl['min'],height=tbl.PD,width=(1+tbl['max']-tbl['min']),align='edge',

                edgecolor='k',color=rgba_colors[tbl.N])

        plt.title(col)

        plt.show()

        print(col.upper(),' Groupings / Range Min / Range Max / Count in Sample of 1,000')

        print(tbl)
for col in metadata['float']:

    if len(cred.iloc[smp][col].value_counts()) <= 1:

        continue

    x_ = cred.iloc[smp][col].fillna(-99)

    x = x_.values.reshape(-1,1)

    clf = clf.fit(x, y)

    if col_levels[col] < 8:

        y.groupby(x_).mean().plot.bar(title=col+' versus default')

        plt.show()

    else:

        x_tree = clf.apply(x)

        tbl = x_.groupby(x_tree).apply(lambda x: pd.Series([x.min(),x.max(),x.count()],index=['min','max','N'])).unstack()

        tbl['PD'] = y.groupby(x_tree).mean()

        tbl = tbl.sort_values('min')

        plt.bar(x=tbl['min'],height=tbl.PD,width=(tbl['max']-tbl['min']),align='edge',

                edgecolor='k',color=rgba_colors[tbl.N.astype('int')])

        plt.title(col)

        plt.show()

        print(col.upper(),' Groupings / Range Min / Range Max / Count in Sample of 1,000')

        print(tbl)