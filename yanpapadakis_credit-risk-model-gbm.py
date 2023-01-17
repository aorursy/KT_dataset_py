import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split



import gc
# Prohibited Variables (redlining regulations)

prohibited = ['zip_code','addr_state']



# Many Errors / Not Codified

many_errors = ['desc','emp_title','title']



# Suspicious Data

# next_pymnt_d: missing correlates suspiciously to default rate

# issue_d and last_credit_pull_d are surpisingly and counterintuitively predictive

# id: significant, maybe is proxy of tenure or time of Origination / drop however

# member_id: same as id

suspect = ['next_pymnt_d', 'issue_d', 'last_credit_pull_d', 'id', 'member_id']



unknown_at_origination = ['funded_amnt_inv', 'funded_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',

    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',

    'total_rec_prncp', 'total_rec_int', #'total_rec_late_fee', 'recoveries',

    'collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d']



zero_variance = ['policy_code','pymnt_plan'] # zero or near-zero variance



quasi_separation = [

    'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',

    'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'inq_last_12m', 

    'open_acc_6m', 'total_cu_tl'

]



skipvars = prohibited + many_errors + suspect + unknown_at_origination + zero_variance + quasi_separation
dates = ['issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'] #,'earliest_cr_line']

    # earliest_cr_line is not parsed correctly automatically

    

cred = pd.read_csv(

    '../input/Data File.csv',

    usecols = lambda x: x not in skipvars,

    parse_dates = list(set(dates) - set(skipvars)),

    date_parser = lambda x: pd.to_datetime(x,format='%b-%y'),

    low_memory=False

)

print('Dataset Dimensions:',cred.shape)
# Remove Already Delinquent



cred = cred[(cred.total_rec_late_fee == 0) & (cred.recoveries == 0)]
with pd.option_context('display.max_rows', 10, 'display.max_columns', None): 

    display(cred)
y = cred['default_ind'].values

print("Mean Bad Rate = {:.2%}".format(np.mean(y)))
def read_earliest_cr_date(s):

    year2 = int(s[4:])

    y = year2 if year2 > 16 else year2 + 100

    return 116 - y



cred['years_since_first_credit'] = cred['earliest_cr_line'].apply(read_earliest_cr_date)

cred.drop('earliest_cr_line', axis=1, inplace=True)
# metadata

metadata = dict()

for v in cred.dtypes.index:

    if v != "default_ind":

        metadata.setdefault(str(cred.dtypes[v]), []).append(v)
# Set missing to default values

cred.fillna({col:'Missing' for col in metadata['object']}, inplace=True)

cred.fillna({col:-99 for col in metadata['int64']}, inplace=True)

cred.fillna({col:-999.0 for col in metadata['float64']}, inplace=True)
# Convert Dates to Integers

base = pd.Timestamp('2000-01-01 00:00:00')

yr = pd.to_timedelta(pd.Timestamp('2001-01-01 00:00:00')-base)



try:

    for col in metadata['datetime64[ns]']:

        cred[col+'_t'] = (cred[col]-base) / yr

    cred.fillna({col+'_t':0 for col in metadata['datetime64[ns]']}, inplace=True)

    metadata['dates_t'] = [col+'_t' for col in metadata['datetime64[ns]']]

except KeyError:

    metadata['dates_t'] = []

    print('No datetime64[ns] Variables in the Model')
predictors = [col for mgroup in ['int64','float64','dates_t'] for col in metadata[mgroup]]
dum = pd.get_dummies(cred[metadata['object']])

dum.shape
cred = pd.concat([cred[predictors], dum],axis=1)

predictors += list(dum.columns)

del dum

gc.collect()
X_train, X_test, y_train, y_test = train_test_split(

    cred.values,

    y,

    test_size=0.20,

    random_state=19,

    stratify = y

)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



# specify your configurations as a dict

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': ['auc','binary_logloss'],

    'max_depth':10,

    'max_bin':255,

    'learning_rate': 0.004,

    'feature_fraction': 0.5,

    'pos_bagging_fraction': 0.6,

    'neg_bagging_fraction':0.05,

    'random_seed':2019,

    'min_data_in_leaf':400,

    'bagging_freq': 1,

    'verbosity': 0

}
out = dict()

gbm = lgb.train(

    params,

    lgb_train,

    num_boost_round=13000,

    verbose_eval=500,

    valid_sets=[lgb_train,lgb_eval],

    feature_name=predictors,

    evals_result = out

)
lgb.plot_metric(out, metric='auc');
lgb.plot_metric(out, metric='binary_logloss');
# Looks better in "Edit Mode"

#lgb.create_tree_digraph(

#    gbm, tree_index=0,

#    show_info=('internal_value', 'internal_count', 'leaf_count'),

#    precision = 3

#)



#You should be able to zoom in on this. Right-click on the graph.
lgb.plot_tree(

    gbm, tree_index=0,

    show_info=('internal_value', 'internal_count', 'leaf_count'),

    precision = 3,

    figsize=(44,35)

);
lgb.plot_importance(gbm, importance_type="gain", precision=0, max_num_features=30, figsize=(7,11));
#lst = sorted(zip(gbm.feature_importance(importance_type="gain"),gbm.feature_name()))

#[nm for imp, nm in lst if imp > 0]
def sensitivity(on_column,column_names,X,model):

    idx = [column_names.index(col) for col in on_column]

    stats = [(i,np.mean(X[:,i]),np.std(X[:,i])) for i in idx]

    res = []

    for i, mean_, std_ in stats:

        logdiff = []

        print("{:20} Mean: {:9.2f} St.Dev.: {:9.2f}".format(column_names[i], mean_, std_))

        original_column = X[:,i]

        X[:,i] = mean_ - 0.5 * std_

        prob = model.predict(X)

        logit0 = np.log( prob / (1-prob) )

        X[:,i] = mean_ + 0.5 * std_

        prob = model.predict(X)

        logit1 = np.log( prob / (1-prob) )

        srs = pd.Series(logit1-logit0, index = logit0, name = column_names[i])

        res += [srs.sort_index()]

        X[:,i] = original_column

    return res



smp = np.random.choice(len(X_test),400)

sens = sensitivity(['revol_util','total_rev_hi_lim'], predictors, X_test[smp], gbm)
import matplotlib.pyplot as plt



for srs in sens:

    avg = pd.Series(srs.mean(),index=srs.index)

    plt.scatter(srs.index, srs, marker='.', label='Logit Diff')

    plt.plot(avg.index,avg,'r--', label='Mean Effect')

    plt.title(srs.name)

    plt.legend()

    plt.grid()

    plt.show()
score_test = gbm.predict(X_test)

score = pd.DataFrame({"true":y_test,"score":score_test})



score.groupby("score")["true"].agg(["sum","size"])["size"].mean() # no score ties
buckets = 20

gains_table = score.groupby(pd.qcut(score.score,buckets))["true"].agg(["sum","size"])

gains_table['bad'] = gains_table['sum']

gains_table['good'] = gains_table['size'] - gains_table['sum']

gains_table['p_bad'] = gains_table['bad'] / gains_table['size']

gains_table['p_good'] = gains_table['good'] / gains_table['size']

gains_table = gains_table.reset_index(drop=True)

gains_table = gains_table.iloc[::-1]
tmp = gains_table[["size","bad","good"]].cumsum()

tmp = tmp / tmp.iloc[-1]

tmp.columns = ["prop","prop_bad","prop_good"]

gains_table = pd.concat([gains_table,tmp],axis=1)

plt.plot(gains_table.prop,gains_table.prop,'k:',label='prop')

plt.plot(gains_table.prop,gains_table.prop_bad,'r-',label='prop_bad')

plt.plot(gains_table.prop,gains_table.prop_good,'b-.',label='prop_good')

plt.legend()

plt.grid()

plt.show()

gains_table.index = gains_table["prop"].apply(lambda x: "{:.1%}".format(x))

gains_table = gains_table.drop(["sum","size","bad","good","prop"],axis=1)

gains_table.style.format("{:.4f}")