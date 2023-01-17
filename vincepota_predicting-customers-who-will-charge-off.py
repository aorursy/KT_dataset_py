import sqlite3

import pandas as pd 

import numpy as np 

import matplotlib.pylab as plt

%matplotlib inline



conn = sqlite3.connect('../input/database.sqlite') # This might take a while to run...

to_parse = ['issue_d' , 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']

df = pd.read_sql_query('select * from loan', con=conn, parse_dates = to_parse)
print('The shape is {}'.format(df.shape))

print('Memory : {} Mb'.format(int(df.memory_usage(deep=False).sum() / 1000000)))
check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))

check_null[check_null>0.6]
df.drop(check_null[check_null>0.5].index, axis=1, inplace=True) 

df.dropna(axis=0, thresh=30, inplace=True)
df.groupby('application_type').size().sort_values()
delete_me = ['index', 'policy_code', 'pymnt_plan', 'url', 'id', 'member_id', 'application_type', 'acc_now_delinq','emp_title', 'zip_code','title']

df.drop(delete_me , axis=1, inplace=True) 
# strip months from 'term' and make it an int

df['term'] = df['term'].str.split(' ').str[1]



#interest rate is a string. Remove % and make it a float

df['int_rate'] = df['int_rate'].str.split('%').str[0]

df['int_rate'] = df.int_rate.astype(float)/100.



# extract numbers from emp_length and fill missing values with the median

df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)

df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())



col_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index

for d in col_dates:

    df[d] = df[d].dt.to_period('M')
df.head()
# pivot_ui(df.sample(frac=0.1))

# opens a new window
loan_status_grouped = df.groupby('loan_status').size().sort_values(ascending=False)/len(df) * 100

loan_status_grouped
df['amt_difference'] = 'eq'

df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amt_difference'] = 'less'
# Make categorical



df['delinq_2yrs_cat'] = 'no'

df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'



df['inq_last_6mths_cat'] = 'no'

df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'



df['pub_rec_cat'] = 'no'

df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'yes'



# Create new metric

df['acc_ratio'] = df.open_acc / df.total_acc
features = ['loan_amnt', 'amt_difference', 'term', 

            'installment', 'grade','emp_length',

            'home_ownership', 'annual_inc','verification_status',

            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 

            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',  

            'loan_status'

           ]
X_clean = df.loc[df.loan_status != 'Current', features]

X_clean.head()
mask = (X_clean.loan_status == 'Charged Off')

X_clean['target'] = 0

X_clean.loc[mask,'target'] = 1
cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']



# Drop any residual missing value (only 24)

X_clean.dropna(axis=0, how = 'any', inplace = True)



X = pd.get_dummies(X_clean[X_clean.columns[:-2]], columns=cat_features).astype(float)

y = X_clean['target']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE



X_scaled = preprocessing.scale(X)

print(X_scaled)

print('   ')

print(X_scaled.shape)
def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):

    

    clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth= 6, n_estimators=100, max_features = 0.3),

            'LogisticRegression' : LogisticRegression(),

            #'GaussianNB': GaussianNB(),

            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)

            }

    cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']



    models_report = pd.DataFrame(columns = cols)

    conf_matrix = dict()



    for clf, clf_name in zip(clfs.values(), clfs.keys()):



        clf.fit(X_train, y_train)



        y_pred = clf.predict(X_test)

        y_score = clf.predict_proba(X_test)[:,1]



        print('computing {} - {} '.format(clf_name, model_type))



        tmp = pd.Series({'model_type': model_type,

                         'model': clf_name,

                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),

                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),

                         'precision_score': metrics.precision_score(y_test, y_pred),

                         'recall_score': metrics.recall_score(y_test, y_pred),

                         'f1_score': metrics.f1_score(y_test, y_pred)})



        models_report = models_report.append(tmp, ignore_index = True)

        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)



        plt.figure(1, figsize=(6,6))

        plt.xlabel('false positive rate')

        plt.ylabel('true positive rate')

        plt.title('ROC curve - {}'.format(model_type))

        plt.plot(fpr, tpr, label = clf_name )

        plt.legend(loc=2, prop={'size':11})

    plt.plot([0,1],[0,1], color = 'black')

    

    return models_report, conf_matrix
#mpl.rc("savefig", dpi=300)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)

models_report, conf_matrix = run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced')
models_report
conf_matrix['LogisticRegression']
index_split = int(len(X)/2)

X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])

X_test, y_test = X_scaled[index_split:], y[index_split:]



#scores = cross_val_score(clf, X_scaled, y , cv=5, scoring='roc_auc')



models_report_bal, conf_matrix_bal = run_models(X_train, y_train, X_test, y_test, model_type = 'Balanced')
models_report_bal
conf_matrix_bal['LogisticRegression']