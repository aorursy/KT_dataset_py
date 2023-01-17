# import the necessary libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.preprocessing import StandardScaler



import sklearn.linear_model as skl_lm

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.model_selection import train_test_split





import statsmodels.api as sm

import statsmodels.formula.api as smf



# initialize some package settings

sns.set(style="whitegrid", color_codes=True, font_scale=1.3)



%matplotlib inline
# read in the data and check the first 5 rows

df = pd.read_csv('../input/data.csv', index_col=0)

df.head()
# remove the 'Unnamed: 32' column

df = df.drop('Unnamed: 32', axis=1)
# Convert the 'target' variable to numeric

df['target'] = df['diagnosis'].apply(lambda x : 1 if x == 'M' else 0)  # Convert to numeric

df = df.drop('diagnosis',axis=1)
# import packages

import pandas as pd

import numpy as np

import pandas.core.algorithms as algos

from pandas import Series

import scipy.stats.stats as stats

import re

import traceback

import string



max_bin = 20

force_bin = 3



# define a binning function

def mono_bin(Y, X, n = max_bin):

    

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]

    r = 0

    while np.abs(r) < 1:

        try:

            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})

            d2 = d1.groupby('Bucket', as_index=True)

            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)

            n = n - 1 

        except Exception as e:

            n = n - 1



    if len(d2) == 1:

        n = force_bin         

        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))

        if len(np.unique(bins)) == 2:

            bins = np.insert(bins, 0, 1)

            bins[1] = bins[1]-(bins[1]/2)

        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 

        d2 = d1.groupby('Bucket', as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["MIN_VALUE"] = d2.min().X

    d3["MAX_VALUE"] = d2.max().X

    d3["COUNT"] = d2.count().Y

    d3["EVENT"] = d2.sum().Y

    d3["NONEVENT"] = d2.count().Y - d2.sum().Y

    d3=d3.reset_index(drop=True)

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    

    return(d3)



def char_bin(Y, X):

        

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]    

    df2 = notmiss.groupby('X',as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["COUNT"] = df2.count().Y

    d3["MIN_VALUE"] = df2.sum().Y.index

    d3["MAX_VALUE"] = d3["MIN_VALUE"]

    d3["EVENT"] = df2.sum().Y

    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    d3 = d3.reset_index(drop=True)

    

    return(d3)



def data_vars(df1, target):

    

    stack = traceback.extract_stack()

    filename, lineno, function_name, code = stack[-2]

    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]

    final = (re.findall(r"[\w']+", vars_name))[-1]

    

    x = df1.dtypes.index

    count = -1

    

    for i in x:

        if i.upper() not in (final.upper()):

            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:

                conv = mono_bin(target, df1[i])

                conv["VAR_NAME"] = i

                count = count + 1

            else:

                conv = char_bin(target, df1[i])

                conv["VAR_NAME"] = i            

                count = count + 1

                

            if count == 0:

                iv_df = conv

            else:

                iv_df = iv_df.append(conv,ignore_index=True)

    

    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})

    iv = iv.reset_index()

    return(iv_df,iv) 
# Weight of Evidence encode the data

final_iv, IV = data_vars(df, df.target)
# Take a look at the variable names and their respective Information Value

IV.sort_values('IV')
transform_vars_list = df.columns.difference(['target'])

transform_prefix = '' # leave this value blank if you need to replace the original column values
for var in transform_vars_list:

    small_df = final_iv[final_iv['VAR_NAME'] == var]

    transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))

    replace_cmd = ''

    replace_cmd1 = ''

    for i in sorted(transform_dict.items()):

        replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '

        replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '

    replace_cmd = replace_cmd + '0'

    replace_cmd1 = replace_cmd1 + '0'

    if replace_cmd != '0':

        try:

            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))

        except:

            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))
# Split the data into training and testing sets

X = df

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
import statsmodels as sm

import pandas as pd

import sklearn.preprocessing as preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from scipy import stats



logreg = LogisticRegression(fit_intercept = False, C = 1e15, class_weight='balanced')

model_log = logreg.fit(X_train, y_train)

model_log
y_pred = model_log.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred)) 

print("The accuracy score is" + " "+ str(accuracy_score(y_test, y_pred)))