# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# General tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# For the tree visualization
import pydot
from IPython.display import Image
#from sklearn.externals.six import StringIO

# For scoring
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse


# For validation
from sklearn.model_selection import train_test_split as split

%matplotlib inline        
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
df.head()
print(df.shape)
print(df.info())
#replacing spaces with an underscore and return lower case text
df.columns = [i.replace(' ', '_').lower() for i in df.columns]
df.columns
#replacing values in education column with description
df['education']=df['education'].astype(str)
d={'1':'undergrad', '2': 'graduate', '3': 'professional'}
df['education']=df['education'].map(d)
df.head()
df.set_index('id', inplace=True)
#df.head()
loans_counts=df['personal_loan'].value_counts().to_frame()
loans_counts
df.describe()
df.nunique()
features = ['age', 'experience', 'family','income']
df[features].hist(figsize=(12, 8))
#Аdding a new column, mortgage indicator
def is_mortgage(row):
    if row['mortgage']>0:
        return 1
    else:
        return 0
df['is_mortgage']=df.apply (lambda row: is_mortgage(row),axis=1)
df.head()
mortgage_counts=df['is_mortgage'].value_counts(normalize=True).to_frame()
mortgage_counts
df_mor=df[df['is_mortgage'].apply (lambda is_mortgage:is_mortgage==1)]
df_mor['mortgage'].hist();
df_cc=df[df['creditcard'].apply (lambda creditcard:creditcard==1)]
df_cc['ccavg'].hist();
cc_counts=df['creditcard'].value_counts(normalize=True).to_frame()
cc_counts
df_test=df[['securities_account', 'cd_account', 'online', 'creditcard', 'personal_loan']]
df_grp=df_test.groupby('personal_loan').sum()
df_grp
sns.regplot(x='income', y='mortgage', data=df_mor)
plt.ylim(0,);
sns.regplot(x='income', y='ccavg', data=df_cc)
plt.ylim(0,);
fig, axarr=plt.subplots(1,2, figsize=(12,4))
df.groupby(['personal_loan'])['income'].mean().plot.bar(ax=axarr[0])
df_mor.groupby('personal_loan')['mortgage'].mean().plot.bar(ax=axarr[1])

axarr[0].set_title('Income vs personal_loan')
axarr[1].set_title('Mortgage vs personal_loan');

df['ccavg']=df['ccavg'].astype('int64')
corr = df.loc[:,df.dtypes == 'int64'].corr()
sns.set(rc={'figure.figsize':(10,7)})
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True));
#correlation between all of the numeric variables in the data frame and the target-personal_loan
#Pandas’ corrwith() method return a pair-wise correlation
correlations = df.corrwith(df['personal_loan']).iloc[:-1].to_frame()
correlations['abs'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('abs', ascending=False)[0]
fig, ax = plt.subplots(figsize=(10,15))
sns.heatmap(sorted_correlations.to_frame(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);
columns_to_show = [ 'income','ccavg', 'cd_account', 'mortgage', 'family']
df.groupby(['personal_loan'])[columns_to_show].agg([np.mean,np.min, np.max])
#Pearson Correlation
from scipy import stats
pearson_coef, p_value=stats.pearsonr(df['income'], df['personal_loan'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value);

#Conclusion:
#Since the p-value is  <  0.001, the correlation between income and personal_loan is statistically significant,
#although the linear relationship isn't extremely strong (~0.5)
pearson_coef, p_value=stats.pearsonr(df['ccavg'], df['personal_loan'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value);
pearson_coef, p_value=stats.pearsonr(df['mortgage'], df['personal_loan'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value);
#correlation with non-numeric column
sns.boxplot(df['education'],
        df['income']).set_title('education vs. income');
y=df['income']
x=df['experience']
plt.scatter(x,y)

plt.title('Experience vs Income')
plt.xlabel('experience')
plt.ylabel('income');
plt.figure(figsize=(35,10)) # adjust the fig size 
sns.boxplot(df['experience'], df['income']).set_title('Income varies by Experience');
df_dr=df.drop(['age', 'experience',  'zip_code', 'education', 'securities_account','online'],  axis=1)
df_dr.head()
df_loan=df_dr[df_dr['personal_loan'].apply (lambda personal_loan:personal_loan==1)]
df_loan.head()
pd.crosstab(df['personal_loan'], df['cd_account'])
sns.countplot(x='cd_account', hue='personal_loan', data=df);
sns.countplot(x='is_mortgage', hue='personal_loan', data=df);
df_mor['mortgage'].plot(kind='density', subplots=True, 
                  layout=(1, 2), sharex=False, figsize=(12, 4));
sns.lmplot('income', 'ccavg', data=df, 
           hue='personal_loan', fit_reg=1);