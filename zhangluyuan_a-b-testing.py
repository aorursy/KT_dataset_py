import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
random.seed(42)
df=pd.read_csv('ab_data.csv')
df.head()
df.shape
df.user_id.nunique()
(df.converted==1).mean()
((df.group=='treatment') & (df.landing_page=='old_page')).sum()+ ((df.group=='control') & (df.landing_page=='new_page')).sum()
df.info()
#identify misaligned rows
df['misaligned']=((df.group=='treatment') & (df.landing_page=='old_page')) | ((df.group=='control') & (df.landing_page=='new_page'))
#extract rows where misgligned==False
df2=df.query('misaligned==False')
df2.shape
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2['user_id'].nunique()
df2['user_id'].value_counts().sort_values(ascending=False).head()
df2.query('user_id==773192')
df2.drop(1899, axis=0, inplace=True)
df2.shape
(df2['converted']==1).mean()
actual_pold=(df2.query('group=="control"')['converted']==1).mean()
actual_pold
actual_pnew=(df2.query('group=="treatment"')['converted']==1).mean()
actual_pnew
(df2['landing_page']=='new_page').mean()
pnew_null=(df2['converted']==1).mean()
pnew_null
pold_null=(df2['converted']==1).mean()
pold_null
p_null=pnew_null
n_new=(df2['landing_page']=='new_page').sum()
n_new
n_old=(df2['landing_page']=='old_page').sum()
n_old
new_page_converted=np.random.binomial(n_new, p_null)
old_page_converted=np.random.binomial(n_old, p_null)
diff=new_page_converted/n_new-old_page_converted/n_old
diff
p_diffs=[]
p_diffs = np.random.binomial(n_new, p_null, 10000)/n_new - np.random.binomial(n_old, p_null, 10000)/n_old   
plt.hist(p_diffs, bins=200)
plt.xlim(-0.005, 0.005)
plt.xlabel('p_diffs')
plt.ylabel('counts')
plt.title('simulated p_diffs distribution')
plt.axvline(0.000, color='red');
actual_diff=actual_pnew-actual_pold
actual_diff, actual_pnew, actual_pold
actual_diff=actual_pnew-actual_pold
(p_diffs>actual_diff).mean()
import statsmodels.api as sm

convert_old = (df2.query('landing_page=="old_page"')['converted']==1).sum()
convert_new = (df2.query('landing_page=="new_page"')['converted']==1).sum()
n_old = (df2['landing_page']=='old_page').sum()
n_new=(df2['landing_page']=='new_page').sum()

convert_old, convert_new, n_old, n_new         
z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')
z_score, p_value
from scipy.stats import norm
# Tells us how significant our z-score is
print(norm.cdf(z_score))

# for our single-sides test, assumed at 95% confidence level, we calculate: 
print(norm.ppf(1-(0.05)))
df2['intercept']=1
df2['ab_page']=pd.get_dummies(df2['group'])['treatment']
df2.head()
lm=sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results=lm.fit()
results.summary()
# read in the country table
country=pd.read_csv('countries.csv')
# merge country and df2 tables
df2=pd.merge(df2,country, on=['user_id'])
df2.head()
# create dummy columns for country
df2[['CA', 'UK','US']]=pd.get_dummies(df2['country'])
df2.head(1)
# run logistic regression on counties
lm=sm.Logit(df2['converted'], df2[['intercept', 'CA', 'UK']])
results=lm.fit()
results.summary()
lm=sm.Logit(df2['converted'], df2[['intercept', 'ab_page', 'CA', 'UK']])
results=lm.fit()
results.summary()

df2['datetime']=pd.to_datetime(df2['timestamp'], errors='coerce')
df2['dow']=df2['datetime'].dt.weekday_name
df2[pd.get_dummies(df2['dow']).columns]=pd.get_dummies(df2['dow'])
df2.head(2)
# perform logistic regression to see if there is significant difference of conversion rates in different day of week
lm=sm.Logit(df2['converted'], df2[['intercept','ab_page','Friday', 'Monday', 'Saturday', 'Thursday', 'Tuesday',
       'Wednesday']])
results=lm.fit()
results.summary()
# calculate mean conversion rates on each day of week
dow_columns=pd.get_dummies(df2['dow'])
dow_rate=pd.DataFrame([(lambda x:(df2[x] * df2.converted).sum()/df2[x].sum()) (x) for x in dow_columns], index=list(pd.get_dummies(df2['dow']).columns), columns=['conversion_rate'])
dow_rate
# create a sub-dataframe that only included Friday and Monday data
sub_df2=df2.query('dow=="Friday" | dow=="Monday"' )
# run a logistic regression to check the significance level of the conversion rate difference between the two days
# Friday is the baseline
lm=sm.Logit(sub_df2['converted'], sub_df2[['intercept', 'ab_page','Monday']])
results=lm.fit()
results.summary()
# check when did the test start and end
df2['datetime'].min(), df2['datetime'].max()
df2['day']=df2['datetime'].dt.day
#create a dataframe that aggregates mean conversion rate of old and new pages of individual day
df2_by_day=pd.DataFrame()
old_df=df2.query('landing_page=="old_page"')
new_df=df2.query('landing_page=="new_page"')
df2_by_day['old_rate']=old_df.groupby('day')['converted'].mean()
df2_by_day['new_rate']=new_df.groupby('day')['converted'].mean()
df2_by_day.reset_index(inplace=True)
# create a scatter plot to see if there is increase in conversion rate as time goes on
plt.scatter(df2_by_day['day'],  df2_by_day['old_rate'], color='green',label='p_old')
plt.scatter(df2_by_day['day'], df2_by_day['new_rate'], color='red',label='p_new')
plt.xlabel('day of January')
plt.ylabel('conversion rate')
#plt.ylim(0.11, 0.13)
plt.legend()
plt.title('conversion rate at different days of the month');
# create a scatter plot to see if there is increase in conversion rate as time goes on
plt.scatter(df2_by_day['day'],  df2_by_day['new_rate']-df2_by_day['old_rate'], color='blue',label='p_new - p_old')
plt.xlabel('day of January')
plt.ylabel('p_new - p_old')
plt.ylim(-0.05, 0.05)
plt.legend()
plt.title('p_new - p_old');
df2_by_day['intercept']=1
lm=sm.OLS(df2_by_day['old_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()
lm=sm.OLS(df2_by_day['new_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()
lm=sm.OLS(df2_by_day['new_rate']-df2_by_day['old_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()
from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])