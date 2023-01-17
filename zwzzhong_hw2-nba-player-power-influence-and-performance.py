import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
nba = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
nba.head()
nba.info()
nba.describe() 
# filling the null value in TWITTER_FAVORITE_COUNT and TWITTER_RETWEET_COUNT
# 3 missing values in TWITTER_FAVORITE_COUNT and TWITTER_RETWEET_COUNT
# calculate the mean and the number of missing value for these two columns

nba_twi_fav_mean = round(nba['TWITTER_FAVORITE_COUNT'].mean())
nba_twi_fav_number = nba['TWITTER_FAVORITE_COUNT'].isnull().sum()
print('***************************************************************\n')
print(f'The mean value of TWITTER_FAVORITE_COUNT is:', nba_twi_fav_mean)
print(f'The number of missing value in TWITTER_FAVORITE_COUNT is:', nba_twi_fav_number, '\n')
print('***************************************************************\n')

nba_twi_re_mean = round(nba['TWITTER_RETWEET_COUNT'].mean())
nba_twi_re_number = nba['TWITTER_RETWEET_COUNT'].isnull().sum()
print(f'The mean value of TWITTER_RETWEET_COUNT is:', nba_twi_re_mean)
print(f'The number of missing value in TWITTER_RETWEET_COUNT is:', nba_twi_re_number,'\n')
print('***************************************************************\n')

# drop missing values and fill them with mean values
nba['TWITTER_FAVORITE_COUNT'].dropna()
nba['TWITTER_FAVORITE_COUNT'][np.isnan(nba['TWITTER_FAVORITE_COUNT'])] = nba_twi_fav_mean
nba['TWITTER_RETWEET_COUNT'].dropna()
nba['TWITTER_RETWEET_COUNT'][np.isnan(nba['TWITTER_RETWEET_COUNT'])] = nba_twi_re_mean

print(f'Is there any missing value in TWITTER_FAVORITE_COUNT?', nba['TWITTER_FAVORITE_COUNT'].isnull().values.any())
print(f'Is there any missing value in TWITTER_RETWEET_COUNT?', nba['TWITTER_RETWEET_COUNT'].isnull().values.any(),'\n')
print('***************************************************************')
nba_corr = nba.corr()
plt.subplots(figsize=(20,15))
columns = nba_corr.nlargest(10, 'WINS_RPM')['WINS_RPM'].index 
coff = np.corrcoef(nba_corr[columns].values.T) 
heatmap = sns.heatmap(coff, annot=True, cmap = 'YlGnBu',yticklabels=columns.values, xticklabels=columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba)
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="WINS_RPM", data=nba)
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="WINS_RPM", data=nba)
results_fav = smf.ols('WINS_RPM ~TWITTER_FAVORITE_COUNT', data=nba).fit()
print (results_fav.summary())
results_retweet = smf.ols('WINS_RPM ~TWITTER_RETWEET_COUNT', data=nba).fit()
print (results_retweet.summary())
results_salary = smf.ols('WINS_RPM ~SALARY_MILLIONS', data=nba).fit()
print (results_salary.summary())
results_three_var = smf.ols('WINS_RPM ~ SALARY_MILLIONS + TWITTER_RETWEET_COUNT + TWITTER_FAVORITE_COUNT', data=nba).fit()
print (results_three_var.summary())
results_all_var = smf.ols('WINS_RPM ~ SALARY_MILLIONS + TWITTER_RETWEET_COUNT + TWITTER_FAVORITE_COUNT + POINTS + TRB + AST + STL + BLK + TOV', data=nba).fit()
print (results_all_var.summary())
results_all_var = smf.ols('WINS_RPM ~ TWITTER_RETWEET_COUNT + TWITTER_FAVORITE_COUNT + POINTS + TRB + AST + STL + BLK + TOV', data=nba).fit()
print (results_all_var.summary())