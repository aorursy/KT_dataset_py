import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
rate = pd.read_csv('../input/dog_rates_tweets.csv')
rate.head()
#Find tweets that contain an “n/10” rating (because not all do). Extract the numeric rating. Exclude tweets that don't contain a rating.
new = rate['text'].str.extract('(\d+(\.\d+)?)/10',expand=True)
rate['rating'] = new[0].astype(float)
rate = rate[rate['rating'].notnull()]
#Remove outliers: there are a few obvious ones. Exclude rating values that are too large to make sense. (Maybe larger than 25/10?)
rate = rate.query("rating < 20")
#Make sure the 'created_at' column is a datetime value, not a string. 
rate['created_at'] = pd.to_datetime(rate['created_at'])
#Create a scatter plot of date vs rating
sns.set(style = 'whitegrid',color_codes=True,font_scale=1.3, rc={"lines.linewidth": 2.5})
plt.xticks(rotation=25)
plt.scatter(rate['created_at'].values, rate['rating'].values, c='b',alpha=0.5, s=5)
rate['timestamp'] = rate['created_at'].apply(lambda x:x.timestamp())
fit = slope , intercept, r_value, p_value, slope_std_error = stats.linregress(rate['timestamp'], rate['rating'])
plt.plot(rate['created_at'].values, rate['timestamp']*fit.slope + fit.intercept, 'r-',linewidth=3)
plt.show()
#Create rate range
def rateSplit(rate):
    if rate < 10:
        rate = "[0,10)"
    elif rate >=10 and rate <= 12:
        rate = "[10-12]"
    elif rate >= 13:
        rate = "(12,∞)"
    return rate

rate['rateRange'] = rate['rating'].apply(lambda x: rateSplit(x))
rate['rateRange'].head()
rate['YearMonth'] = rate['created_at'].apply(lambda x:x.strftime('%Y-%m'))
# rate2 = rate.groupby(['YearMonth' , 'rateRange'])['rateRange'].count()
#### VERY VERY USEFUL WAY FOR STACKED BAR CHART !!! ####
rate2 = rate.groupby(['YearMonth' , 'rateRange'])['rateRange'].count().unstack()
# rate2 = rate.groupby(['YearMonth'])['rateRange'].count()
rate2.plot(kind='bar', stacked = True)
plt.show()
x = rate['timestamp']
y = rate['rating']
# X = sm.add_constant(x)
result = sm.OLS(y, x).fit()
print(result.summary()) 
plt.hist(result.resid, bins=50)
plt.show()


