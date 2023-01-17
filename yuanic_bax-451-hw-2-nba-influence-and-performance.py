import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
## load data: team_valuations
nba_2017_team_val = pd.read_csv("../input/nba_2017_att_val.csv")
nba_2017_team_val.columns = ['Unnamed: 0', 'TEAM', 'GMS', 'TOTAL ATTENDANCE', 'AVG ATTENDANCE', 'PCT', 'VALUE_MILLIONS']
nba_2017_team_val.head()
## number of teams in NBA
len(nba_2017_team_val)
## load data: salary
nba_2017_salary = pd.read_csv("../input/nba_2017_salary.csv")
nba_2017_salary.head()
## load data: salary twitter wiki
nba_2017_salary_wiki_twitter = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
nba_2017_salary_wiki_twitter.head()
## merge salray twitter wiki with salary to get team name & team val
nba_2017_salary_wiki_twitter_tm  = pd.merge(nba_2017_salary_wiki_twitter,nba_2017_salary,how='left',left_on='PLAYER', right_on='NAME')
nba_2017_salary_wiki_twitter_tm_val = pd.merge(nba_2017_salary_wiki_twitter_tm,nba_2017_team_val,how='left',left_on='TEAM_y', right_on='TEAM')
nba_2017_salary_wiki_twitter_tm_val.columns
attributes = nba_2017_salary_wiki_twitter_tm_val[['PLAYER','TEAM_y','VALUE_MILLIONS','SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','3P','3PA']]
attributes.columns = ['PLAYER','TEAM','VALUE_MILLIONS','SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','PT3','PT3_A']
sns.heatmap(attributes.corr())

fig = plt.figure()
sns.distplot(nba_2017_team_val['VALUE_MILLIONS'])
plt.title('NBA Teams Valuation', fontsize=16)
nba_2017_team_val['VALUE_MILLIONS'].describe()
nba_2017_team_val.sort_values('VALUE_MILLIONS')
## Team valuation and attendance
fig = plt.figure()
sns.pairplot(nba_2017_team_val[['VALUE_MILLIONS','AVG ATTENDANCE']])
## number of players with data
len(attributes)
## Team Super Star: Personal Attributes (Stats from then Best Player on Team for each attribute)
attributes['PT3_Success']=attributes['PT3']/attributes['PT3_A']
super_star = attributes[['TEAM','VALUE_MILLIONS','SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','PT3','PT3_A','PT3_Success']].groupby('TEAM').max()
super_star.head()
attributes[attributes['TEAM']=='New York Knicks'].sort_values('TWITTER_RETWEET_COUNT')[['TEAM','PLAYER','PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','PT3','PT3_A','PT3_Success']]
## predicting team value using superstars
import statsmodels.api as sm

results = smf.ols('VALUE_MILLIONS ~ TWITTER_RETWEET_COUNT', data=super_star.dropna()).fit()
print(results.summary())

#pageviews: rsq 0.132 pval 0.053 aic 462.1 **
#twitter_favourite_count: rsq 0.068 pval 0.171 aic 464.1 **
#twitter_retweet_count: rsq 0.158 pval 0.033 aic 461.2 ***
#pf (personal fouls): rsq 0.025 pval 0.413 aic 465.5 
#points: rsq 0.018 pval 0.483 aic 465.6
#PT3: rsq 0.059 pval 0.204 aic 464.4
#PT3_A: rsq 0.074 pval 0.153 aic 463.9
#PT3_Success: rsq 0.001 pval 0.882 aic 466.2

## Teamwork: Team Attributes (Stats Total from all players within the Team for each attribute)
teamwork = attributes[['TEAM','SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','PT3','PT3_A']].groupby('TEAM').sum()
teamwork = pd.merge(teamwork,nba_2017_team_val,how='left',left_on='TEAM', right_on='TEAM')
teamwork['PT3_Success']=teamwork['PT3']/teamwork['PT3_A']
teamwork.columns
teamwork[teamwork['TEAM']=='New York Knicks'][['TEAM','PAGEVIEWS', 'TWITTER_FAVORITE_COUNT',
       'TWITTER_RETWEET_COUNT','PF','POINTS','PT3','PT3_A','PT3_Success']]
## predicting team value using teamwork

results = smf.ols('VALUE_MILLIONS ~ TWITTER_RETWEET_COUNT+POINTS', data=teamwork.dropna()).fit()
print(results.summary())

#pageviews: rsq 0.153 pval 0.036 aic 461.4 ***
#twitter_favourite_count: rsq 0.086 pval 0.122 aic 463.6 **
#twitter_retweet_count: rsq 0.208 pval 0.013 aic 459.4 ***
#pf (personal fouls): rsq 0.002 pval 0.800 aic 466.1 
#points: rsq 0.003 pval 0.767 aic 466.1
#PT3: rsq 0.05 pval 0.245 aic 464.7
#PT3_A: rsq 0.061 pval 0.197 aic 464.4
#PT3_Success: rsq 0.002 pval 0.824 aic 464.4

## superstar contribution to the team
combined = pd.merge(super_star,teamwork,how='left',left_on='TEAM', right_on='TEAM')
combined.columns
combined_tweet = combined[['TEAM','VALUE_MILLIONS_x','TWITTER_RETWEET_COUNT_x','TWITTER_RETWEET_COUNT_y']]
combined_tweet.columns = ['TEAM','VALUE_MILLIONS','TWITTER_RETWEET_COUNT_SUPERSTAR','TWITTER_RETWEET_COUNT_TEAMWORK']
combined_tweet['SUPERSTAR_RETWEET_CONTRIBUTION'] = combined_tweet['TWITTER_RETWEET_COUNT_SUPERSTAR']/combined_tweet['TWITTER_RETWEET_COUNT_TEAMWORK']
player_cnt = attributes[['TEAM','PLAYER']].groupby('TEAM').count()
combined_tweets_player_cnt =pd.merge(combined_tweet,player_cnt,how='left',left_on='TEAM', right_on='TEAM')
sns.distplot(combined_tweet['SUPERSTAR_RETWEET_CONTRIBUTION'])
## predicting team value using teamwork and superstars
## TWEETS
results = smf.ols('VALUE_MILLIONS ~ SUPERSTAR_RETWEET_CONTRIBUTION + TWITTER_RETWEET_COUNT_TEAMWORK', data=combined_tweets_player_cnt.dropna()).fit()
print(results.summary())

#tweet_superstar+tweet_team: ad.rsq 0.203 pval 0.185,0.069 aic 459.4
#tweet_superstar_con+tweet_team: ad.rsq 0.175 pval 0.354,0.010 aic 460.5
#tweet_superstar_con+tweet_superstar: ad.rsq 0.129 pval 0.314,0.022 aic 462.0
#tweet_teamwork+player: ad.rsq 0.159 pval 0.012,0.536 aic 461.0


