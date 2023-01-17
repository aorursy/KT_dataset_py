DF = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')
df = DF
df.head()
dfSC = df[df.county == "Santa Clara"]
dfSC
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("darkgrid")
fig = plt.figure(figsize=(20, 10))
axL = sns.barplot(x='date',
              y='cases',
              data=dfSC,
             color='b',
                 label='Confirmed cases')
axL.set_xticklabels(dfSC.date, rotation=90)
axL.set(ylabel='Confirmed cases')
axL.legend(loc='upper left')

axR = axL.twinx()
axR = sns.pointplot(x='date',
              y='deaths',
              data=dfSC,
             color='r')
plt.tight_layout()
axR.legend(['Death count'], loc='center left')
axR.set(ylabel='Death count')
axL = axL.set_title("Santa clara time view", fontsize=20, pad=-10)
dfCA = df[df.state == 'California']
fig = plt.figure(figsize=(20, 10))
n = 5
dfCA.date = pd.to_datetime(dfCA.date)
topNCounties = dfCA.nlargest(1, 'date', keep='all').nlargest(n, 'cases', keep='all')['county'].values
dfCAtopNCounties = dfCA[dfCA.county.isin(topNCounties)]
axL = sns.pointplot(x='date', y='cases', hue='county', data=dfCAtopNCounties[dfCAtopNCounties.date > '2020-03-10'], ci=0)
axL.set_xticklabels(dfCAtopNCounties.date.drop_duplicates().dt.strftime('%Y-%m-%d'), rotation=90)
axL.set(ylabel='Confirmed cases', xlabel='Date')
axL.legend()
axL.set_title("Top 5 Conties in CA by Confirmed cases ", fontsize=20, pad=0)
plt.tight_layout()
dfCA = df[df.state == 'California']
fig = plt.figure(figsize=(20, 10))
n = 5
dfCA.date = pd.to_datetime(dfCA.date)
topNCounties = dfCA.nlargest(1, 'date', keep='all').nlargest(n, 'deaths', keep='all')['county'].values
dfCAtopNCounties = dfCA[dfCA.county.isin(topNCounties)]
axL = sns.pointplot(x='date', y='deaths', hue='county', data=dfCAtopNCounties[dfCAtopNCounties.date > '2020-03-10'], ci=0)
axL.set_xticklabels(dfCAtopNCounties.date.drop_duplicates().dt.strftime('%Y-%m-%d'), rotation=90)
axL.set(ylabel='Deaths', xlabel='Date')
axL.legend()
axL.set_title("Top 5 Conties in CA by Deaths", fontsize=20, pad=0)
plt.tight_layout()