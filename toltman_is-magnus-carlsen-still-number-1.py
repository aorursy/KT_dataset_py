import pandas as pd
df = pd.read_csv('../input/fide_historical.csv')
df.head()
df.info()
df['date'] = pd.to_datetime(df['ranking_date'])
df[df['name'] == 'Carlsen, Magnus'].head()
df.loc[df.loc[df['name'] == 'Carlsen, Magnus', ['rating']].idxmax()]
plt = df[df['name'] == 'Carlsen, Magnus'].plot(x='date', y='rating',

                                              title='Magnus Carlsen\'s Rating Progress',

                                              figsize=(8,5),

                                              legend=False)
df[df['date'] == '2017-06-27'].sort_values(by='rating', ascending=False).head()
plt = df[df['name'] == 'So, Wesley'].plot(x='date', y='rating', 

                                          title='Wesley So\'s Rating Progress',

                                          color='orange', figsize=(8,5),

                                          legend=False

                                         )
ratings = pd.pivot_table(data=df, index='date', values='rating', columns='name')
player_list = ['Carlsen, Magnus', 

               'So, Wesley', 

               'Caruana, Fabiano']

plt = ratings[player_list][ratings.index > pd.to_datetime('2010-01-01')].plot(figsize=(8,5))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.set_title('Various Player Ratings History')

plt.set_xlabel('Year')
player_list = ['Carlsen, Magnus', 

               'Kasparov, Garry']

plt = ratings[player_list].plot(figsize=(8,5))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.set_title('Various Player Ratings History')

plt.set_xlabel('Year')
ratings['max'] = ratings.loc[:, ratings.columns != 'Carlsen, Magnus'].max(axis=1)

ratings['diff'] = ratings['Carlsen, Magnus'] - ratings['max']
plt = ratings['diff'][ratings.index > pd.to_datetime('2007-01-01')].plot(figsize=(8,5))

plt.set_title('Difference in Rating Carlsen and Next Clostest Challenger')
player_list = ['Carlsen, Magnus', 'max',]

plt = ratings[player_list][ratings.index > pd.to_datetime('2007-01-01')].plot(figsize=(10,6))

plt.legend(loc='lower right', labels=['Magnus Carlsen', 'Next Highest Player'])

plt.set_title('Magnus Carlsen vs Rest of the World').set_fontsize(16)

plt.set_xlabel('Year').set_fontsize(14)

plt.set_ylabel('FIDE Rating').set_fontsize(14)
ratings.loc[ratings.index > pd.to_datetime('2007-01-01'), 

            ratings.columns != 'Carlsen, Magnus'].idxmax(axis=1).unique()