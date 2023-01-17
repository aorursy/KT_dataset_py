import pandas as pd



from plotly import graph_objects as go
f = '/kaggle/input/screen-actors-guild-awards/screen_actor_guild_awards.csv'

df = pd.read_csv(f)
df.head()
df['year'] = df.year.apply(lambda x: str(x)[:4])

years = df.year.unique()[:-2]
data = df.loc[df.won].groupby('show', as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure(data=[

    go.Bar(x=data['show'], y=data['won'], text=data['won'], textposition='auto')

])



fig.update_layout(title='Biggest Winners')



fig.show()
data = df.loc[df.won == False].groupby('show', as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure(data=[

    go.Bar(x=data['show'], y=data['won'], text=data['won'], textposition='auto')

])



fig.update_layout(title='Biggest Nominees')



fig.show()
data = df[['year', 'show', 'won']].groupby(['year', 'show'], as_index=False).count().sort_values(by='year', ascending=False)

shows_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(shows_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['show'], textposition='auto')])

fig.update_layout(title='Shows by year - Nominees')

fig.show()
data = df[['year', 'show', 'won']].loc[df.won].groupby(['year', 'show'], as_index=False).count().sort_values(by='year', ascending=False)

shows_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(shows_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['show'], textposition='auto')])

fig.update_layout(title='Shows by year - Winners')

fig.show()
data = df[['year', 'show', 'won']].loc[df.won == False].groupby(['year', 'show'], as_index=False).count().sort_values(by='year', ascending=False)

shows_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(shows_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['show'], textposition='auto')])

fig.update_layout(title='Shows by year - Nominations')

fig.show()
data = df.groupby('show', as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure()



y = [df['won'].loc[(df['show'] == show[1]) & (df.won)].count() for show in data['show'].items()]

fig.add_trace(go.Bar(x=data['show'], y=y, text=y, textposition='auto', name='# won'))



y = [df['won'].loc[(df['show'] == show[1]) & (df.won == False)].count() for show in data['show'].items()]

fig.add_trace(go.Bar(x=data['show'], y=y, text=y, textposition='auto', name='# nominee'))



fig.update_layout(title='Show that more appeared', barmode='stack')



fig.show()
data = df.loc[df.won].groupby(['full_name'], as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure(data=[

    go.Bar(x=data['full_name'], y=data['won'], text=data['won'], textposition='auto')

])



fig.update_layout(title='People with most statues')



fig.show()
data = df.loc[df.won == False].groupby(['full_name'], as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure(data=[

    go.Bar(x=data['full_name'], y=data['won'], text=data['won'], textposition='auto')

])



fig.update_layout(title='People that were most nominated')



fig.show()
data = df.groupby('full_name', as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure()



y = [df['won'].loc[(df['full_name'] == show[1]) & (df.won)].count() for show in data['full_name'].items()]

fig.add_trace(go.Bar(x=data['full_name'], y=y, text=y, textposition='auto', name='# won'))



y = [df['won'].loc[(df['full_name'] == show[1]) & (df.won == False)].count() for show in data['full_name'].items()]

fig.add_trace(go.Bar(x=data['full_name'], y=y, text=y, textposition='auto', name='# nominee'))



fig.update_layout(title='People that more appeared', barmode='stack')



fig.show()
data = df[['year', 'full_name', 'won']].groupby(['year', 'full_name'], as_index=False).count().sort_values(by='year', ascending=False)

people_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(people_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['full_name'], textposition='auto')])

fig.update_layout(title='People by year - Nominees')

fig.show()
data = df[['year', 'full_name', 'won']].loc[df.won].groupby(['year', 'full_name'], as_index=False).count().sort_values(by='year', ascending=False)

people_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(people_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['full_name'], textposition='auto')])

fig.update_layout(title='Shows by year - Winners')

fig.show()
data = df[['year', 'full_name', 'won']].loc[df.won == False].groupby(['year', 'full_name'], as_index=False).count().sort_values(by='year', ascending=False)

people_year = [data.loc[data.year == year].max() for year in years]



temp = pd.DataFrame(people_year)

temp = temp.loc[temp.year.notnull()]



fig = go.Figure(data=[go.Bar(x=temp['year'], y=temp['won'], text=temp['full_name'], textposition='auto')])

fig.update_layout(title='Shows by year - Nominations')

fig.show()
data = df.groupby('category', as_index=False).count().sort_values(by='won', ascending=False)[:10]



fig = go.Figure()



y = [df['won'].loc[(df['category'] == show[1]) & (df.won)].count() for show in data['category'].items()]

fig.add_trace(go.Bar(x=data['category'], y=y, text=y, textposition='auto', name='# won'))



y = [df['won'].loc[(df['category'] == show[1]) & (df.won == False)].count() for show in data['category'].items()]

fig.add_trace(go.Bar(x=data['category'], y=y, text=y, textposition='auto', name='# nominee'))



fig.update_layout(title='Category that more appeared', barmode='stack')



fig.show()
temp = df[['year', 'category', 'won']].copy()



temp.loc[df.category.str.contains('MALE ACTOR'), 'category'] = 'Actor'

temp.loc[df.category.str.contains('FEMALE ACTOR'), 'category'] = 'Actress'

temp = temp.loc[(temp.category == 'Actor') | (temp.category == 'Actress')]



data = temp.groupby(['category'], as_index=False).count()

fig = go.Figure(go.Pie(labels=data['category'], values=data['won'], textinfo='label+percent'))

fig.update_layout(title='Actor <i>vs</i> Actress')

fig.show()