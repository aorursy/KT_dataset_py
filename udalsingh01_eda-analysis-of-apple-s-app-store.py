import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
app_data = pd.read_csv('../input/AppleStore.csv')
app_desc = pd.read_csv('../input/appleStore_description.csv')
app_data.head()
# app_data['price'].sort_values(ascending=False)
app_data[['track_name','price']].sort_values(by='price',ascending=False)[:2]
app_data['prime_genre'].value_counts().plot(kind='barh', figsize=(15,8),logx=True, title='Most developed app', fontsize=12)
app_data[app_data.prime_genre == 'Games']['cont_rating'].value_counts().plot(kind='pie', figsize=(8,8),fontsize=18)
# 0.0 -> free, 0.99-5.0 -> low, 6.0-20.0 -> medium, 21.0-more ->high 
cat_price = pd.cut(app_data['price'], pd.IntervalIndex.from_tuples([(-1.0,0.0),(0.1,5.0),(5.1,20.0),(20.1,500.0)]), 
                   labels=["Free","Low", "medium", "High"])
to_rep = ['(-1.0, 0.0]','(0.1, 5.0]','(5.1, 20.0]','(20.1, 500.0]']
rep_val = ["Free","Low ($0.1-$5.0)", "Medium ($5.1-$20.0)", "High ($20.1-500.0)"]

s=cat_price.apply(str)

app_data['cat_price'] = s.replace(to_replace=to_rep, value=rep_val)

app_data['cat_price'].value_counts().plot('bar',figsize=(10,8),fontsize=18)

app_data['cat_price'].value_counts()[0]/app_data['cat_price'].count()*100


# popular = Avg_user_rating * Total_ratting (rating_count_tot*user_rating)
# below setting popularity for whole dataset
app_data['popularity'] = app_data['rating_count_tot'] * app_data['user_rating'] 


# Now populatiy for free apps --- Below are the top 10 free Games
pop_free_games = app_data[app_data['prime_genre'] == 'Games'][app_data['cat_price']=='Free'][['track_name','popularity']].sort_values(by='popularity',ascending=False)[:10]


sns.barplot(x=pop_free_games['popularity'], y=pop_free_games['track_name'])


app_data[app_data['cat_price'] == 'Free'][['track_name','popularity']].sort_values(by='popularity',ascending=False)[:10]
app_data[app_data['prime_genre'] != 'Games'][['track_name','popularity']].sort_values(by='popularity',ascending=False)[:10]
app_data_corr = app_data.corr()


plt.figure(figsize=(10,10))
sns.heatmap(app_data_corr)
app_data_corr["rating_count_tot"].sort_values(ascending=False)

app_data.plot(x='lang.num',y='rating_count_tot',kind='scatter',figsize=(10,10))

# revenue = rating_count_total * price
app_data['revenue'] = app_data['rating_count_tot'] * app_data['price']
rev = app_data[['track_name','revenue']].sort_values('revenue',ascending=False)[:10]
plt.barh(rev['track_name'],rev['revenue'], )
