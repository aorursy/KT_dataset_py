import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

app_store_dataset = pd.read_csv('../input/AppleStore.csv').set_index('id')
app_store_description_dataset = pd.read_csv('../input/appleStore_description.csv').set_index('id')

app_store_dataset.columns.values
# Feature engineer:
df = app_store_dataset
df_description = app_store_description_dataset

# Change 1,0 columns into boolean
df['vpp_lic'] = df['vpp_lic'].astype('bool') 

df['cont_rating'].unique()
#array(['4+', '12+', '17+', '9+'], dtype=object)

# Remove the '+' from the rating and change dtype to int
df['cont_rating'] = (df['cont_rating']
                    .apply   (lambda x: x.replace('+',''))
                    .astype  (int))

df['currency'].unique()
# array(['USD'], dtype=object) 

# Currency is the same params to all table - remove this column
df.drop('currency', axis = 1, inplace = True)

# Crate 3 columns with Major version, Minor verion, Patch version
df['ver'].isnull().sum()
# 0 - no null in the columns

# Remove unNeeded string (v3.4 , version3.4 , V3.V4) 
df['ver'] = df['ver'].apply(lambda x: re.findall("([\d.]*\d+)", x)[0])
df['major_version']  = df['ver'].str.split('.').str[0]
df['minor_version']  = df['ver'].str.split('.').str[1]
df['patch_version']  = df['ver'].str.split('.').str[2]

# Remove null (version without minor or patch)
df['major_version'].fillna(value = 0, inplace = True)
df['minor_version'].fillna(value = 0, inplace = True)
df['patch_version'].fillna(value = 0, inplace = True)

# Change all type into int
df['major_version'] = df['major_version'].astype(int)
df['minor_version'] = df['minor_version'].astype(int)
df['patch_version'] = df['patch_version'].astype(int)
# remove the version column (all data in the new culoms)
df.drop('ver',axis = 1,inplace = True)

# add descrition to the main table
df = pd.concat([df, df_description['app_desc']], axis=1)

# add new column with is app is free (price == 0)
df['free_app'] = df['price'] == 0 

# add new column wich is app is paid
df['paid_app'] = df['free_app']== 0

df.head(10)
#most rating apps
df.sort_values('rating_count_tot', ascending = False)['track_name'].head()
plt.figure(figsize=(9,5))
#df=app.iloc[:,[3,5,6,7,8,9]]
sns.heatmap(df.corr(),linewidths=.5,cmap="YlGnBu")
plt.show()
# Find price Exceeding observations:
df.plot.scatter('rating_count_tot' , 'price')
plt.show()

df_numerics = df.select_dtypes(exclude=['object'])
cols = list(df_numerics.columns)
df_copy = df.copy()

df_copy['price_zscore'] =  (df_copy['price'] - df['price'].mean())/df_copy['price'].std(ddof=0)
print("prices zscore")
print(df_copy['price_zscore'].nlargest(5))

# id
# 551215116    51.137358
# 308368164    42.564853
# 849732663    16.847336
# 320279293    12.561083
# 491998279     9.989332
# Name: price_zscore, dtype: float64

# We have 2 row with extremely deviation after the standard deviation 
# remove the out liars 
df = df.drop(df[df['price'] > 200].index)
#How much of which genre in appStore?
df["prime_genre"] = df['prime_genre'].astype('category')
df["prime_genre"].value_counts(ascending = True).plot.barh(figsize=(10, 5), title = "Genre Count")
plt.xlabel('Count')
plt.show()

# What is the mean price for each genre
df.groupby('prime_genre')['price'].mean().sort_values().plot.barh(figsize=(10, 5), title = "Mean price Vs Genre")
plt.xlabel('Mean Price')
plt.show()

# Free app Vs Paid app

free_app_mean = df.groupby('free_app').mean()

# rating count free vs paid
free_app_mean['rating_count_tot'].plot.bar(title='rating count free vs paid')
plt.xlabel('Free app')
plt.ylabel('rating countr')
plt.show()
# paid much more rating.

# size bytes free vs paid
free_app_mean['size_bytes'].plot.bar(title='size bytes free vs paid')
plt.xlabel('Free app')
plt.ylabel('size bytes')
plt.show()
# free app as more bytes size (bigger apps)


#compering for each genre the paid and unpaid apps
df.groupby('prime_genre')[['free_app','paid_app']] \
    .sum().plot(kind='bar', rot=100, figsize=(15, 5), grid =True)
plt.show()

#how many apps are rated with each rating?
df.groupby('user_rating').size().plot.bar()
plt.show()
# the connection between the rating of the app and the size of app
rating_v_size=app_store_dataset.groupby('size_bytes')['user_rating'].mean().hist(bins=10,color='purple')
rating_v_size.set_title('connection between rating and size')
rating_v_size.set_xlabel('user_rating')
rating_v_size.set_ylabel('size_bytes')
import nltk
from nltk.corpus import stopwords

# Word analyiest

stop_words = stopwords.words('english')
stop_words.extend(['the','for','and','of','a','by'])
pat = r'[{}[]^`´-_·@|¿?¡!\"?.!/;:<>’•“”–»%■]'.format('|'.join(stop_words))

def cleanString(series):
    series = (pd.Series    (' '.join(series).lower().split())
             .apply       (lambda x: nltk.word_tokenize(x))
             .apply       (lambda x: ''.join(filter(str.isalpha, x)))
             )
    return (series[~series.isin(stop_words)]
            .value_counts()
            .iloc[1:]
           )

mostRepeatedWordsTitle = cleanString(df['track_name'])[:15]
print(mostRepeatedWordsTitle)
mostRepeatedWordsTitle.plot.bar(title = "Most repeated words Title", rot=100,figsize=(10, 5))
plt.show()


must_used_word = mostRepeatedWordsTitle[:10].index
rating_vs_words = pd.Series()
price_vs_words = pd.Series()

rating_vs_words['rating mean'] = df['rating_count_ver'].mean()
price_vs_words['price mean'] = df['price'].mean()

for word in must_used_word:
    filter_by_word_df = df[df['track_name'].str.lower().str.contains(word)]
    rating_vs_words[word] = filter_by_word_df['rating_count_ver'].mean()
    price_vs_words[word] = filter_by_word_df['price'].mean()

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()

rating_vs_words.plot.bar(title = "Words Vs Rating",
                         ax=axes[0], 
                         rot=100,figsize=(10, 10))
axes[0].axhline(rating_vs_words['rating mean'].max(), color="gray")

price_vs_words.plot.bar(title = "Words Vs Price",
                        ax=axes[1],
                        rot=100,figsize=(10, 10))
axes[1].axhline(price_vs_words['price mean'].max(), color="gray")


plt.show()
# Check correlation between common word and user rating , price , free app

df_copy = df.copy()
colums_names = []
for commonWord in mostRepeatedWordsTitle[:5].index:
    colum = commonWord+"_word"
    df_copy[colum] = df['track_name'].str.lower().str.contains(commonWord)
    colums_names.append(colum)

df_copy.groupby('user_rating')[colums_names].sum().plot(kind='bar',
                                                        rot=100,
                                                        figsize=(20, 10),
                                                        grid =True,
                                                        title = "User Vs key word")
plt.legend(prop={'size': 20})
plt.show()

df_copy.groupby('price')[colums_names].sum().plot(kind='bar',
                                                  rot=100,
                                                  figsize=(20, 10),
                                                  grid =True,
                                                  title = "Price Vs key word")
plt.legend(prop={'size': 20})
plt.xlim(-1,10)
plt.show()

df_copy.groupby('free_app')[colums_names].sum().plot(kind='bar',
                                                        rot=100,
                                                        figsize=(10, 5),
                                                        grid =True,
                                                        title = "Free app Vs key word")
plt.legend(prop={'size': 10})
plt.show()

# age investigation
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# precentage of each age from the total
percantage_age=df.groupby('cont_rating')['cont_rating'].apply (lambda x:x.count()/len(df)).plot.pie(autopct='%.2f%%',counterclock=False, shadow=True)
plt.title('percentage of app ages')
plt.show()
# how mush support languages in any version (show only major)
major_lang = df.groupby('major_version').agg([np.mean,np.size])['lang.num']
major_lang = major_lang[major_lang['size'] >= 10]
major_lang['mean'].plot.bar(color = 'blue' ,alpha=0.5, title = "Major ver count Vs support languages")
major_lang['mean'].plot(color = 'red',figsize=(10, 5))
plt.ylabel('Mean support languages')
plt.show()

# how mush thr rating total in any version (show only major)
major_rating_count_tot = df.groupby('major_version').agg([np.mean,np.size])['rating_count_tot']
major_rating_count_tot = major_rating_count_tot[major_rating_count_tot['size'] >= 10]
major_rating_count_tot['mean'].plot.bar(color = 'blue' ,alpha=0.5, title = "Major ver count Vs rating count")
major_rating_count_tot['mean'].plot(color = 'red',figsize=(10, 5))
plt.ylabel('Mean rating count')
plt.show()
# screens 

from scipy import stats 

df.groupby('ipadSc_urls.num').mean()['rating_count_tot'].plot.bar(title = "Rating vs Snapshot screens")
plt.ylabel('Mean rating')
plt.xlabel('Number of snapshot screens')
plt.show()

ipadSc_rating_count = df.groupby('ipadSc_urls.num').count()['rating_count_tot']
ipadSc_rating_std = df.groupby('ipadSc_urls.num').std()['rating_count_tot']
ipadSc_rating_mean = df.groupby('ipadSc_urls.num').mean()['rating_count_tot']

print('mean rating count by screen count')
print(df.groupby('ipadSc_urls.num').mean()['rating_count_tot'])
print()
print('std rating count by screen count')
print(df.groupby('ipadSc_urls.num').std()['rating_count_tot'])
print()
print('count rating count by screen count')
print(df.groupby('ipadSc_urls.num').count()['rating_count_tot'])
f_val, p_val = stats.f_oneway(ipadSc_rating_count , ipadSc_rating_mean,ipadSc_rating_std)  
print()
print ("One-way ANOVA P =", p_val)
print("If P < 5% (0.05) we can claim with high confidence that the means of the results of all three experiments are very significantly different")
