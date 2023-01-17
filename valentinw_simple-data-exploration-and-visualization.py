#Package importation

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import squarify 
#Data Importation

train = pd.read_csv('../input/train.tsv',delimiter='\t', dtype={'item_description': str})
train.shape

train.info()

train.isnull().sum()

train.head(5)
i = train.name.value_counts().size

print ('Number of distinct items names : ', i)
nb_cat = train.category_name.value_counts().size

print ('We have', nb_cat, 'different items categories.')
plt.figure(figsize=(17,10))

sns.countplot(y = train.category_name, \

              order = train.category_name.value_counts().iloc[:10].index, \

                                                      orient = 'v')

plt.title('Top 10 categories', fontsize = 25)

plt.ylabel('Category name', fontsize = 20)

plt.xlabel('Number of product in the category', fontsize = 20)
# Size of each category

cats = pd.DataFrame(train.category_name.value_counts())

cats.reset_index(level = 0, inplace=True)

cats = cats.sort_values(by='category_name', ascending = False).head(20)

cats.columns =('category_name', 'size')



# Price by category

group = train.groupby(train.category_name)

mean_price = group.price.mean()

mean_price = pd.DataFrame(mean_price)

mean_price.reset_index(level = 0, inplace=True)



# Merging

cats = pd.merge(cats, mean_price, how='left', on = 'category_name')



# Colors setting

cmap = matplotlib.cm.viridis

mini=min(cats['size'])

maxi=max(cats['size'])

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

colors = [cmap(norm(value)) for value in cats['size']]



# Labels setting

labels = ["%s\n%d items\n Mean price : %d$" % (label) \

          for label in zip(cats['category_name'], cats['size'], cats['price'])]



# Plotting

plt.figure(figsize=(30,20))

plt.rc('font', size=15)

squarify.plot(sizes = cats['size'], label = labels, alpha = .7, color=colors)

plt.axis('off')
group = train.groupby(train.category_name)

mean_price = pd.DataFrame(group.price.mean())

mean_price = mean_price.sort_values(by='price', ascending = False).head(20)

mean_price.reset_index(level = 0, inplace=True)



plt.figure(figsize=(17,20))

sns.barplot(x = 'price', y = 'category_name', data = mean_price, orient = 'h')

plt.title('Top 20 categories with higher mean price', fontsize = 30)

plt.ylabel('Categories', fontsize = 25)

plt.xlabel('Mean price', fontsize = 25)
mean_price_2 = pd.DataFrame(group.price.mean())

mean_price_2.reset_index(level = 0, inplace=True)



plt.figure(figsize =(12,7))

sns.kdeplot(mean_price_2.price, shade = True)

plt.title('Mean price by category distribution', fontsize = 20)

plt.xlabel('Mean price of each category', fontsize = 16)
train['cat1'] = train.category_name.str.extract('([^/]+)/[^/]+/[^/]+')

train['cat2'] = train.category_name.str.extract('([^/]+/[^/]+)/[^/]+')



plt.figure(figsize = (10,12))

train.name.groupby(train.cat1).count().plot(kind = 'pie')

plt.title ('First levels of categories', fontsize = 20)

plt.axis('equal')

plt.ylabel('')
# We stock each variable's repartition into a dictionary

alldf = {}

for col in train.cat1[train.cat1.isnull() == False].unique() :



    temp = train.cat2[train['cat1'] == col]

    temp = pd.DataFrame(temp.value_counts().reset_index())

    

    alldf[col] = temp



# Now we can plot it    

i = 0

fig, axs = plt.subplots(5,2, figsize=(20,20))   

plt.suptitle('Zoom on the second level of categories', fontsize = 40) 



for cat in alldf:

    temp = alldf[cat]

    sns.barplot('cat2', 'index', data = temp, ax = axs.flatten()[i])

    axs.flatten()[i].set_ylabel('')

    axs.flatten()[i].set_xlabel('Frequency')

    i+=1    
plt.figure(figsize=(20,20))

sns.boxplot( x = 'price' , y = 'cat1', data = train, orient = 'h')

plt.title('Prices of the first level of categories', fontsize = 30)

plt.ylabel ('First level categories', fontsize = 20)

plt.xlabel ('Price', fontsize = 20)
#Firstly, we create a ranking of our level 2 categories, by price

level2 =  train.groupby('cat2')

rank_level2 = pd.DataFrame(level2.mean()).sort_values(by='price')



#Then, we stock the top 15 most expensive into a list

top_cat2 = rank_level2.tail(15).reset_index()

top_cat2_list = top_cat2.cat2.unique().tolist()

#We don't only want mean price by category, but all basics statistics, so we need the full series

top_cat2_full = train.loc[train['cat2'].isin(top_cat2_list)]



#We can now plot it !

plt.figure(figsize=(20,20))

sns.boxplot(y ='cat2',x= 'price', data = top_cat2_full, orient = 'h')

plt.title('Top 15 second levels categories with highest prices ', fontsize = 30)

plt.ylabel ('Second level categories', fontsize = 20)

plt.xlabel ('Price', fontsize = 20)
botom_cat2 = rank_level2.head(15).reset_index()

botom_cat2_list = botom_cat2.cat2.unique().tolist()

botom_cat2_full = train.loc[train['cat2'].isin(botom_cat2_list)]



plt.figure(figsize=(20,20))

sns.boxplot(y ='cat2',x= 'price', data = botom_cat2_full, orient = 'h')

plt.title('Top 15 second levels categories with lowest prices ', fontsize = 30)

plt.ylabel ('Second level categories', fontsize = 20)

plt.xlabel ('Price', fontsize = 20)
i = train.brand_name.value_counts().size

print('We have', i, 'different brands.') 
plt.figure(figsize=(17,10))

sns.countplot(y = train.brand_name, \

              order = train.brand_name.value_counts().iloc[:10].index, \

                                                      orient = 'v')

plt.title('Top 10 brands', fontsize = 25)

plt.ylabel('Brand name', fontsize = 20)

plt.xlabel('Number of product of the brand', fontsize = 20)
group = train.groupby (train.brand_name)

ranking = pd.DataFrame(group.price.mean())

ranking.reset_index(level = 0, inplace=True)

ranking = ranking.sort_values(by='price', ascending = False).head(15)



plt.figure(figsize=(14,12))

sns.barplot(x = 'price', y = 'brand_name', data = ranking, orient = 'h')

plt.title('Top 15 most expensive brands', fontsize = 30)

plt.ylabel('Categories', fontsize = 25)

plt.xlabel('Mean price', fontsize = 25)
# Brands sorted by number of item

brands = pd.DataFrame(train.brand_name.value_counts())

brands.reset_index(level = 0, inplace=True)

brands = brands.sort_values(by='brand_name', ascending = False).head(15)

brands.columns = ('brand_name', 'number_of_item')



# Brands by price

group = train.groupby (train.brand_name)

brands_prices = pd.DataFrame(group.price.mean())

brands_prices.reset_index(level = 0, inplace=True)



# Merging

brands = pd.merge(brands, brands_prices, how = 'left', on = 'brand_name')



# Labels setting

labels = ["%s\n%d items\n Mean price : %d$" % (label) \

          for label in zip(brands['brand_name'], brands['number_of_item'], brands['price'])]



# Plotting

plt.figure(figsize=(22,13))

plt.rc('font', size=18)

squarify.plot(sizes = brands['number_of_item'], label = labels, alpha = .7, color=colors)

plt.title('Brands treemap', fontsize = 35)

plt.axis('off')
total = float(len(train.item_condition_id))



plt.figure(figsize=(17,10))

ax = sns.countplot(train.item_condition_id)



plt.title('Repartition of conditions', fontsize = 25)

plt.ylabel('Number of items', fontsize = 20)

plt.xlabel('Item condition ID', fontsize = 20)



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:.2f}%'.format((height/total)*100),

            ha="center") 
pd.options.display.float_format = '{:.2f}'.format

train.price.describe()
i = train.price.quantile(0.99)

print ('The 99th quantile is :', i)
plt.figure(figsize=(17,10))

sns.kdeplot(train.price, shade = True)

plt.title('Simple distribution plot of the price', fontsize =25)
i = train.price[train.price == 0].count()

print (i, 'items have a price of zero.')
price_of_zero = train.loc[train.price == 0]



plt.figure(figsize=(17,10))

sns.countplot(y = price_of_zero.category_name, \

              order = price_of_zero.category_name.value_counts().iloc[:10].index, \

                                                      orient = 'v')

plt.title('Top 10 categories of items with a price of 0', fontsize = 25)

plt.ylabel('Category name',  fontsize = 20)

plt.xlabel('Number of product in the category',  fontsize = 20)
exp = train[train['price'] > 200]

exp.name = exp.name.str.upper()



wc = WordCloud(background_color="white", max_words=5000, 

               stopwords=STOPWORDS, max_font_size= 50)



wc.generate(" ".join(str(s) for s in exp.name.values))



plt.figure(figsize=(20,12))

plt.title('What are the most expensive items', fontsize = 30)

plt.axis('off')

plt.imshow(wc, interpolation='bilinear')
total = float(len(train.shipping))



plt.figure(figsize=(10,7))

ax = sns.countplot(train.shipping)

plt.title('Shipping fee paid by seller (1) or by buyer (0)', fontsize = 25)

plt.ylabel('Number of products', fontsize = 20)

plt.xlabel('')



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:.2f}%'.format((height/total)*100),

            ha="center") 
plt.figure(figsize=(10,10))

sns.boxplot(x=train.shipping, y = train.price, showfliers=False, orient = 'v')

plt.title('Does shipping depend of prices ?', fontsize = 25)

plt.xlabel('Shipping fee paid by seller (1) or by buyer (0)', fontsize = 20)

plt.ylabel('Price without outliers', fontsize = 20)
train['no_descrip'] = 0

train.loc[train.item_description=='No description yet', 'no_descrip'] = 1

i = str(round(train['no_descrip'].value_counts(normalize=True).iloc[1] * 100,2)) + '%'



print(i, 'of the items have no a description.')
train['no_descrip'] = 0

train.loc[train.item_description=='No description yet', 'no_descrip'] = 1

i = str(round(train['no_descrip'].value_counts(normalize=True).iloc[1] * 100,2)) + '%'

print(i, 'of the items have no a description. \n')



i1 = str(round((train.no_descrip[train.price > 100].sum() / len(train.no_descrip))*100,2)) + '%'

i2 = str(round((train.no_descrip[train.price <= 100].sum() / len(train.no_descrip))*100,2)) + '%'



print('While', i2, 'of the items with a price lower than 100$ have no description, \n only',\

      i1, 'of the items with a price higher than 100$ have no description.')
wc = WordCloud(background_color="white", max_words=5000, 

               stopwords=STOPWORDS, max_font_size= 50)



wc.generate(" ".join(str(s) for s in train.item_description.values))



plt.figure(figsize=(20,12))

plt.axis('off')

plt.imshow(wc, interpolation='bilinear')

train['coms_length'] = train['item_description'].str.len()



# Some descriptive statistics

pd.options.display.float_format = '{:.2f}'.format

train['coms_length'].describe()
# The full distribution

plt.figure(figsize=(10,10))

sns.kdeplot(train['coms_length'], shade = True)

plt.title ('Distribution of the description length', fontsize = 20)

plt.xlabel('Description length', fontsize = 12)
plt.figure(figsize=(20,20))

sns.regplot(x ='coms_length',y='price', data = train, scatter_kws={'s':2})

plt.title ('Description length VS price', fontsize = 20)

plt.xlabel('Description length', fontsize = 20)

plt.ylabel('Price', fontsize = 20)