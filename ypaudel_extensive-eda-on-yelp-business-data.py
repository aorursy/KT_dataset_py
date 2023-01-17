#First we will load the necessary libraries and classes

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import json

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline  

%pylab inline

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
biz_f = open('../input/yelp_academic_dataset_business.json',encoding="utf8")

biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])

biz_f.close()

biz_df.info()
# We should have a feel for how the data frame itself looks like before doing anything else on it

biz_df.head(5)
#Check the missing values for each column

biz_df.isnull().sum()
biz_df[biz_df['categories'].isnull()].head()
#Total number of reviews associated with nan values in categories

biz_df[biz_df['categories'].isnull()]['review_count'].sum()
biz_df.dropna(subset=['categories'],inplace=True)
#Total number of unique business categories in the dataset

biz_df['categories'].nunique()
#Plotting top 25 most reviewed businesses among all categories

ax = sns.catplot(x="review_count", y="name",data= biz_df.nlargest(20,'review_count'), 

                 kind="bar",hue= "categories", dodge= False, height= 10 )



plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Top 25 Most Reviewed Businesses And Categories Lables Used') # can also get the figure from plt.gcf()
#Trim any leading white-space. This takes care of cases where some items were not being sorted properly due to whitespaces.

biz_df['categories'].str.strip()



#Create an sorted string, does not create list, and can use .nunique() function if it's not list

#biz_df['categories'] = biz_df['categories'].apply(lambda x: ', '.join(sorted(x.split(', '))))



#Following turns into lists, .nunique() does not work on lists, so need to count list length of the items (!!USE THIS!!)

biz_df['categories'] = biz_df.categories.map(lambda x: [i.strip() for i in sorted(x.split(", "))])



#biz_df['categories'].nunique()

biz_df['categories']



#Count of unique combinations of business categories after alphabetically ordering the labels

print (biz_df['categories'].apply(tuple).nunique())
#Add a new column to count the number of category keywords used

biz_df['Num_Keywords'] = biz_df['categories'].str.len()



#Top 20 categories with most keyword

biz_df[['categories','Num_Keywords']].sort_values('Num_Keywords',ascending = False).head(10)
fig = plt.figure()

ax = fig.add_subplot(111)



x = biz_df['Num_Keywords']

numBins = 100

ax.hist(x,numBins,color='green',alpha=0.7)

plt.show()
#Populating the number of times each combination of unique category combinations found used in the dataset.

df_biz_CountBizPerCat = pd.DataFrame(biz_df.groupby(biz_df['categories'].map(tuple))['Num_Keywords'].count())



#Looking at 'n' category combinations

n = 10

df_biz_CountBizPerCat.sort_values(['Num_Keywords'], ascending = 1).head(n)
#New lets look at the distribution of review counts across major categories

df_BusinessesPerCategories_pre = pd.DataFrame(biz_df.groupby(biz_df['categories'].map(tuple))['review_count'].sum())

df_BusinessesPerCategories_pre.reset_index(level=0, inplace=True) #reset index to column

df_BusinessesPerCategories_pre['Cum_review_count'] = df_BusinessesPerCategories_pre['review_count'].cumsum(axis = 0)

df_BusinessesPerCategories_pre['Percent'] = (df_BusinessesPerCategories_pre['review_count']/df_BusinessesPerCategories_pre['review_count'].sum())*100.00

df_BusinessesPerCategories_pre = df_BusinessesPerCategories_pre.sort_values(['Percent'], ascending = 0)



df_BusinessesPerCategories_pre['Cum_Percent'] = df_BusinessesPerCategories_pre['Percent'].cumsum(axis = 0)

df_BusinessesPerCategories_pre = df_BusinessesPerCategories_pre.sort_values(['Percent'], ascending = 0)



df_BusinessesPerCategories_pre.head(10)
#What are the top-10 business categories before we remove the businesses using more than 3 top-level labels

ax = sns.catplot(x="Percent", y="categories",kind="bar",data=df_BusinessesPerCategories_pre.head(10), aspect= 1.5)

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Top 10 Business Categories by Total Review Counts (Before)') # can also get the figure from plt.gcf()

#List of top-level business categories as foudn in Yelp website

major_categories = ['Active Life', 'Arts & Entertainment', 'Automotive', 'Beauty & Spas', 'Education', 'Event Planning & Services', 'Financial Services', 'Food','Health & Medical',

                    'Home Services', 'Hotels & Travel', 'Local Flavor', 'Local Services', 'Mass Media', 'Nightlife', 'Pets', 'Professional Services', 'Public Services & Government', 

                    'Real estate','Religious Organizations','Restaurants', 'Shopping']
#Creating two empty columns for major category keywords and the counts of such use for each business and categories they have used

biz_df['Count_MajorCategories'] = NaN

biz_df['MajorCategories'] = NaN



#Populating the values in the new columns for each business

biz_df['Count_MajorCategories'] = biz_df['categories'].apply(lambda x: len([value for value in major_categories if value in x]))

biz_df['MajorCategories'] = biz_df['categories'].apply(lambda x: [str(value) for value in major_categories if str(value) in x])



##Printing count of businesses with exactly one top-level category

print(len(biz_df[biz_df.Count_MajorCategories <= 3].sort_values('Count_MajorCategories'))/biz_df['business_id'].count()*100)



##Printing count of businesses with more than one to-level category

print(len(biz_df[biz_df.Count_MajorCategories > 3].sort_values('Count_MajorCategories'))/biz_df['business_id'].count()*100)
print(biz_df['Count_MajorCategories'].value_counts()/biz_df['business_id'].count()*100)



#Count of unique combinations of top-level business categories

print (biz_df['MajorCategories'].apply(tuple).nunique())



#Count of unique combinations of top-level business categories for businesses with no more than 3 such labels used

print (biz_df[biz_df.Count_MajorCategories > 3]['MajorCategories'].apply(tuple).nunique())
#New lets look at the distribution of review counts across major categories

# df_BusinessesPerCategories_major = biz_df.groupby(['MajorCategories'])['review_count'].sum().reset_index().sort_values('review_count',ascending = False)

# df_BusinessesPerCategories_major['Percent'] = (df_BusinessesPerCategories_major['review_count']/df_BusinessesPerCategories_major['review_count'].sum())*100.00

# df_BusinessesPerCategories_major.head()



#New lets look at the distribution of review counts across major categories

df_BusinessesPerCategories_major = pd.DataFrame(biz_df.groupby(biz_df['MajorCategories'].map(tuple))['review_count'].sum())

df_BusinessesPerCategories_major.reset_index(level=0, inplace=True) #reset index to column

df_BusinessesPerCategories_major['Cum_review_count'] = df_BusinessesPerCategories_major['review_count'].cumsum(axis = 0)

df_BusinessesPerCategories_major['Percent'] = (df_BusinessesPerCategories_major['review_count']/df_BusinessesPerCategories_major['review_count'].sum())*100.00

df_BusinessesPerCategories_major = df_BusinessesPerCategories_major.sort_values(['Percent'], ascending = 0)



df_BusinessesPerCategories_major['Cum_Percent'] = df_BusinessesPerCategories_major['Percent'].cumsum(axis = 0)

df_BusinessesPerCategories_major = df_BusinessesPerCategories_major.sort_values(['Percent'], ascending = 0)



df_BusinessesPerCategories_major.head(10)
#Converting the list back to string because some operations and plots are not easy to deal with list types.

biz_df['MajorCategories'] = biz_df.MajorCategories.apply(', '.join)

pd.DataFrame(biz_df['MajorCategories'].unique()).head(10)
# #Plotting top 25 most reviewed businesses among all categories

ax = sns.catplot(x="review_count", y="name",data= biz_df.nlargest(25,'review_count'), 

                  kind="bar", hue = 'MajorCategories', dodge= False, height= 10 )



plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Top 25 Most Reviewed Businesses And Categories Lables Used') # can also get the figure from plt.gcf()
#Drop the businesses that use more than three top-level business categories

to_drop = list(biz_df.query("Count_MajorCategories > 3")['MajorCategories'])

to_drop

biz_df = biz_df[~biz_df.MajorCategories.isin(to_drop)]

biz_df.head(3)
print (biz_df['MajorCategories'].nunique())
#Following turns into lists, .nunique() does not work on lists, so need to count list length of the items (!!USE THIS!!)

#biz_df['MajorCategories'] = biz_df.MajorCategories.map(lambda x: [i.strip() for i in sorted(x.split(", "))])
biz_df.head(5)
#Finding the distrubution of the counts of businesses per category

df_BusinessesPerCategories = pd.DataFrame(biz_df['MajorCategories'].value_counts())

df_BusinessesPerCategories.reset_index(level=0, inplace=True) #reset index to column

df_BusinessesPerCategories.rename(columns={'MajorCategories':'Count_Businesses','index':'MajorCategories'}, inplace=True) #Renaming columns

df_BusinessesPerCategories['Cum_Count_Businesses'] = df_BusinessesPerCategories['Count_Businesses'].cumsum(axis = 0)



df_BusinessesPerCategories['Percent_Busnisses'] = (df_BusinessesPerCategories['Count_Businesses']/df_BusinessesPerCategories['Count_Businesses'].sum())*100.00

df_BusinessesPerCategories = df_BusinessesPerCategories.sort_values(['Percent_Busnisses'], ascending = 0)

df_BusinessesPerCategories = df_BusinessesPerCategories.reset_index(drop=True)



df_BusinessesPerCategories['Cum_Percent_Busnisses'] = df_BusinessesPerCategories['Percent_Busnisses'].cumsum(axis = 0)

df_BusinessesPerCategories = df_BusinessesPerCategories.sort_values(['Percent_Busnisses'], ascending = 0)



#df_categoriesDist['Cum_'] = df_categoriesDist['MajorCategories'].cumsum(axis = 0)

df_BusinessesPerCategories.head(10)
#Finding the distribution of review counts per category

df_ReviewsPerCategories = pd.DataFrame(biz_df.groupby(['MajorCategories'])['review_count'].sum())

df_ReviewsPerCategories.reset_index(level=0, inplace=True) #reset index to column

df_ReviewsPerCategories['Cum_review_count'] = df_ReviewsPerCategories['review_count'].cumsum(axis = 0)



df_ReviewsPerCategories['Percent_reviews'] = (df_ReviewsPerCategories['review_count']/df_ReviewsPerCategories['review_count'].sum())*100.00

df_ReviewsPerCategories = df_ReviewsPerCategories.sort_values(['Percent_reviews'], ascending = 0)

df_ReviewsPerCategories = df_ReviewsPerCategories.reset_index(drop=True)



df_ReviewsPerCategories['Cum_Percent_reviews'] = df_ReviewsPerCategories['Percent_reviews'].cumsum(axis = 0)

#df_ReviewsPerCategories = df_ReviewsPerCategories.sort_values(['Percent'], ascending = 0)



#df_categoriesDist['Cum_'] = df_categoriesDist['MajorCategories'].cumsum(axis = 0)

df_ReviewsPerCategories.head(10)
df_BusinessesPerCategories= df_BusinessesPerCategories.head(50)

df_ReviewsPerCategories = df_ReviewsPerCategories.head(50)
#Joining the reviews counts and business counts dataframe per business category

df_CtReviewsAndBiz_PerCat = df_ReviewsPerCategories.merge(df_BusinessesPerCategories, on='MajorCategories', how='inner')



#Adding weighted column Ct_ReviewPerBiz

df_CtReviewsAndBiz_PerCat['Weight'] = df_CtReviewsAndBiz_PerCat['Percent_reviews']*df_CtReviewsAndBiz_PerCat['Percent_Busnisses']



#Lets also add the differences in percent of business vs. percent of review and sort the df with this new column to help with vizualization later on

df_CtReviewsAndBiz_PerCat['Percent_Diff'] = df_CtReviewsAndBiz_PerCat['Percent_reviews']-df_CtReviewsAndBiz_PerCat['Percent_Busnisses']

df_CtReviewsAndBiz_PerCat = df_CtReviewsAndBiz_PerCat.sort_values(['Percent_Diff'], ascending = 0)



df_CtReviewsAndBiz_PerCat = df_CtReviewsAndBiz_PerCat.reset_index(drop=True)

df_CtReviewsAndBiz_PerCat.head(5)
import seaborn as sns

sns.set(style="white")



# Load the example mpg dataset

plt_data = df_CtReviewsAndBiz_PerCat.head(50)



# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="Percent_reviews", y="Percent_Busnisses", hue="MajorCategories", size="Weight",

            sizes=(100, 1000), alpha=0.7, palette="muted", height=10, data=plt_data)
plt_data = df_CtReviewsAndBiz_PerCat.head(50)



plt_data = plt_data[['MajorCategories','Percent_reviews','Percent_Busnisses']]

plt_data = plt_data.melt('MajorCategories', var_name='Group', value_name='Percent')

plt_data.head(5)
sns.set(style="whitegrid")



# Load the example Titanic dataset

titanic = sns.load_dataset("titanic")



# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Percent", y="MajorCategories", hue="Group", data=plt_data,

                height=12, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("Business Categories")
#Next lets also look at the reviews distribution by cities

df_reviewsPerCity = pd.DataFrame(biz_df.groupby(['city'])['review_count'].sum().sort_values(ascending=False))

df_reviewsPerCity.reset_index(level=0, inplace=True) #reset index to column

df_reviewsPerCity['Cum_review_count'] = df_reviewsPerCity['review_count'].cumsum(axis = 0)



df_reviewsPerCity['Percent_reviews'] = (df_reviewsPerCity['review_count']/df_reviewsPerCity['review_count'].sum())*100.00

df_reviewsPerCity = df_reviewsPerCity.sort_values(['Percent_reviews'], ascending = 0)

df_reviewsPerCity = df_reviewsPerCity.reset_index(drop=True)



df_reviewsPerCity['Cum_Percent_reviews'] = df_reviewsPerCity['Percent_reviews'].cumsum(axis = 0)



ax = sns.catplot(x="Percent_reviews", y="city",kind="bar",data=df_reviewsPerCity.head(25), height = 7)

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Top 25 Cities by Total Review Counts') # can also get the figure from plt.gcf()



df_reviewsPerCity.head(5)
#Next lets also look at the business distribution by cities

df_bizPerCity = pd.DataFrame(biz_df.groupby(['city'])['business_id'].count().sort_values(ascending=False))

df_bizPerCity.reset_index(level=0, inplace=True) #reset index to column

df_bizPerCity['Cum_Biz_Count'] = df_bizPerCity['business_id'].cumsum(axis = 0)



df_bizPerCity['Percent_Biz'] = (df_bizPerCity['business_id']/df_bizPerCity['business_id'].sum())*100.00

df_bizPerCity = df_bizPerCity.sort_values(['Percent_Biz'], ascending = 0)

df_bizPerCity = df_bizPerCity.reset_index(drop=True)



df_bizPerCity['Cum_Percent_Biz'] = df_bizPerCity['Percent_Biz'].cumsum(axis = 0)



ax = sns.catplot(x="Percent_Biz", y="city",kind="bar",data=df_bizPerCity.head(25), height = 7)

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Top 25 Cities by Total Business Counts') # can also get the figure from plt.gcf()



df_reviewsPerCity.head(5)
#Joining the reviews counts and business counts dataframe per business category

df_CtReviewsAndBiz_PerCity = df_bizPerCity.merge(df_reviewsPerCity, on='city', how='inner')



# #Adding weighted column Ct_ReviewPerBiz

df_CtReviewsAndBiz_PerCity['Weight'] = df_CtReviewsAndBiz_PerCity['Percent_reviews']*df_CtReviewsAndBiz_PerCity['Percent_Biz']



#Lets also add the differences in percent of business vs. percent of review and sort the df with this new column to help with vizualization later on

df_CtReviewsAndBiz_PerCity['Percent_Diff'] = df_CtReviewsAndBiz_PerCity['Percent_reviews']-df_CtReviewsAndBiz_PerCity['Percent_Biz']

df_CtReviewsAndBiz_PerCity = df_CtReviewsAndBiz_PerCity.sort_values(['Percent_Diff'], ascending = 0)



# df_CtReviewsAndBiz_PerCity = df_CtReviewsAndBiz_PerCity.reset_index(drop=True)

df_CtReviewsAndBiz_PerCity.head(10)
import seaborn as sns

sns.set(style="white")



# Load the example mpg dataset

plt_data = df_CtReviewsAndBiz_PerCity.head(50)



# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="Percent_reviews", y="Percent_Biz", hue="city", size="Weight",

            sizes=(100, 1000), alpha=0.7, palette="muted", height=10, data=plt_data)
plt_data = df_CtReviewsAndBiz_PerCity.head(20)



plt_data = plt_data[['city','Percent_reviews','Percent_Biz']]

plt_data = plt_data.melt('city', var_name='Group', value_name='Percent')



sns.set(style="whitegrid")



# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Percent", y="city", hue="Group", data=plt_data,

                aspect=2, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("City")
df_BusinessesPerCategories_perCity = (biz_df.groupby(['city','MajorCategories']).agg({'business_id':'count', 'review_count': 'sum'}).reset_index().rename(columns={'business_id':'biz_count','MajorCategories':'MajorCat'}))
#Order by cities and biz counts to populate ordered cum  counts of biz per city

df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','biz_count'], ascending = [0,0])

df_BusinessesPerCategories_perCity['Cum_biz_ct'] = df_BusinessesPerCategories_perCity.groupby(['city'])['biz_count'].transform(pd.Series.cumsum)



#Order by cities and review counts to populate ordered cum  counts of reviews per city

df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','review_count'], ascending = [0,0])

df_BusinessesPerCategories_perCity['Cum_review_ct'] = df_BusinessesPerCategories_perCity.groupby(['city'])['review_count'].transform(pd.Series.cumsum)



#Populating percent business and cumulitive percent business columns

total = df_BusinessesPerCategories_perCity.groupby('city')['biz_count'].transform('sum')

df_BusinessesPerCategories_perCity['biz_pc'] = (df_BusinessesPerCategories_perCity['biz_count']/total)*100

#Sort the df by percent biz before populating cumulitive biz percent

df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','biz_pc'], ascending = [0,0])

df_BusinessesPerCategories_perCity['Cum_biz_pc'] = df_BusinessesPerCategories_perCity.groupby(['city'])['biz_pc'].transform(pd.Series.cumsum)



#Populating percent reviews and cumulitive reviews  columns

total = df_BusinessesPerCategories_perCity.groupby('city')['review_count'].transform('sum')

df_BusinessesPerCategories_perCity['review_pc'] = (df_BusinessesPerCategories_perCity['review_count']/total)*100

#Sort the df by percent reviews before populating cumulitive reviews percent

df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','review_pc'], ascending = [0,0])

df_BusinessesPerCategories_perCity['Cum_review_pc'] = df_BusinessesPerCategories_perCity.groupby(['city'])['review_pc'].transform(pd.Series.cumsum)
n_top_categories = 10

df_topX_reviewCats_perCity = df_BusinessesPerCategories_perCity.loc[df_BusinessesPerCategories_perCity.

                                                                    groupby('city')['review_pc'].nlargest(n_top_categories)

                                                                    .reset_index()['level_1']]



n_top_cities = 10

cities_to_analyze = df_CtReviewsAndBiz_PerCity.nlargest(n_top_cities, 'review_count')['city']

plt_data = df_topX_reviewCats_perCity[df_topX_reviewCats_perCity.city.isin(cities_to_analyze)]

plt_data = plt_data[['city','MajorCat','biz_pc','review_pc']]



#plt_data.melt('name').sort_values('name')

#plt_data = plt_data[['city','MajorCat','biz_count']]

#plt_data = plt_data.melt('city', var_name='Group', value_name='Percent')

plt_data[plt_data.city == 'Las Vegas'].head(5)
sns.set(style="whitegrid")



# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="review_pc", y="MajorCat", col = "city",col_wrap = 3 ,data=plt_data,

                aspect=0.9,height=5, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("City")
analysis_categories = ['Restaurants','Food,Restaurants','Nightlife,Restaurants','Food,Nightlife,Restaurants']
df_biz_Resturants = biz_df[biz_df.MajorCategories.isin(analysis_categories)]

print(df_biz_Resturants['MajorCategories'].value_counts())

df_biz_Resturants.head(5)
df_Resturants_LV = df_biz_Resturants[df_biz_Resturants['city'] == 'Las Vegas']
sns.regplot(x="latitude", y="longitude",data=df_Resturants_LV,scatter_kws={"color":"darkred","alpha":0.2,"s":3},fit_reg=False)
# Lets also load the reviews data into pandas dataframe and look into the size, and structure of the dataframe, columns and their data types

# review_file = open('../input/yelp_academic_dataset_review.json',encoding="utf8")

# review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(5200000)])

# review_file.close()

# review_df.info()
# review_df['stars'].value_counts()
# # Lets observe the reviews dataframe as well

# review_df.head()
# #Drop the reviews of businesses that do not belong to the categories we are interested in.

# to_keep = list(biz_df['business_id'].unique())

# to_keep

# review_df = review_df[review_df.business_id.isin(to_keep)]

# review_df.info()



# del review_df

# del restaurant_reviews



# del [[review_df,restaurant_reviews]]

# gc.collect()
#join dataframe

restaurant_reviews = biz_df.merge(review_df, on='business_id', how='inner')

restaurant_reviews.info()