# OVERVIEW



# This exercise is to advance my knowledge of Pandas by evaluating Airbnb Los Angeles listings data 

# for 2018 and 2019. The goal was to practice different methods to explore, clean and reshape the data

# in order to assess trends about year-over-year changes in listings' prices, number of reviews,

# neighborhoods, room types, and affordability. Specifically, I wanted to see if there were any 

# correlations between those parameters, as well as if there were any significant changes within the 

# two-year period.



# Since I'm not familiar with descriptive statistics and regression, I will opt for aggregations, 

# comparisons and data slicing to analyze my datasets.
# DATA PROFILE



# The two datasets I'm working with were sourced from “Inside Airbnb” page, “Get the Data” section: 

# http://insideairbnb.com/get-the-data.html



# Both datasets contain over 40K rows of data populated with numberic and string data. Both datasets

# have exactly the same column structure.



# For data cleaning, I intend to utilize filtering, groupings, data conversions, and sorting.
# ANALYSIS
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df2018 = pd.read_csv("../input/Airbnb_LosAngeles_listings_2018.csv")

df2019 = pd.read_csv("../input/Airbnb_LosAngeles_listings_2019.csv")
import matplotlib.pyplot as plt

import seaborn as sns
# Previewing the dataset



df2019
# Removing everything outside the City of Los Angeles



df2019_revised = df2019[df2019.neighbourhood_group == 'City of Los Angeles']

df2018_revised = df2018[df2018.neighbourhood_group == 'City of Los Angeles']
# Previewing updated dataset, checking the Neighborhood Group



df2019_revised.head(5)
# 1. AVERAGE LISTING PRICE BY NEIGHBORHOOD
# Summarizing counts by neighborhood to determine how many neighborhood we're analyzing in 2019

# Then sorting the counts from highest to lowest



df2019_revised.groupby("neighbourhood").size().sort_values(ascending=False)
# From the summary above, we see that the top 5 neighborhoods by listings in 2019 were Hollywood, 

# Venice, Downtown, Hollywood Hills and Westlake
# Doing the same thing for 2018



df2018_revised.groupby("neighbourhood").size().sort_values(ascending=False)
# From the summary above, we see that the top 5 neighborhoods by listings in 2018 were Hollywood, 

# Venice, Downtown, Hollywood Hills and Westlake - same as in 2019. What did change was the number of

# listings: 2018 had more listings than 2019 in all top 5 neighborhoods and, presumably, other 

# neighborhoods as well.
# Aggregating neighborhoods' prices for 2019

# Displaying aggregared prices as an average to determine an average price per neighborhood

# Sorting average prices from highest to lowest, showing top 10



print("2019 Average Listing Price by Neighborhood")

price_per_neighbourhood_2019 = pd.pivot_table(df2019_revised, values='price', index=['neighbourhood'], aggfunc=np.average)

price_per_neighbourhood_2019.sort_values(by='price',ascending=False).head(10)
# From the summary above, we see that top 5 most expensive neighborhoods in 2019 were Bel Air, Beverly

# Crest, Hollywood Hills West, Century City and Windsor Square
# Doing the same for 2018



print("2018 Average Listing Price by Neighborhood")

price_per_neighbourhood_2018 = pd.pivot_table(df2018_revised, values='price', index=['neighbourhood'], aggfunc=np.average)

price_per_neighbourhood_2018.sort_values(by='price',ascending=False).head(10)
# It appears that while there were fewer listings in 2019, they were more expensive on average than the

# listings in 2018
# To verify this, I want to compare the average listing price in 2018 and 2019



def price_comparison():

    if df2019_revised['price'].mean() > df2018_revised['price'].mean():

        print("Average listing price in 2019 was higher than in 2018.")

    elif df2019_revised['price'].mean() < df2018_revised['price'].mean():

        print("Average listing price in 2019 was lower than in 2018.")

    else:

        print("Average listing price in 2019 was the same as in 2018.")



price_comparison()
# Next, I want to see the maximum and minimal listing prices in 2019



df2019_revised['price'].describe()
# Seeing the same maximum and minimal listing prices in 2018



df2018_revised['price'].describe()
# Based on the minimal, I see that some listings have a $0 price in 2019

# I want to remove them to keep the counts clean



df2019_revised = df2019_revised[df2019.price != 0]

df2019_revised['price'].describe()
# After the revision, the new min listing price in 2019 is $10
# Doing the same for 2018 data



df2018_revised = df2018_revised[df2018.price != 0]

df2018_revised['price'].describe()
# After the revision, the new min listing price in 2018 is $4
# 2. LISTINGS' AFFORDABILITY
# Grouping the prices into categories based on affordability 



def price_ranking(price):

    if price < 200:

        return "Affordable"

    elif price < 1000 and price >= 200:

        return "Mid-range"

    else:

        return "Expensive"
# Counting the prices per each affordability category for 2019 data



price_ranking_2019 = df2019_revised['price'].apply(price_ranking).value_counts()

price_ranking_2019
# Visualizing price affordability distribution for 2019



print("Listing Price Distribution by Affordability (2019)")

price_ranking_2019.plot(kind='bar')
# Counting the prices per each affordability category for 2018 data



price_ranking_2018 = df2018_revised['price'].apply(price_ranking).value_counts()

price_ranking_2018
# Visualizing price affordability distribution for 2018



print("Listing Price Distribution by Affordability (2018)")

price_ranking_2018.plot(kind='bar')
# Next, I want to visualize affordability on the same graph for 2018 and 2019

# I first check the data type for "price_ranking"



type(price_ranking_2019)
# Since it's a series, I need to convert it to DataFrame and check



affordability_2019 = price_ranking_2019.to_frame()

type(affordability_2019)
# Doing the same for 2018



affordability_2018 = price_ranking_2018.to_frame()

type(affordability_2018)
# UNABLE TO MERGE AND SUBSEQUENTLY VISUALIZE ON THE SAME GRAPH



# Issue: the "merge" method doesn't take in converted dataFrame parameters. Nor does the "concat" method.

# Initially, I thought the problem was the data coming from a function "price_ranking", but now I don't

# believe it's the case.
# 3. LISTING PRICE BY ROOM TYPE
# Evaluating the room type for 2019



room_type_2019 = df2019_revised['room_type'].value_counts()

room_type_2019
# Evaluating the room type for 2018



room_type_2018 = df2018_revised['room_type'].value_counts()

room_type_2018
# Checking the data type for the room_type



type(room_type_2018)
# Converting the series into a data frame (2018 and 2019)



listings_by_room_type_2019 = room_type_2019.to_frame()

listings_by_room_type_2018 = room_type_2018.to_frame()
# AGAIN, UNABLE TO MERGE AND SUBSEQUENTLY VISUALIZE ON THE SAME GRAPH



# Data is NOT coming from a function, but the "merge" and "concat" methods didn't work for value counts.
# Assessing average listing price by room type for 2019



print("Average Price by Room Type in 2019")

price_per_room_type_2019 = pd.pivot_table(df2019_revised, values='price', index=['room_type'], aggfunc=np.average).sort_values(by='price',ascending=False)

price_per_room_type_2019
# Per the above, Entire Home/Apt was the most expensive, while a Shared Room was the cheapest on

# average in 2019
# Assessing average listing price by room type for 2018



print("Average Price by Room Type in 2018")

price_per_room_type_2018 = pd.pivot_table(df2018_revised, values='price', index=['room_type'], aggfunc=np.average).sort_values(by='price',ascending=False)

price_per_room_type_2018
# Per the above, Entire House/Apt was the most expensive, while a Shared Room was the cheapest on 

# average in 2018
# To visualize the data, I'm merging two pivot tables into one



price_per_room_type = pd.merge(price_per_room_type_2018, price_per_room_type_2019, how='right', on='room_type')

price_per_room_type
# Converting NaN to a 0 (can't drop it completely without losing the entire row)



price_per_room_type.fillna(0)
# Visualizing the merged table



price_per_room_type = price_per_room_type.rename(columns={'price_x': '2018','price_y': '2019'})

ax = price_per_room_type.plot(kind='bar') 

ax.legend(['Ave price (2018)', 'Ave price (2019)']);

ax.set_xlabel('Room type')

ax.set_ylabel("Average price")
# All individual room type categories had a higher average listing price in 2019 than in 2018. 

# 2018 data didn't track (or had?) a "Hotel room" category.
# 4. PRICE / REVIEWS EVALUATION
# Hypothesis: most people can afford less expensive options. Therefore, there'll be more reviews for 

# lower-priced listings



df2019_revised.plot(kind='scatter', x='number_of_reviews', y='price', alpha=0.5)
# Per the scatterpot above, the hypothesis appears somewhat correct, but I don't believe it's conclusive.
# To test the hypothesis, I'll first sort the price per neighborhood from cheapest to most expensive



price_per_neighbourhood_2018.sort_values(by='price',ascending=True).head(10)
# Since we've seen that price correlates with Neighborhood, I'll check the hypothesis above by evaluating 

# the average number of reviews against the neighboorhoods



reviews_per_neighborhood_2019 = pd.pivot_table(df2019_revised, values='number_of_reviews', index=['neighbourhood'], aggfunc=np.average).sort_values(by='number_of_reviews',ascending=False).head(10)

reviews_per_neighborhood_2019
# Here, the results paint a different picture. The cheapest neighborhoods don't have the highest

# number of reviews. Therefore, it appears that "neighborhood" isn't a driving factor for reviews.
# Evaluating average number of reviews per room type in 2018



reviews_per_room_type_2019 = pd.pivot_table(df2019_revised, values='number_of_reviews', index=['room_type'], aggfunc=np.average).sort_values(by='number_of_reviews',ascending=False)

reviews_per_room_type_2019
# Given that Entire Home/Apt accomodations are the most expensive on average, it's interesting that it 

# had the highest average number of reviews. Maybe people want to reflect if the price was worth it. Or, 

# perhaps, groups of people that tend to rent entire apartments, are more likely to have at least one

# person among them to leave a review. The current datasets don't shed light on the causes.



# It's also interesting how much the number of reviews dropped for Shared room (also the cheapest).

# Maybe people just don't care. 
# Evaluating average number of reviews per room type in 2018



reviews_per_room_type_2018 = pd.pivot_table(df2018_revised, values='number_of_reviews', index=['room_type'], aggfunc=np.average).sort_values(by='number_of_reviews',ascending=False)

reviews_per_room_type_2018
# Same trend for all room types in 2018 as in 2019. Average number of reviews was lower as a whole. 
# To visualize the data, I'm merging two pivot tables into one



reviews_per_room_type = pd.merge(reviews_per_room_type_2018, reviews_per_room_type_2019, how='right', on='room_type')

reviews_per_room_type
# Visualizing the merged table



reviews_per_room_type = reviews_per_room_type.rename(columns={'number_of_reviews_x': '2018','number_of_reviews_y': '2019'})

af = reviews_per_room_type.plot(kind='bar') 

af.legend(['Ave reviews (2018)', 'Ave reviews (2019)']);

af.set_xlabel('Room type')

af.set_ylabel("Average reviews")
# CONCLUSIONS / DIRECTIONS FOR FUTURE WORK



# This exercise left me curious about more efficient ways to process data from multiple sources 

# simultaneously. I'm convinced that my approach was far from elegant and concise.



# I'm left with questions about data type conversions and merging/concatenation convensions for the 

# converted output. There were two areas where I struggled to visualize results because the "merge" method

# wasn't working for the converted data.



# One of the most interesting things I've learned was that styling an element could affect its data type.

# It was counter-intuitive, but fascinating.



# I was excited to incorporate pivot tables and basic chart formatting into my workflow. Those are the

# things I use in my professional life, so it was easy to double-check my results in Excel.



# There are two things I'd love to further explore in the future:



# a) There is a column with the Last Review Date. I'd love to see if seasonality is a factor in both

# the number of reviews and prices.



# b) The datasets contain the hosts' first names. I'd love to investigate if there are any differences 

# based on those names being American vs. non-American (subjectively speaking). For example, if hosts with

# foreign-sounding first names had fewer reviews than hosts with traditional American names. But that

# type of analysis is beyond my current abilities. 