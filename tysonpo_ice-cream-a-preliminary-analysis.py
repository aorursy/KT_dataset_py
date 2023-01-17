import matplotlib.pyplot as plt

import math

import os

import random

import cv2



# code for visualization borrowed from Marília Prata: https://www.kaggle.com/mpwolke/cherry-oops-i-didn-t-it-again

def visualize_images(path, n_images, is_random=True, figsize=(16, 16)):

    plt.figure(figsize=figsize)

    w = int(n_images ** .5)

    h = math.ceil(n_images / w)

    

    all_names = os.listdir(path)

    image_names = all_names[:n_images]   

    if is_random:

        random.seed(0)

        image_names = random.sample(all_names, n_images)

            

    for ind, image_name in enumerate(image_names):

        img = cv2.imread(os.path.join(path, image_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        plt.subplot(h, w, ind + 1)

        plt.imshow(img)

        plt.xticks([])

        plt.yticks([])

    

    plt.show()

visualize_images('../input/ice-cream-dataset/combined/images', 9)
import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

import matplotlib.dates as mdates
prod = pd.read_csv("../input/ice-cream-dataset/bj/products.csv")

rev = pd.read_csv("../input/ice-cream-dataset/bj/reviews.csv")
prod.head()
prod.shape
sns.distplot(prod["rating"], bins=25)

plt.xlim([1,5])

plt.show()
prod["rating"].describe()
prod[["name","rating","rating_count"]].sort_values("rating", ascending=False).head(10)
prod[["name","rating","rating_count"]].sort_values("rating").head(10)
prod[["name","rating","rating_count"]].sort_values("rating_count", ascending=False).head(10)
big_ingred_list = []

for ingred_list in prod["ingredients"]:

    # we can't quite do:  big_ingred_list.extend(ingred_list.split(", "))

    # because there are commas within ingredients i.e.  "LIQUID SUGAR (SUGAR, WATER)" is 1 ingredient

    start = 0

    inside = False

    for i,char in enumerate(ingred_list):

        if char == "(":

            inside = True

        if char == ")":

            inside = False

        if not inside and char == ",":

            big_ingred_list.append(ingred_list[start:i].lstrip())

            start = i+1



ct = Counter(big_ingred_list)

most_common = ct.most_common(30)

most_common
ct.most_common()[-20:]
# Number of unique ingredients

len(ct)
top_flavor_ingreds = prod.sort_values("rating", ascending=False)["ingredients"].head(10)

big_ingred_list2 = []

for ingred_list in top_flavor_ingreds:

    start = 0

    inside = False

    for i,char in enumerate(ingred_list):

        if char == "(":

            inside = True

        if char == ")":

            inside = False

        if not inside and char == ",":

            big_ingred_list2.append(ingred_list[start:i].lstrip())

            start = i+1



ct2 = Counter(big_ingred_list2)

[(ingred,count) for ingred,count in ct2.most_common() if ingred not in [i for i,x in most_common]]
rev.head()
rev.shape
rev["date"] = pd.to_datetime(rev["date"], format="%Y-%m-%d")

mpl_data = mdates.date2num(rev["date"])

plt.hist(mpl_data, bins="auto")

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))

plt.show()
# group by flavor and compute standard deviation of star ratings

std_rev = rev.groupby("key")["stars"].std().sort_values(ascending=False).head(5)

std_rev.name = "stdev_stars"



# check to see how many ratings they have, and the mean rating

pd.concat([std_rev,prod[["key","rating","rating_count"]].set_index("key")], axis=1).head(5)
sum_votes = rev[["helpful_yes","helpful_no"]].sum(axis=1) # sum yes & no votes for each review

has_votes = sum_votes > 0 # reviews with votes

rev["vote_ratio"] = rev.loc[has_votes,"helpful_yes"].div(sum_votes)

sns.distplot(rev["vote_ratio"], bins=15)

plt.xlim([0,1])

plt.show()
print("\n\n".join(rev.sort_values("vote_ratio")["text"].head(5).values))