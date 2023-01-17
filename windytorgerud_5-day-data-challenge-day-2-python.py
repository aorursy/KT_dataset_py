# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns
cereal = pd.read_csv(filepath_or_buffer = '../input/cereal.csv')
cereal.describe()



# ... how do you have a negative for sugars, potass or carbo? is -1 what they're using 

# if they dont have an entry, aka NaN?
cereal.head()
cereal.tail()
# lets take a look at distribution of sugars across cereals. plot a histogram of sugars column

sns.distplot(cereal["sugars"], kde = False).set_title("Histogram of Sugar content in Cereals")
# lets take a look at distribution of peoples rating across cereals. plot a histogram of rating column

sns.distplot(cereal["rating"], kde = False).set_title("Histogram of Ratings of Cereals")
# What are the different manufacturers present in the data?

cereal.mfr.unique()
# get each manufacturers ratings, separated

nRating = cereal.rating[cereal.mfr=="N"]

aRating = cereal.rating[cereal.mfr=="A"]

kRating = cereal.rating[cereal.mfr=="K"]

gRating = cereal.rating[cereal.mfr=="G"]

pRating = cereal.rating[cereal.mfr=="P"]

qRating = cereal.rating[cereal.mfr=="Q"]

rRating = cereal.rating[cereal.mfr=="R"]
# sanity check to see if this worked as expected..  

kRating.head()
# Copy the manufacturer names from the data info

# A = American Home Food Products;

# G = General Mills

# K = Kelloggs

# N = Nabisco

# P = Post

# Q = Quaker Oats

# R = Ralston Purina



# lets take a look at distribution of peoples rating across cereal brands. plot histograms

sns.distplot(kRating, kde = False, label = "Kelloggs").set_title("Histogram of Ratings of Cereals by Company")

sns.distplot(gRating, kde = False, label = "General Mills")

sns.distplot(pRating, kde = False, label = "Post")

sns.distplot(qRating, kde = False, label = "Quaker Oats")

sns.distplot(aRating, kde = False, label = "American Home Food Products")

sns.distplot(nRating, kde = False, label = "Nabisco")

sns.distplot(rRating, kde = False, label = "Ralston Purina").legend()

# what is that highly rated Kelloggs cereal?

cereal.name[cereal.rating>80]
# what are the other highly rated ones?

cereal.name[cereal.rating>60]