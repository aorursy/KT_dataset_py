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
from scipy.stats import ttest_ind
import seaborn as sns

cereal = pd.read_csv(filepath_or_buffer = '../input/cereal.csv')
cereal.describe()
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
# get each manufacturers sugar, separated

nSugars = cereal.sugars[cereal.mfr=="N"]

aSugars = cereal.sugars[cereal.mfr=="A"]

kSugars = cereal.sugars[cereal.mfr=="K"]

gSugars = cereal.sugars[cereal.mfr=="G"]

pSugars = cereal.sugars[cereal.mfr=="P"]

qSugars = cereal.sugars[cereal.mfr=="Q"]

rSugars = cereal.sugars[cereal.mfr=="R"]
# Copy the manufacturer names from the data info

# A = American Home Food Products;

# G = General Mills

# K = Kelloggs

# N = Nabisco

# P = Post

# Q = Quaker Oats

# R = Ralston Purina



# lets take a look at distribution of peoples rating across cereal brands. plot histograms

sns.distplot(kSugars, kde = False, label = "Kelloggs").set_title("Histogram of Sugars of Cereals by Company")

sns.distplot(gSugars, kde = False, label = "General Mills")

sns.distplot(pSugars, kde = False, label = "Post")

sns.distplot(qSugars, kde = False, label = "Quaker Oats")

sns.distplot(aSugars, kde = False, label = "American Home Food Products")

sns.distplot(nSugars, kde = False, label = "Nabisco")

sns.distplot(rSugars, kde = False, label = "Ralston Purina").legend()
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
# do Kellogs and Post have statisically significantly different Sugar levels

# across their various cereals?

ttest_ind(kSugars, pSugars, equal_var=False)
# do Kellogs and Post have statisically significantly different User ratings

# across their various cereals?

ttest_ind(kRating, pRating, equal_var=False)