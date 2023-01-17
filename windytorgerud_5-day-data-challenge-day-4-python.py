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
# What are the different manufacturers present in the data?

cereal.mfr.unique()
# Copy the manufacturer names from the data info

# A = American Home Food Products;

# G = General Mills

# K = Kelloggs

# N = Nabisco

# P = Post

# Q = Quaker Oats

# R = Ralston Purina

sns.countplot(cereal.mfr).set_title("Cereal counts by Manufacturer")



# this doesn't say much about actual cereals out there by company, just how many/company those 

# who curated these data happened to collect.
