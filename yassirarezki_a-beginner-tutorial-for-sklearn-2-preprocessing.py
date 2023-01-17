import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")
df.head()
df.shape
df.isnull().sum()
df.info()
# We create a frame for the figure. The figsize is only to set the size of the figure.

plt.figure(figsize=(16,9))



# we use the function distplot from the sns library to plot the historgram in the previously specified space.

sns.distplot(df.charges)

print("The mean of the charges is {0:.2f}".format(df.charges.mean()))
print("The standard deviation of the charges is {0:.2f}".format(df.charges.std(),2))
# First, we need to import the scale from preprocessing that also belongs to sklearn

from sklearn.preprocessing import scale

charges_scaled = scale(df.charges)
print("The mean of the charges (scaled) is {0:.2}".format(charges_scaled.mean()))

print("The standard deviation of the charges (scaled) is {0:.2}".format(charges_scaled.std()))
plt.figure(figsize=(16,9))

sns.distplot(charges_scaled)
from sklearn.preprocessing import StandardScaler

# Create a transformer and call it as you like. Here I used my_created_transformer

my_created_transformer = StandardScaler()

# Now we fit our tranformer to our data.

my_created_transformer.fit(df.charges)

# After that we can transform the data

charges_scaled = my_created_transformer.transform(df.charges)

from sklearn.preprocessing import StandardScaler

# Create a transformer and call it as you like. Here I used my_created_transformer

my_created_transformer = StandardScaler()

# Now we fit our tranformer to our data.

my_created_transformer.fit(df.charges.values.reshape(-1, 1))

# After that we can transform the data

charges_scaled = my_created_transformer.transform(df.charges.values.reshape(-1, 1))
print("The mean of the charges (scaled) is {0:.2}".format(charges_scaled.mean()))

print("The standard deviation of the charges (scaled) is {0:.2}".format(charges_scaled.std()))
age_scaled = my_created_transformer.transform(df.age.values.reshape(-1, 1))
# Import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

# Create a new transformer. If you do not set any paramter, the data is scaled to the range [0, 1]. 

# in order to set a range, you need to include: feature_range=(min, max). In the example we use min=3, max=7

my_created_transformer = MinMaxScaler(feature_range=(3, 7))

# Fit the data to scaler

my_created_transformer.fit(df.charges.values.reshape(-1, 1))

#Transform the data

charges_scaled_range = my_created_transformer.transform(df.charges.values.reshape(-1, 1))

plt.figure(figsize=(16,9))

sns.distplot(charges_scaled_range)