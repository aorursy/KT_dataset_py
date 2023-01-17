# Load packages

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as smf

import statsmodels.api as sm

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])

properties = pd.read_csv('../input/properties_2016.csv')
print ("Shape Of Train: ",train.shape)

print ("Shape Of Properties: ",properties.shape)
merged = pd.merge(train,properties,on="parcelid",how="left")
merged.head(3).transpose()
merged = merged[['parcelid','transactiondate','bathroomcnt',

                 'bedroomcnt','calculatedbathnbr','calculatedfinishedsquarefeet',

                 'lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt'

        ]]

merged.head(3)
# Load ggplot package

from ggplot import *
# Relationship between calculatedfinishedsquarefeet and structuretaxvaluedollarcnt for bedroomcnt

p = ggplot(merged,aes(x='structuretaxvaluedollarcnt', y='calculatedfinishedsquarefeet')) + geom_point(size=150, color = 'blue') + stat_smooth(color = 'red', se=False, span=0.2) + facet_grid('bedroomcnt')

p + xlab("structuretaxvaluedollarcnt") + ylab("squarefeet") + ggtitle("House: squarefeet vs structuretaxvaluedollarcnt")
# Correlation heatmap

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("Zillow Price Correlation Heatmap")

corr = merged.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Split dataset into train data and test data

x_train ,x_test = train_test_split(merged,test_size=0.3)

x_train.head()

x_test.head()