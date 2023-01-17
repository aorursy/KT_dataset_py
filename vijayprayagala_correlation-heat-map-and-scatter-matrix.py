# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#Load Libraries

import pandas

import numpy

import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix
# Load Data

train = pandas.read_csv("../input/train.csv")
# Descriptive statistics

# shape

print(train.shape)

# types

print(train.dtypes)

# descriptions, change precision to 2 places

pandas.set_option('precision', 2)

print(train.describe(include='all'))

# correlation

#pandas.set_option('max_columns', 4)

#pandas.set_option('expand_frame_repr',False)

#print(train.corr(method='pearson'))
#Visualizing Data

#print(train['price_doc'].head())

# scatter plot matrix

x=train['floor']

y=train['price_doc']

plt.figure(1)

plt.scatter(x,y)

plt.xlabel("Number of Floors")

plt.ylabel("Price Of House")

plt.title("Number Floors VS Price of House")

plt.show()
plt.figure(2)

plt.hist(x,73,range=(0,73),alpha=0.25)

plt.title("Histogram for Number of Floors")

plt.show()
def plot_corr_matrix(data,attr,fig_no):

    correlations=data_basic.corr()

    fig=plt.figure(fig_no)

    ax=fig.add_subplot(111)

    ax.set_title("Correlation Matrix for Specified Attributes")

    ax.set_xticklabels(['']+attr)

    ax.set_yticklabels(['']+attr)

    cax=ax.matshow(correlations,vmax=1,vmin=-1)

    fig.colorbar(cax)

    plt.show()
#Plot Correlation Matrix

attr_basic=['price_doc','full_sq','life_sq','floor','max_floor','material','num_room','state','product_type','sub_area']

data_basic=train.loc[:,attr_basic]

plot_corr_matrix(data_basic,attr_basic,3)
#Plot Scatter Matrix

scatter_matrix(data_basic)

plt.show()
attr_others = ['full_all','male_f','young_*','work_*','ekder_*','build_count_*','x_count_500','x_part_500','_sqm',\

                          'cafe_count_d_price_p','trc_','prom_','green_','metro_','_avto_','mkad_','ttk_','sadovoe_','bulvar_ring_','kremlin_',\

                          'zd_vokzaly_','oil_chemistry_','ts_']

data_others=train.loc[:,attr_others]

plot_corr_matrix(data_others,attr_others,5)