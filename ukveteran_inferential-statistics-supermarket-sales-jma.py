# importing all the basic libraries



# for using division module

from __future__ import division



# for basic operations

import pandas as pd

import numpy as np



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for avoiding warnings

import warnings

warnings.filterwarnings('ignore')
# reading the data

df = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')



# lets check the shape of the dataset

df.shape
# lets check the head of the dataset

pd.set_option('max_columns', 82)

df.head()
df.columns
df['Customer type'].value_counts()
plt.rcParams['figure.figsize'] = (11, 4)

plt.style.use('fivethirtyeight')



plt.xticks(rotation=30)

sns.distplot(df['Unit price'])

plt.title('Distribution of Target Column')

plt.show()
# lets take seed so that everytime the random values come out to be constant

np.random.seed(6)



# lets take 500 sample values from the dataset of 1460 values

sample_ages = np.random.choice(a= df['Unit price'], size=500)



# getting the sample mean

print ("Sample mean:", sample_ages.mean() )          



# getting the population mean

print("Population mean:", df['Unit price'].mean())
# lets import the scipy package

import scipy.stats as stats

import math



# lets seed the random values

np.random.seed(10)



# lets take a sample size

sample_size = 1000

sample = np.random.choice(a= df['Unit price'],

                          size = sample_size)

sample_mean = sample.mean()



# Get the z-critical value*

z_critical = stats.norm.ppf(q = 0.95)  



 # Check the z-critical value  

print("z-critical value: ",z_critical)                                



# Get the population standard deviation

pop_stdev = df['Unit price'].std()  



# checking the margin of error

margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 



# defining our confidence interval

confidence_interval = (sample_mean - margin_of_error,

                       sample_mean + margin_of_error)  



# lets print the results

print("Confidence interval:",end=" ")

print(confidence_interval)

print("True mean: {}".format(df['Unit price'].mean()))
np.random.seed(12)



sample_size = 500



intervals = []

sample_means = []



for sample in range(25):

    sample = np.random.choice(a= df['Unit price'], size = sample_size)

    sample_mean = sample.mean()

    sample_means.append(sample_mean)



     # Get the z-critical value* 

    z_critical = stats.norm.ppf(q = 0.975)         



    # Get the population standard deviation

    pop_stdev = df['Unit price'].std()  



    stats.norm.ppf(q = 0.025)



    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))



    confidence_interval = (sample_mean - margin_of_error,

                           sample_mean + margin_of_error)  

    

    intervals.append(confidence_interval)

    



plt.figure(figsize=(13, 9))



plt.errorbar(x=np.arange(0.1, 25, 1), 

             y=sample_means, 

             yerr=[(top-bot)/2 for top,bot in intervals],

             fmt='o')



plt.hlines(xmin=0, xmax=25,

           y=df['Unit price'].mean(), 

           linewidth=2.0,

           color="red")

plt.title('Confidence Intervals for 25 Trials', fontsize = 20)

plt.show()
import scipy.stats as sp

def compute_freq_chi2(x,y):

    freqtab = pd.crosstab(x,y)

    print("Frequency table")

    print("============================")

    print(freqtab)

    print("============================")

    chi2, pval, dof, expected = sp.chi2_contingency(freqtab)

    print("ChiSquare test statistic: ",chi2)

    print("p-value: ",pval)

    return





price = pd.qcut(df['Unit price'], 3, labels = ['High', 'Medium', 'Low'])

compute_freq_chi2(df.Gender, price)
import scipy.stats as sp

def compute_freq_chi2(x,y):

    freqtab = pd.crosstab(x,y)

    print("Frequency table")

    print("============================")

    print(freqtab)

    print("============================")

    chi2, pval, dof, expected = sp.chi2_contingency(freqtab)

    print("ChiSquare test statistic: ",chi2)

    print("p-value: ",pval)

    return





price = pd.qcut(df['Unit price'], 3, labels = ['High', 'Medium', 'Low'])

compute_freq_chi2(df.City, price)