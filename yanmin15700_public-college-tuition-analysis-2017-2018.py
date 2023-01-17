# <1> 

# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
# <2>

# read the data - only read the needed columns: CONTROL, TUITIONFEE_IN

column_list= ['CONTROL', 'TUITIONFEE_IN']

data = pd.read_csv("../input/MERGED2017_18_PP.csv", usecols=column_list)

data.info()
# <3>

# Filter the data so you have only public institutions

df = data[data['CONTROL']== 1]

print(df.head(20))
df.info()
df.describe()
# <3>

# Drop the colleges that either have zero tuition or have missing tuition. The remaining colleges constitute the population.



df = df.dropna()

df.info()
# <4> 

# Get a random sample of 100 colleges from the population

# 1st method

pop = df.sample(100)

pop
sample_mean = pop["TUITIONFEE_IN"].mean()

sample_std = pop['TUITIONFEE_IN'].std()

print(sample_mean)

print(round(sample_std,2))
# <4>

# Random sample of 100 colleges

# method 2

SAMPLE_SIZE = 100         # This variable will be used through out the rest of cells



df_sample = df.sample(SAMPLE_SIZE)

sample_mean = df_sample["TUITIONFEE_IN"].mean()

sample_mean                
# <5> Calculate the sample mean and sample standard error

sample_std = df_sample["TUITIONFEE_IN"].std()

round(sample_std, 2)
# <6>

# Calculate the confidence interval of the mean estimate at 68%, 95%. and 99.7%

import math

std_err = sample_std / math.sqrt(SAMPLE_SIZE)       # standard error

std_err
# Calculate 68% Confidence Interval (CI) - one standard error from the population mean

# 68% chances the population mean is within the sample_mean (+ or -) the standard error (SE)

LCL_68 = sample_mean -  std_err

UCL_68 = sample_mean +  std_err



print("Lower confidence limit at 68% confidence level = ", round(LCL_68,2))

print("Upper confidence limit at 68% confidence level = ", round(UCL_68,2))
# Calculate 95% Confidence Interval (CI) - one standard error from the population mean

# 95% chances the population mean is within the sample_mean + or - 2 * the standard error (SE)

LCL_95 = sample_mean -  2 * std_err

UCL_95 = sample_mean +  2 * std_err

print("Lower confidence limit at 95% confidence level = ", round(LCL_95,2))

print("Upper confidence limit at 95% confidence level = ", round(UCL_95,2))
# Calculate 99.7% Confidence Interval (CI) - one standard error from the population mean

# 99.7% chances the population mean is within the sample_mean + or - 3 * the standard error (SE)



LCL_997 = sample_mean -  3 * std_err

UCL_997 = sample_mean +  3 * std_err

print("Lower confidence limit at 99.7% confidence level = ", round(LCL_997,2))

print("Upper confidence limit at 99.7% confidence level = ", round(UCL_997,2))
# <7>

# Calculate the population mean

mean = df['TUITIONFEE_IN'].mean()

round(mean,2)
# Compare the population mean with the sample mean - display the difference

mean_dif = mean - sample_mean

round(mean_dif,2)
# Check the confidence intervals and determine if the population mean within the confidence intervals calculated above.
