import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#!pip install xlrd
salary=pd.read_excel("/kaggle/input/normal-distribution/Employeesalarydata.xlsx")
salary.head()
# find mean median mode - all are equal for normal distribution 
mean = np.average(salary["Emplyees' salry in rupees"])
mean
## find standard deviation
SD = np.std(salary["Emplyees' salry in rupees"])
SD
# What is the probability that the salary of a randomly selected employee is in the range of ₹18,976 to ₹20,989?
LCV = mean - 18976
UCV = mean - 20989
UCV,LCV  #If we devide these values by standard deviation.will come close to 1 that means variable lies in 68% range from mean.
# find z-score value : What is the Z-score for the salary of ₹18,000?
(18000-mean)/SD
## What is the Z-score for the IQ of 120?
## the average IQ of humans is 100 with a standard deviation of 15. 
(120-100)/15