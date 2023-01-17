# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rc
from matplotlib.ticker import StrMethodFormatter
%matplotlib inline
from scipy import stats

# Regular expressions
import re

# seaborn : advanced visualization
import seaborn as sns
print('seaborn version\t:',sns.__version__)

loan = pd.read_csv('../input/tvs-loan-default/TVS.csv')
loan.head()
loan_1 = loan.rename(columns={'V1': 'Customer ID',
'V2': 'Customer has bounced in first EMI', 
'V3': 'No of times bounced 12 months',
'V4': 'Maximum MOB',
'V5': 'No of times bounced while repaying the loan',
'V6': 'EMI',
'V7': 'Loan Amount',
'V8': 'Tenure',
'V9': 'Dealer codes from where customer has purchased the Two wheeler',
'V10': 'Product code of Two wheeler', 
'V11': 'No of advance EMI paid',
'V12': 'Rate of interest',
'V13': 'Gender',
'V14': 'Employment type',
'V15': 'Resident type of customer',
'V16': 'Date of birth',
'V17': 'Customer age when loanwas taken',
'V18': 'No of loans',
'V19': 'No of secured loans',
'V20': 'No of unsecured loans',
'V21': 'Max amount sanctioned in the Live loans',
'V22': 'No of new loans in last 3 months',
'V23': 'Total sanctioned amount in the secured Loans which are Live',
'V24': 'Total sanctioned amount in the unsecured Loans which are Live',
'V25': 'Maximum amount sanctioned for any Two wheeler loan',
'V26': 'Time since last Personal loan taken (in months)',
'V27': 'Time since first consumer durables loan taken (in months)',
'V28': 'No of times 30 days past due in last 6 months',
'V29': 'No of times 60 days past due in last 6 months',
'V30': 'No of times 90 days past due in last 3 months',
'V31': 'Tier',
'V32': 'Target variable'})

loan_1.head()
loan_1.columns
not_required_cols = ['Time since first consumer durables loan taken (in months)','Time since last Personal loan taken (in months)']
loan_1.drop(labels = not_required_cols, axis =1, inplace=True)
loan_1.shape
loan_1.Gender.value_counts()
sns.set(style="darkgrid")

fig = plt.figure(figsize=(13,6))
plt.subplot(121)

loan_1["Gender"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["gold","b"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.title("distribution of client owning a car")

plt.subplot(122)
ax = loan_1["Gender"].value_counts().plot(kind="bar",color='b')
plt.xticks(rotation=0)
plt.title("Count of target variable")

plt.show()
sns.set(style="darkgrid")

fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(25,8))
ax[0].set_title("Loan Amount (Distribution Plot)")
sns.distplot(loan_1['Loan Amount'],ax=ax[0])
ax[1].set_title("Loan Amount (Violin Plot)")
sns.violinplot(data =loan_1, x='Loan Amount',ax=ax[1], inner="quartile")
ax[2].set_title("Loan Amount (Box Plot)")
sns.boxplot(data =loan_1, x='Loan Amount',ax=ax[2],orient='v')
plt.show()
sns.set(style="darkgrid")

fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(25,8))
ax[0].set_title("Loan Amount (Distribution Plot)")
sns.distplot(loan_1['Rate of interest'],ax=ax[0])
ax[1].set_title("Loan Amount (Violin Plot)")
sns.violinplot(data =loan_1, x='Rate of interest',ax=ax[1], inner="quartile")
ax[2].set_title("Loan Amount (Box Plot)")
sns.boxplot(data =loan_1, x='Rate of interest',ax=ax[2],orient='v')
plt.show()
plt.figure(figsize=(18,6))

#bar plot
loan_1['Customer age when loanwas taken'].value_counts().plot(kind='bar',color='b',alpha=0.7, edgecolor='black')
plt.xlabel("Age", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title(" Age of Customer when the loan was approved")
plt.legend(loc="best",prop={"size":12})
plt.show()
plt.figure(figsize=(18,6))

#histogarm
sns.distplot(loan_1["Customer age when loanwas taken"],color="b")
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(8,8))
loan_1["Product code of Two wheeler"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",5),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                               shadow =True)
plt.title("Two Wheeler Type")
#MC : Motorcycle , MO : Moped, SC : Scooter
plt.show()
plt.figure(figsize=(24,8))
sns.countplot(x="No of times bounced while repaying the loan", hue="Employment type", data=loan_1, edgecolor='k')
plt.xlim(0, 6)
plt.ylim(0, 16000)
plt.legend(loc='upper right')
plt.show()