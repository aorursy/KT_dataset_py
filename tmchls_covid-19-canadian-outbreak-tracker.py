# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing the dataset
p=pd.read_csv('/kaggle/input/uncover/UNCOVER/canadian_outbreak_tracker/canada-cumulative-case-count-by-new-hybrid-regional-health-boundaries.csv')
p.head()#printing first 5 rows
#checking for missinig values
p.isnull().any()
#using missingno library to identify missing values visually 
import missingno as msno
msno.matrix(p)
#determining the datatypes of features
p.info()
p['deaths']=p['deaths'].fillna(0)
p['recovered']=p['recovered'].fillna(0)
p['tests']=np.where(p['tests'].isnull(),p['tests'].mean(),p['tests'])
#checking again
msno.matrix(p)
#display of first 5 rows without null values
p.head()
#finding active cases and storing it in the dataframe 
p['active']=p['casecount']-p['recovered']-p['deaths']
#viewing the changes
p.head()
#removing the unrelevant columns
p.drop(p.columns[[4,30,31,32,33,34,35]],axis=1,inplace=True)
p.head()
#importing visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
#taking all population age groups in separate dataframe for easier implementation
y=p[["pop0to4_2019", "pop5to9_2019","pop10to14_2019","pop15to19_2019",
'pop20to24_2019',
'pop25to29_2019',
'pop30to34_2019',
'pop35to39_2019',
'pop40to44_2019',
'pop45to49_2019',
'pop50to54_2019',
'pop55to59_2019',
'pop60to64_2019',
'pop65to69_2019',
'pop70to74_2019',
'pop75to79_2019','pop80to84_2019','pop85older']]
#plotting each category with respect to deaths occured
plt.figure(figsize=(40,30))
plt.subplots_adjust(hspace=1.0)
j=1
for i in y.columns:
    plt.subplot(4,5,j)
    sns.scatterplot(p['deaths'],p[i],ci=None)
    plt.ylabel(i)
    plt.xlabel('deaths')
    plt.xticks(rotation=90)
    j+=1
plt.suptitle('DEATHS PER AGE POPULATION CATEGORY',fontsize=41)     
#converting datatypes of average age and median age for easier scalability
p['averageage_2019']=p['averageage_2019'].astype('int64')
p['medianage_2019']=p['medianage_2019'].astype('int64')
p.info()
#identify average age of individuals who are dead
plt.figure(figsize=(20,20))
sns.barplot('averageage_2019','deaths',data=p,ci=None)
#identifying median age of individuals who are dead
plt.figure(figsize=(20,20))
sns.barplot('medianage_2019','deaths',data=p,ci=None)
#identifying area having most deaths
plt.figure(figsize=(30,30))
sns.barplot('deaths','engname',hue='province',data=p,ci=None)
#identifying the casecount age
plt.figure(figsize=(30,30))
sns.barplot('averageage_2019','casecount',data=p,ci=None)