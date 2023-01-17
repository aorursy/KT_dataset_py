# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# This is the data for "County Data"
main_file_path = '../input/acs2015_county_data.csv'
data1 = pd.read_csv(main_file_path)

# These are the columns for the "County Data"
data1.columns
# This is the data for "Census Tract Data"
main_file_path2 = '../input/acs2015_census_tract_data.csv'
data2 = pd.read_csv(main_file_path2)

# These are the columns for the "Census Tract data"
data2.columns
# It looks like the columns for the "County Data" and the "Census Tract data" are pretty similar. What is the difference between them?
# This is a summary of "County Data"
data1.describe()

data2.describe()
data1.head()
data2.head()
data2.shape
#Ok I think I understand, the values in data2 are smaller subsections of data used to create the values in data which is sorted by County, I guess we'll use data1 from now on
data1.shape
#going to drop any rows with Null values, this is to see the before and after so we can see what we're doing
#Great! Doesn't look like we lost too many values!
data1 = data1.dropna()
data1.shape
#Lets see how many of each race is in the US as a percentage of the total 
race_columns = ['White','Black','Native','Asian','Pacific']
census_races =data1[race_columns]

#There are more white people than black people as expected
(census_races.sum()/len(census_races)).plot.bar(title = "Percentage of Americans by Race")
f = ( data1.loc[:, ['Hispanic', 'Black', 'White', 'Asian', 'Native','Pacific','Poverty']]).corr()

sns.heatmap(f, annot=True)
#Although there are many non-black areas stricken with poverty, it is clear to see as the percentage black increases, poverty does too
sns.jointplot(x='Black', y='Poverty', data=data1, kind="reg")
#it's interesting to see that as percent White increases, poverty levels decrease until about 80% white. However, they rise again afterwards. Maybe this is due to rural farm communities?
sns.jointplot(x="White",y="Poverty", data = data1, kind="reg")
#This plot shows the above information, but it really shows how white most Counties in America are.
sns.jointplot(x="White",y="Poverty", data = data1, kind = "hex")
#Asian percentages are also inversely correlated with poverty, however, you can see that the communities that are very strongly asian, there is limited poverty. The correlation may be a little misleading in this case since
#I think the strength of the relationship is more powerful than the value implies.
sns.lmplot(x="Asian", y= "Poverty", data = data1, lowess=True)
#There are two interesting pieces of information. 1: There are areas in the US which are almost 100% hispanic. 2: These areas are very poor.
sns.jointplot(x="Hispanic",y = "Poverty",data = data1, kind="reg")