# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt           #Useful for Visualization
import seaborn as sns                     #Useful for Visualization 
%matplotlib inline                       

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Reading the Input File 
h1b=pd.read_csv("../input/h1b_kaggle.csv")
h1b.head()    #Checking the top 5 Data
h1b.info()          #Checking total entries,all the columns and it's datatype in the dataset.


#We have null values as well in our dataset.So let's check how many non null values we have for each column
h1b[h1b.isnull()==False].count()    #Total nonnull values for each column
#Total Unique Number of Job Titles
h1b['JOB_TITLE'].nunique()      #So we have 287549 unique JOB_TITLES
#Top 10 Job Titles
h1b['JOB_TITLE'].value_counts().head(10)

#Programmer Analyst have most number of H1B applications.

#Bar Plot of top 10 JOB_TITLES
h1b['JOB_TITLE'].value_counts().head(10).plot(kind='bar')
#Same Bar Plot but here have just scaled the y-axis in multiple of 1K
plt.figure(figsize=(12,8))  #Increases the size of the graph.Compare it with the above one which is without figsize  
(h1b['JOB_TITLE'].value_counts()/(1000)).head(10).plot(kind='bar')
plt.ylabel('# Value in thousands')
#Similarly we can plot the graph for EMPLOYER_NAME

#Top 5 Employer 
h1b['EMPLOYER_NAME'].value_counts().head().plot(kind='bar')

#We observe Infosys Limited applies most for h1b
#Simple countplot of whether the H1B applications are for a Full_Time_Position or not.
sns.countplot(x=h1b['FULL_TIME_POSITION']) 

#Most of the H1B applications are for Full_Time_Position
#Creating a New Column State from Column WORKSITE
h1b['State']=h1b['WORKSITE'].apply(lambda x:x.split(',')[1])  
h1b.head()  #Now a new column State has been added in the end
#Top 10 States with most number of H1b Applications
h1b.State.value_counts().head(10)

#California is a state with most number of applications and we know because it is the technology hub of USA
#Now we will focus on the Job_Position 'DATA SCIENTIST'
h1b[h1b['JOB_TITLE']=='DATA SCIENTIST']['JOB_TITLE'].count()  #Total H1B Applications for Data Scientist

#H1B Applications for a Data Scientist split by every Year 
h1b[h1b['JOB_TITLE']=='DATA SCIENTIST'].groupby('YEAR')['JOB_TITLE'].count()  

#We observe the number of applications increasing exponentially each year.The reason for it is the demand for DataScientist is increasing and it is the hot job of 20th century.  
 #Top 10 Companies that apply for H1B application for the Position of  Data Scientist
h1b[h1b['JOB_TITLE']=='DATA SCIENTIST']['EMPLOYER_NAME'].value_counts().head(10)       

#We can also do further analysis by checking for each year.
#Let's Check for only Year 2016 which Employer had maximum applications

#Solving this problem by 2 methods

#                               Method 1:
print(h1b[h1b['YEAR']==2016.0].groupby('EMPLOYER_NAME').size().nlargest(5))          
print('\n')
#                               Method 2:
print(h1b[h1b['YEAR']==2016.0]['EMPLOYER_NAME'].value_counts().head(5))

#Seems Infosys had maximum application for the Year 2016
#Plotting a Bar Graph for the above observation
h1b[h1b['YEAR']==2016.0]['EMPLOYER_NAME'].value_counts().head(5).plot(kind='bar')
#Now we will analyze the mean PREVAILING_WAGE by considering Different columns.

#   1:PREVAILING_WAGE across Different State
h1b.groupby('State')['PREVAILING_WAGE'].mean().nlargest(5)
#These are the 5 States with maximum mean salary.By looking at the values we think something is wrong as the mean can't be so high.
#So we will check for the State South Dakota
#   2:So to dig further of the above problem we will only consider those values which have CASE_STATUS='CERTIFIED'as that is what matters

h1b[(h1b['CASE_STATUS']=='CERTIFIED') & (h1b['State']==' SOUTH DAKOTA')]['PREVAILING_WAGE'].mean()
# Just Check for South Dakota.It's mean Wage for Certified Status is 68348.587529.That means the Wage were very high for few applications as we observed above but then the government didn't certified those applications. 

# Now again we will focus for DATA SCIENTIST

# Mean Salary of all the Wages for a CERTIFIED Data Scientist in all the State
h1b[(h1b['JOB_TITLE']=='DATA SCIENTIST') & (h1b['CASE_STATUS']=='CERTIFIED')].groupby('State')['PREVAILING_WAGE'].mean().nlargest(10)     

#Mean Wage of a Data Scientist with Case_Status Certified is highest in California.
#Mean wage of a Data Scientist with status as certified in California for each Year.
h1b[(h1b['State']==' CALIFORNIA') & (h1b['CASE_STATUS']=='CERTIFIED') & (h1b['JOB_TITLE']=='DATA SCIENTIST')].groupby('YEAR')['PREVAILING_WAGE'].mean()

#The wage is somewhat stable in last couple of years.
