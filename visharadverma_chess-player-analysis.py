# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv("/kaggle/input/titled-chess-players-india-july-2020/indian_titled_players_july_2020.csv")

df1.head()
df2 = df1.drop(columns="Fide id")

df2.head()
df3 = df2.copy()



df3['Year_of_birth'] = df3['Year_of_birth'].apply(lambda x:2020-x)

df3.head()
df4 = df3.drop(columns="Federation")

df4.head()
df5 = df4.copy()

df5 = df5.rename({"Year_of_birth": "Age"}, axis=1)

df5.tail()
df5.isnull().sum()
#Function to find the number of zeroes

def zero(series):

    count = 0

    for x in range(len(df5)):

        if (df5[series][x] == 0):

            count += 1

    return count



b = zero('Blitz_rating')

r = zero('Rapid_rating')



print("The no of zero values in Rapid Rating are: %d"%(r))

print("The no of zero values in Blitz Rating are: %d"%(b))
df5["Standard_Rating"].mean()
plt.style.use('ggplot')



#Print indexes in a list

x = [i for i in range(1,425)]



#Print Standard and Blitz values

Standard = df5.iloc[:,4].values

Blitz = df5.iloc[:,6].values



fig1, ax1 = plt.subplots()

fig2, ax2 = plt.subplots()

fig3, ax3 = plt.subplots()



#Function for Rating Values with no zeroes

def zeroes(series):

    L = []

    count = 0

    for value in df5[series]:

        if(value != 0):

            L.append(value)

    return L



#Function for Rating count with no zeroes

def count(series):

    x=0

    count = []

    for value in df5[series]:

        if(value != 0):

            x=x+1;

            count.append(x)

    return count



rapid = zeroes('Rapid_rating')

rapid_count = count('Rapid_rating')



blitz = zeroes("Blitz_rating")

blitz_count = count("Blitz_rating")



ax1.scatter(x, Standard, label='Standard Rating')

ax2.scatter(rapid_count, rapid, label='Rapid Rating', color="green")

ax3.scatter(blitz_count, blitz, label='Blitz Rating', color="Blue")



ax1.set_ylabel("Standard")

ax2.set_ylabel("Rapid")

ax3.set_ylabel("Blitz")



ax1.legend()

ax2.legend()

ax3.legend()



plt.xlabel("Player No")



plt.show()



print("The mean of Standard Rating is: ", round(df5["Standard_Rating"].mean()),0)

print("The mean of Rapid Rating (Excluding 0) is: ",round(sum(rapid)/rapid_count[-1]),0)

print("The mean of Blitz Rating (Excluding 0) is: ",round(sum(blitz)/blitz_count[-1]),0)
def average(series, gender):

    count = 0

    sum = 0

    avglist = []

    for x in range(len(df5)):

        if (df5[series][x] != 0):

            if(df5['Gender'][x]==gender):

                avglist.append(df5[series][x])

    mean = np.mean(avglist)

    return mean
filt1 = df5['Gender'] == 'M'

filt2 = df5['Gender'] == 'F'



#Since Standard Rating has no 0 values, we can directly use the mean method

meanMS = df5['Standard_Rating'].loc[filt1].mean()

meanFS = df5['Standard_Rating'].loc[filt2].mean()



#We use the function for both rapid and blitz rating to remove 0 values and accurately calculate mean.

meanMR = average("Rapid_rating","M")

meanFR = average("Rapid_rating","F")



meanMB = average("Blitz_rating","M")

meanFB = average("Blitz_rating","F")



N = 2

ind = np.arange(N)    

width = 0.15     



#We add and subtract 0.15 to make the bars side by side

p1 = plt.bar(ind-0.15, [meanMS,meanFS], width=0.15, label="Standard")

p2 = plt.bar(ind, [meanMR,meanFR], width=0.15, label="Rapid")

p3 = plt.bar(ind+0.15, [meanMB,meanFB], width=0.15, label="Blitz")



plt.ylabel('Rating')

plt.title("Mean of all Ratings")

plt.xticks(ind, ('Male',"Female"))

plt.yticks(np.arange(0, 2500, 200))

plt.legend(loc="upper center")



plt.tight_layout()



plt.show()



print("For Males, the Average Standard Rating is %d, Rapid Rating is %d, and Blitz Rating is %d" %(meanMS, meanMR, meanMB))

print("For Females, the Average Standard Rating is %d, Rapid Rating is %d, and Blitz Rating is %d" %(meanFS, meanFR, meanFB))
filt1 = df5['Gender'] == 'M'

filt2 = df5['Gender'] == 'F'



#Since Standard Rating has no 0 values, we can directly use the mean method

meanMS = df5['Standard_Rating'].loc[filt1].mean()

meanFS = df5['Standard_Rating'].loc[filt2].mean()



#We use the function for both rapid and blitz rating to remove 0 values and accurately calculate mean.

meanMR = average("Rapid_rating","M")

meanFR = average("Rapid_rating","F")



meanMB = average("Blitz_rating","M")

meanFB = average("Blitz_rating","F")



#Subplots

fig1, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)



ax1.bar(['M','F'],[meanMS,meanFS], width=0.35, label="Standard Rating")

ax2.bar(['M','F'],[meanMR,meanFR], width=0.35, label="Rapid Rating", color="#233342")

ax3.bar(['M','F'],[meanMB,meanFB], width=0.35, label="Blitz Rating", color="#234223")



ax1.set_ylabel("Standard")

ax2.set_ylabel("Rapid")

ax3.set_ylabel("Blitz")



ax2.set_xlabel("Gender")



plt.tight_layout()



plt.show()
#To check for any 0 values

print("The number of 0 values in Age column are: ",zero("Age"))



#No zero values, so we can move forward
df5["Age"].nlargest(10)



#85 is an outlier, hence let us not consider it. Since it is only one value, it wont affect our analysis much if we delete it



df6 = df5.drop(173)

df6.reset_index(drop=True)

df6["Age"].nlargest(10)  





#Now we will use the df6 dataframe for the age analysis
# Function to insert row in the dataframe, as we need consistent indexes. I renamed the record to XXXX and made his age the mean age

def Insert_row(row_number, df, row_value): 

    start_upper = 0

   

    end_upper = row_number 

 

    start_lower = row_number 



    end_lower = df.shape[0] 



    upper_half = [*range(start_upper, end_upper, 1)] 



    lower_half = [*range(start_lower, end_lower, 1)] 



    lower_half = [x.__add__(1) for x in lower_half] 



    index_ = upper_half + lower_half 



    df.index = index_ 



    df.loc[row_number] = row_value 

 

    df = df.sort_index() 



    return df 

   

# Let's create a row which we want to insert 

row_number = 173

row_value = ['XXXX', 30, "M",'IM',2315,0,0] 

  

if row_number > df6.index.max()+1: 

    print("Invalid row number") 

else: 

      

    # Let's call the function and insert the row 

    # at the 173rd position 

    df6 = Insert_row(row_number, df6, row_value) 

   

     

df6.loc[170:180]
plt.style.use('fivethirtyeight')



#To generate subplots

fig1, ax1 = plt.subplots(sharey=True)

fig2, ax2 = plt.subplots(sharey=True)

fig3, ax3 = plt.subplots(sharey=True)



def GenderValues(series,gender):

    g_values = []

    for x in range(len(df6)):

        if(df6['Gender'][x] == gender):

            k = df6[series][x]

            g_values.append(k)

    return g_values



bins = []

x = 0

while x<90:

    bins.append(x)

    x = x + 5



#We convert the values to a List, for easy graphing

ages_no_gender = df6["Age"].values

ages_male = GenderValues("Age","M")

ages_female = GenderValues("Age","F")



#The main graphing code

ax1.hist(ages_no_gender, bins=bins,histtype='bar', label="Ages of all Genders", edgecolor="black")

ax2.hist(ages_male, bins=bins, histtype="bar", label="Ages of Males", edgecolor="Black")

ax3.hist(ages_female, bins=bins, histtype="bar", label="Ages of Females", edgecolor="Black")



#We calculate the mean of the ages, respective of Gender

mean_age = df6["Age"].mean()

mean_age_male = average("Age","M")

mean_age_female = average("Age","F")



#We plot a line to show where the mean is

ax1.axvline(mean_age, color="#c70e24", label='Age Mean of all Genders', linewidth=2)

ax2.axvline(mean_age_male, color="#c70e24", label='Age Mean of Males', linewidth=2)

ax3.axvline(mean_age_female, color="#c70e24", label='Age Mean of all Females', linewidth=2)



#Titles

ax1.set_title('Ages of all Grand-Masters')

ax2.set_title('Ages of all Male Grand-Masters')

ax3.set_title("Ages of all Female Grandmasters")



#Labels for X-Axis

ax1.set_xlabel('Ages')

ax2.set_xlabel('Ages')

ax3.set_xlabel('Ages')



#Labels for Y-Axis

ax1.set_ylabel('Occurence')

ax2.set_ylabel('Occurence')

ax3.set_ylabel('Occurence')



#The Limits for the Y graph

axes = plt.gca()

ax1.set_ylim([0,80])

ax2.set_ylim([0,60])

ax3.set_ylim([0,60])



#Legends

ax1.legend()

ax2.legend()

ax3.legend()



plt.tight_layout()



plt.show()



#For user reference, how much the mean really is

print("The Age Mean for all the grandmasters is: ",round(mean_age,1))

print("The Age Mean for the male grandmasters is: ",round(mean_age_male,0))

print("The Age Mean for the female grandmasters is: ",round(mean_age_female,0))