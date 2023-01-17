# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/googleplaystore.csv")
data.info()
data.head(10)
data.Category.unique
data.Category.value_counts()


# delete rows if it has a nan value

data.dropna(subset=["Rating"], inplace=True)



# deleting space between 2 words

data = data.rename(columns={'Content Rating': 'ContentRating'})



# deleting a row which has 19.0000 values in Rating column

data = data[data.Rating != 19.0000]

#%% Cleaning datas in Reviwes and in Size



# delete M letter from end of number in Reviews column

data["Reviews"] = [each.replace('M','') if "M" in each else each for each in data.Reviews]



# convert Reviews column to numeric from string

data["Reviews"] = pd.to_numeric(data["Reviews"])



# delete M letter fron end of number in Size column

data["Size"] = [each.replace('M','') if "M" in each else each for each in data.Size]



# replace k with 000 because k means 1000

data["Size"] = [each.replace('k','000') if "k" in each else each for each in data.Size]



# we filter "Varies with device" in Size column

data = data[data.Size != "Varies with device"]



# delete "+" 

data["Size"] = [each.replace('+','') if "+" in each else each for each in data.Size]



# delete ","

data["Size"] = [each.replace(',','') if "," in each else each for each in data.Size]



# convert Size columnm to numeric from string

data["Size"] = pd.to_numeric(data["Size"])



# delete + in Installs column

data["Installs"] = [each.replace('+','') if "+" in each else each for each in data.Installs]



# delete , in Installs column

data["Installs"] = [each.replace(',','') if "," in each else each for each in data.Installs]



# replace Free with 0

data["Installs"] = [each.replace('Free','0') if "Free" in each else each for each in data.Installs]



# convert Installs column to numeric from string

data["Installs"] = pd.to_numeric(data["Installs"])



# deleting $ in Price column

data["Price"] = [each.replace('$','') if "$" in each else each for each in data.Price]



# convert Price column to numeric from string

data["Price"] = pd.to_numeric(data["Price"])
# it looks that there is a colleration between only Installs and Reviews as numerical

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.1f',ax=ax)

plt.show()





# separate data into 5 parts according to Rating

worstRating = data[data['Rating']<2.500]

lowerRating = data[(data['Rating']>=2.500) & (data['Rating']<3.500)]

mediumRating =  data[(data['Rating']>=3.500) & (data['Rating']<4.000)]

goodRating =    data[(data['Rating']>=4.000) & (data['Rating']<4.500)]

veryGoodRating = data[data['Rating']>4.500]

# %% sorting and normalization



# sorting according to value of Reviews

new_index = (worstRating['Reviews'].sort_values(ascending=False)).index.values

sorted_worst = data.reindex(new_index)

# normalization of Reviews column. 11773 is the biggest value of the Reviews column in sorted worst

sorted_worst['Reviews'] = sorted_worst.Reviews/11773



# sorting according to value of Reviews

new_index = (lowerRating['Reviews'].sort_values(ascending=False)).index.values

sorted_lower = data.reindex(new_index)

# normalization of Reviews column.

sorted_lower['Reviews'] = sorted_lower.Reviews/381023



# sorting according to value of Reviews

new_index = (mediumRating['Reviews'].sort_values(ascending=False)).index.values

sorted_medium = data.reindex(new_index)

# normalization of Reviews column.

sorted_medium['Reviews'] = sorted_medium.Reviews/1828284



# sorting according to value of Reviews

new_index = (goodRating['Reviews'].sort_values(ascending=False)).index.values

sorted_good = data.reindex(new_index)

# normalization of Reviews column.

sorted_good['Reviews'] = sorted_good.Reviews/22430188





# sorting according to value of Reviews

new_index = (veryGoodRating['Reviews'].sort_values(ascending=False)).index.values

sorted_veryGood = data.reindex(new_index)

# normalization of Reviews column.

sorted_veryGood['Reviews'] = sorted_veryGood.Reviews/44893888
# %% scatter plot

plt.scatter(sorted_worst.Reviews,sorted_worst.Installs,color="blue",label="worst")

plt.scatter(sorted_lower.Reviews,sorted_lower.Installs,color="red",label="lower")

plt.scatter(sorted_medium.Reviews,sorted_medium.Installs,color="green",label="medium")

plt.scatter(sorted_good.Reviews,sorted_good.Installs,color="orange",label="good")

plt.scatter(sorted_veryGood.Reviews,sorted_veryGood.Installs,color="purple",label="very good")



plt.legend()

plt.xlabel("Reviews")

plt.ylabel("Installs")

plt.title("Normalized Scatter plot")

plt.show()
# %% point plot

# distribution of 5 groups as scatter plot

sorted_lower_first_twenty = sorted_lower.head(20)

sorted_lower_first_twenty['Size'] = sorted_lower.Size/78.0000

sorted_lower_first_twenty['Installs'] = sorted_lower.Installs/50000000





f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='Reviews',y='Installs',data=sorted_lower_first_twenty,color='lime',alpha=0.8)

sns.pointplot(x='Reviews',y='Size',    data=sorted_lower_first_twenty,color='red',alpha=0.8)

plt.text(40,0.6,'Installs',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'Size',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Reviews',fontsize = 15,color='blue')

plt.ylabel('Installs or Size',fontsize = 15,color='blue')

plt.title('Sorted-normalized lower group',fontsize = 20,color='blue')

plt.grid()
sorted_good_first_twenty = sorted_good.head(20)

sorted_good_first_twenty['Size'] = sorted_good.Size/986000

sorted_good_first_twenty['Installs'] = sorted_good.Installs/500000000



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='Reviews',y='Installs',data=sorted_good_first_twenty,color='blue',alpha=0.8)

sns.pointplot(x='Reviews',y='Size',    data=sorted_good_first_twenty,color='orange',alpha=0.8)

plt.text(40,0.6,'Installs',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'Size',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Reviews',fontsize = 15,color='blue')

plt.ylabel('DENENEInstalls or Size',fontsize = 15,color='blue')

plt.title('Sorted-normalized good group',fontsize = 20,color='blue')

plt.grid()
sns.countplot(data.Type)

plt.title("count of free and paid",color = 'blue',fontsize=15)
#%% pie plot



labels = data.Category.value_counts().index

#colors = ['grey','blue','red','yellow']

#explode = [0,0,0,0]

sizes = data.Category.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.title('Categories',color = 'blue',fontsize = 15)



# find smaller categories than 3% at sizes

categoryCount = data["Category"].value_counts(dropna =False)









# sum of numbers of Category in filtreYuzdeUc

toplamCategory = categoryCount.sum(axis=0)





oran=[]

for i in categoryCount:

    oran.append(i/toplamCategory)