#Hi,This is Yokhesh. In this program, I am using the google apps dataset to perform some data cleaning, 
#data visulaisation and some useful calculation that could help us better understand the data.
# Initially, we are assigning each data set to a variable.
# The variable 'data' contains the info from googleplaystore_user_reviews.csv
# The variable 'data1' contains the info from googleplaystore.csv
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/googleplaystore_user_reviews.csv")
data1 = pd.read_csv("../input/googleplaystore.csv")


data1 = data1.dropna()
data1.head()

# Any results you write to the current directory are saved as output.
data1.describe()

data1.info()
# Now, we will be be calculating the number of apps under each category and 
#then a bar chart to visualize the difference in the number of apps from one category to another
#Number of apps under every category
f = data1.groupby('Category').size().reset_index(name='count')


e1 = f.sort_values(by=['count'],ascending=False)
print(e1)
count = e1["count"]
cat = e1["Category"]
plt.figure(figsize=(20, 30))
plt.bar(cat, count, width=0.3) 
plt.xticks(rotation=90)

plt.xlabel('App Category')
plt.ylabel('Instances')




# The category 'Family' seems to have the maximum number of apps with a total count of 1746
#The next important feature in the dataset is the rating for each apps.
#Now, we will see the avergae rating for each category and arrange in the ascending order.
#Again, a bar chart is used to visualize it. This will give us an idea which category is most preferred 
#and used.
#Average rating for each category
ra = data1[['Category','Rating']]
cate = []
rating_average = []
for index, row in e1.iterrows():
    cate.append(row['Category'])
    # art has all the rating corresponding to one category in row of e1
    art = ra.loc[ra.Category == row['Category']] 
    #Now, we are summing up all the ratings belonging to one category
    Total = art['Rating'].sum()
    rating_average.append(Total/row['count'])
z = [x for _,x in sorted(zip(rating_average,cate))] 
rate = rating_average.sort()
plt.figure(figsize=(20, 30))
plt.bar(z, rating_average, width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('App Category')
plt.ylabel('Average Rating')

#Now, we are removing some formatting errors and symbols from the columns category, Installs and price

# Removing formatting errors in each row
data1.Category = data1.Category.apply(lambda x: x.replace('_',' ')if '_' in str(x) else x)
data1.Installs = data1.Installs.apply(lambda x: x.replace('+','')if '+' in str(x) else x)
data1.Price = data1.Price.apply(lambda x: x.replace('$','')if '$' in str(x) else x)

#Calculating the unique varieties in content rating
data1.groupby('Content Rating').size().reset_index(name='count1')

#Calculating the unique varieties in Type
data1.groupby('Type').size().reset_index(name='count')
#Calculating the unique varieties in Size
data1.groupby('Size').size().reset_index(name='count')
#Calculating the unique varieties in Android Version
data1.groupby('Android Ver').size().reset_index(name='count')
#In the following piece of code, we are developing a pie chart for each different category
# Pie chart for the App category
f = data1.groupby('Category').size().reset_index(name='count')
cat = f['Category']
no = f['count']
#plt.figure(figsize=(16,8))
#plt.pie(no,labels = cat)
fig1, ax1 = plt.subplots()
ax1.pie(no, labels=cat,autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 7)
plt.legend(cat,loc="center")
#We then visualize the average number of Installs made under each category
# Average number of installs under each category
inst = data1[['Category','Installs']]
inst.Installs = inst.Installs.apply(lambda x: x.replace(',','')if ',' in str(x) else x)
f = inst.groupby('Category').size().reset_index(name='count')
inst['Installs'] = inst['Installs'] 
t = inst.groupby('Category').Installs.sum().reset_index(name='count1')
er = t['count1']
er1 = f['count']
for i in range(len(er)):
    y = int(er[i])
    er[i] = y//(er1[i])
t = pd.DataFrame(t)
t.sort_values(by=['count1'],ascending=False)

#Then we segregate the apps based on whether its free or paid and use a pie chart to show the difference
#Percentage of free and paid apps
fre = data1[['Category','Type']]
f = fre.groupby('Type').Category.size().reset_index(name='count')
label = ['Paid','Free']
fig1, ax1 = plt.subplots()
ax1.pie(f['count'],labels = f['Type'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
# According to the pie chart, only 6.9% of the apps are paid.
# Now that we have calculated the total number of free and paid apps.
#The next step would be to calcuate the number of free apps under each category
r = fre['Category']
f = data1.groupby('Category').size().reset_index(name='count')
cat = f['Category']
cat1 = []
total_free=[]
total_paid = []
for i in range(len(cat)):
    # tr contains the columns containing only the category present in cat[i]
    tr = fre.loc[fre['Category'] == cat[i]]
    p = tr.groupby('Type').Category.size().reset_index(name='count')
    o = p['count']
    free =o[0]
    if (len(o)) == 2:
        paid = o[1]
    else:
        paid = 0
    #free = p.loc[p['Type'] == 'Free', 'count']
    #free = p[1]
    cat1.append(cat[i])
    total_free.append(free)
    total_paid.append(paid)


z1 = [x for _,x in sorted(zip(total_free,cat1))] 
tf = total_free.sort()
plt.figure(figsize=(10, 10))
plt.bar(z1, total_free, width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('App Category')
plt.ylabel('Free Apps')
plt.title('Number of free apps under each category')
#The family category has the maximum number of free apps
#Then we calculate the number of paid apps under each category
z2 = [x for _,x in sorted(zip(total_paid,cat1))] 
tp = total_paid.sort()
plt.figure(figsize=(10, 10))
plt.bar(z2, total_paid, width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('App Category')
plt.ylabel('Paid Apps')
plt.title('Number of paid apps under each category')
#Now we will focus more on the paid apps.
# We are calculating the average price of apps under each category in dollars
price = data1[['Category','Price']]
app_price = []
number_of_app_under_each_category = f['count']
for i in range(len(cat)):
    ret = price.loc[fre['Category'] == cat[i]]
    ret['Price'] = pd.to_numeric(ret['Price'])
    Total = ret['Price'].sum()
    app_price.append(Total/number_of_app_under_each_category[i])
    #print(Total)
z3 = [x for _,x in sorted(zip(app_price,cat1))] 
plt.figure(figsize=(10, 10))
plt.bar(z3, app_price, width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('App Category')
plt.ylabel('Price')
plt.title('Average price under each category')
##The Art and Design category appears to be the most expensive one ranging close to $7.8
# The app with highest price
data1['Price'] = pd.to_numeric(data1['Price'])
data1.loc[data1['Price'].idxmax()]
#The apps with the highest price is I'm Rich - Trump Edition under LIFESTYLE category
#Next, we calculate the percentage of apps under each content ratings
#Percentage of different apps content rating
fre = data1[['Category','Content Rating']]
f = fre.groupby('Content Rating').Category.size().reset_index(name='count')
#label = ['Paid','Free']
fig1, ax1 = plt.subplots()
ax1.pie(f['count'],labels = f['Content Rating'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
print(f)
f
label = f['Content Rating']
#z2 = [x for _,x in sorted(zip(total_paid,cat1))] 
plt.bar(label, f['count'], width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('Content Rating')
plt.ylabel('Number of Apps')
plt.title('Number of Apps under each content Rating')

# We then calculate the percentage of Installs overall
#Percentage of Installs
fre = data1[['Category','Installs']]
f = fre.groupby('Installs').Category.size().reset_index(name='count')
fig1, ax1 = plt.subplots()
ax1.pie(f['count'],labels = f['Installs'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
print(f)
plt.title('Percentage of different apps content rating')
#Since we have good idea of percentage in content ratings and installs.
# We are merging them to both to obtain a rating vs installs plot
# Plot of Installs vs Rating
ivr = data1[['Rating', 'Installs']]
rate = ivr['Rating']
inst = ivr['Installs']
plt.scatter(rate, inst)
plt.xlabel('App Rating')
plt.ylabel('Installs')
plt.title('Installs Vs Rating')
plt.show()
# The next calulation is the number of apps under each anroid version.
# Plot of Installs vs Type
av = data1.groupby('Android Ver').size().reset_index(name='count')
plt.bar(av['Android Ver'], av['count'], width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('Android Version')
plt.ylabel('count')
plt.title('Number of Apps under each Android Version')
#Now, we move to the next dataset googleplaystore_user_reviews.csv
data = data.dropna()

data.describe()
data.info()
# Arranging the apps in order of most reviewed
ef = data.groupby('App').size().reset_index(name='count')
apps = ef.sort_values(by=['count'],ascending=False)
print(apps)
just_app = apps['App']
#Looks like the bowmasters apps has been reviewed 312 times.
#Calculating the positive, negative and neutral review of the most reviewed app and visulaizing with the 
#bar chart
# positive and negative review of the most reviewed apps
re = data[['App','Sentiment']]
a = re.loc[re.App == 'Bowmasters'] 
b = a.groupby('Sentiment').App.size().reset_index(name='count')
plt.bar(b['Sentiment'], b['count'], width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Bar chart for Sentiment of review for the Bowmasters app')
fig1, ax1 = plt.subplots()
ax1.pie(b['count'], labels = b['Sentiment'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
plt.title('Pie chart for Sentiment of review for the Bowmasters app')
# We are doing a similar calculation other famous apps like facebook and google
a1 = re.loc[re.App == 'Facebook'] 
b1 = a1.groupby('Sentiment').App.size().reset_index(name='count')
plt.bar(b1['Sentiment'], b1['count'], width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Bar chart for Sentiment of review for the Facebook app')
fig1, ax1 = plt.subplots()
ax1.pie(b1['count'], labels = b1['Sentiment'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
plt.title('Pie chart for Sentiment of review for the Bowmasters app')
a2 = re.loc[re.App == 'Google'] 
b2 = a2.groupby('Sentiment').App.size().reset_index(name='count')
plt.bar(b2['Sentiment'], b2['count'], width=0.3) 
plt.xticks(rotation=90)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Bar chart for Sentiment of review for the Google app')
fig1, ax1 = plt.subplots()
ax1.pie(b2['count'], labels = b2['Sentiment'],autopct='%1.1f%%', 
        shadow=False, startangle=90, radius = 1)
plt.title('Pie chart for Sentiment of review for the Google app')
# Now, we take sentiment polarity into account. A positive review would have a pisitive polarity 
#and a negative review will have a negative polarity. 
# We are taking the average of the sentiment polarity under each app and arraging the apps in descending
#order of the average
resp = data[['App','Sentiment_Polarity']]
teh = resp.groupby('App').Sentiment_Polarity.mean().reset_index(name='count1')
sentiment_pol = teh.sort_values(by=['count1'],ascending=False)
print(sentiment_pol)
#Now, we are doing the same with subjectivity.
# We are taking the average of the sentiment subjectivity under each app and arraging the apps in descending
#order of the average
resp1 = data[['App','Sentiment_Subjectivity']]
teh1 = resp1.groupby('App').Sentiment_Subjectivity.mean().reset_index(name='count2')
sentiment_sub = teh1.sort_values(by=['count2'],ascending=False)
print(sentiment_sub)
