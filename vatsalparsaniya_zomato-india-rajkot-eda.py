import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

import zipfile

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# # if Data in Zip file 

# zf = zipfile.ZipFile('Data/zomato-restaurants-in-india.zip') 

# Zomato_d = pd.read_csv(zf.open('zomato_restaurants_in_India.csv'))

# Zomato_d.head()
# read Zomato_india data

Zomato_india =pd.read_csv('/kaggle/input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv')
Zomato_india.head()
Zomato_india.info()
# removing Dummy Data

unique_id_size = len(Zomato_india.res_id.unique())

print("unique_id_size : ",unique_id_size)

total_data_size = Zomato_india.shape[0]

print("total_data_size : ",total_data_size)

print("total number of Dummy Data :",total_data_size-unique_id_size)

Zomato_india.drop_duplicates(subset="res_id",keep='first',inplace=True)

unique_id_size = len(Zomato_india.res_id.unique())

print("\nunique_id_size : ",unique_id_size)

total_data_size = Zomato_india.shape[0]

print("total_data_size : ",total_data_size)

print("total number of Dummy Data :",total_data_size-unique_id_size)
Zomato_india.info()
#top 20 franchises...

number_of_franchised = Zomato_india.name.value_counts()[:20]

plt.figure(figsize =(10,10))

g = sns.barplot(y= number_of_franchised.index,x=number_of_franchised)

g.set_yticklabels(g.get_yticklabels(),rotation=15)

plt.title("Most 20 franchised Restaurants in India....")

plt.xlabel("franchises...")

plt.show()
home_delivery = Zomato_india.delivery.value_counts()

sizes =[home_delivery[0],home_delivery[1],home_delivery[-1]]

plt.figure(figsize=(7,7))

g = plt.pie(sizes,autopct='%1.1f%%',shadow=False,explode=(0,0.05,0.05),

           wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'dashed', 'antialiased': True},

           labels = ["Not Decide = {}".format(home_delivery[0]),"No = {}".format(home_delivery[-1]),

                    "Yes = {}".format(home_delivery[1])])

plt.title("Delivery At Home By Restorents :")

plt.legend(loc = 4)

plt.show()
Zomato_india_group_by_city = Zomato_india.groupby(Zomato_india.city)

franchies_dataset = pd.DataFrame()

for city_name,group in Zomato_india_group_by_city:

    temp_data = {}

    temp_data["city"] = city_name

    for name_of_franchies in number_of_franchised.index:

        value = group[group.name == name_of_franchies].res_id.unique() 

        try:

            temp_data[name_of_franchies] = len(value)

        except:

            temp_data[name_of_franchies] = 0

    franchies_dataset = franchies_dataset.append(temp_data,ignore_index=True)

franchies_dataset = franchies_dataset.set_index("city")

franchies_dataset = franchies_dataset.astype(int)
franchies_dataset.head(20)
# For top 20 franchised Restaurants 

plt.figure(figsize =(50,10))

g = sns.barplot(y= franchies_dataset.sum(axis=1),x=franchies_dataset.index)

g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.title("Numbers of top  franchised Restaurants in India ....")

plt.xlabel("City...")

plt.ylabel("number Of Restorents...")

plt.show()
# How Domino's Pizza distributes over the India

plt.figure(figsize =(40,10))

plt.subplot(2,1,1)

g = sns.barplot(y= franchies_dataset["Domino's Pizza"],x=franchies_dataset.index)

g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.title("Numbers of top  franchised Restaurants in India ....")

plt.xlabel("City...")

plt.ylabel("number Of Restorents...")

plt.subplot(2,1,2)

g = sns.barplot(y= franchies_dataset["Cafe Coffee Day"],x=franchies_dataset.index)

g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.title("Numbers of top  franchised Restaurants in India ....")

plt.xlabel("City...")

plt.ylabel("number Of Restorents...")

plt.show()
Zomata_Cuisines = pd.DataFrame()

for element in Zomato_india.cuisines.unique():

    temp_dataFrame = pd.DataFrame(str(element).split(", "))

    Zomata_Cuisines = pd.concat([Zomata_Cuisines,temp_dataFrame])

    Zomata_Cuisines.drop_duplicates(keep='first',inplace= True)

Zomata_Cuisines = Zomata_Cuisines.set_index(0)

Zomata_Cuisines.index.unique()     
Zomata_Cuisines_data = pd.DataFrame()

for name , group in Zomato_india.groupby("city"):

    temp_data = dict((el,0) for el in Zomata_Cuisines.index.unique())

    temp_data["city"] = name

    for cuisin in group.cuisines.unique():

        for sub_cuisin in str(cuisin).split(", "):

            temp_data[sub_cuisin] = temp_data[sub_cuisin] + 1

    Zomata_Cuisines_data  = Zomata_Cuisines_data.append(temp_data,ignore_index=True)

Zomata_Cuisines_data = Zomata_Cuisines_data.set_index("city")

Zomata_Cuisines_data = Zomata_Cuisines_data.astype(int)

Zomata_Cuisines_data.head()
temp_data = Zomata_Cuisines_data.T

value = temp_data.sum(axis=1).sort_values(ascending=False)[:20]

temp_data = temp_data[temp_data.index.isin(value.index)]

Top_20_Cuisines = temp_data.T

Top_20_Cuisines.head(10)
Top_20_Cuisines.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
Top_20_Cuisines.T.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
# TOP 10 HEIGHEST REATED RESTAURANTS

Zomato_india["rating_number"] = (Zomato_india.aggregate_rating*1000)/Zomato_india.votes

Top_10_restaurants = pd.DataFrame(columns=["1_st","2_nd","3_rd","4_th","5_th","6_th","7_th","8_th","9_th","10_th"])

City_name = pd.DataFrame(columns=["city"])

for c_name,group in Zomato_india.groupby("city"):

    temp = group.sort_values(["rating_number"],axis=0,ascending=False)

    temp_data = []

    City_name.loc[len(City_name)] = c_name

    for name in temp.name[:10]:

        temp_data.append(name)

    Top_10_restaurants.loc[len(Top_10_restaurants)] = temp_data

Top_10_restaurants = pd.concat([Top_10_restaurants,City_name],axis=1)

Top_10_restaurants = Top_10_restaurants.set_index("city")

Top_10_restaurants.head(10)
# RATING PIE CHART 

sizes = [len(Zomato_india.loc[Zomato_india.aggregate_rating >=4]),

        len(Zomato_india.loc[(Zomato_india.aggregate_rating >=3) & Zomato_india.aggregate_rating < 4]),

        len(Zomato_india.loc[(Zomato_india.aggregate_rating >=2) & Zomato_india.aggregate_rating < 3]),

        len(Zomato_india.loc[(Zomato_india.aggregate_rating >=1) & Zomato_india.aggregate_rating < 2]),

        len(Zomato_india.loc[(Zomato_india.aggregate_rating >=0) & Zomato_india.aggregate_rating < 1])]

plt.figure(figsize=(7,7))

g = plt.pie(sizes,autopct='%1.1f%%',shadow=True,explode=(0.1,0.1,0.1,0.1,0.1),

           wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'dashed', 'antialiased': True},

           labels = [">=4",">=3 & <4",">=2 & <3",">=1 & <2",">=0 & <1"])

plt.title("RATING PIE CHART")

plt.legend(loc = 3)

plt.show()
# Rating Table 

Rating_data = pd.DataFrame(columns=[">=4",">=3 & <4",">=2 & <3",">=1 & <2",">=0 & <1"])

City_name = pd.DataFrame(columns=["city"])

for c_name,group in Zomato_india.groupby("city"):

    City_name.loc[len(City_name)] = c_name

    temp_data = [len(group.loc[Zomato_india.aggregate_rating >=4]),

        len(group.loc[(Zomato_india.aggregate_rating >=3) & group.aggregate_rating < 4]),

        len(group.loc[(Zomato_india.aggregate_rating >=2) & group.aggregate_rating < 3]),

        len(group.loc[(Zomato_india.aggregate_rating >=1) & group.aggregate_rating < 2]),

        len(group.loc[(Zomato_india.aggregate_rating >=0) & group.aggregate_rating < 1])]

    Rating_data.loc[len(Rating_data)] = temp_data

Rating_data = pd.concat([City_name,Rating_data],axis=1)

Rating_data = Rating_data.set_index("city")

Rating_data.head(10)
plt.figure(figsize =(15,15))

sns.set_context("notebook", font_scale=1.1)

g = sns.scatterplot(y="average_cost_for_two",hue="rating_text",x="aggregate_rating",data=Zomato_india)

plt.title("average_cost_for_two VS rating_text")

plt.legend(loc = 2)

plt.show()
# aggregate_rating distributions over the india

sns.distplot(Zomato_india.aggregate_rating)

plt.show()
# number of 4+ rated Restaurant

plt.figure(figsize =(60,10))

temp = Rating_data

temp = temp.sort_values([">=4"],axis=0,ascending=False)

g = sns.barplot(x= temp.index,y=temp[">=4"])

g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.title("number of 4+ rated Restaurant")

plt.xlabel("City")

plt.show()
Zomato_india.establishment.unique()
# establishment setup in india

establishment_data = pd.DataFrame()

for name , group in Zomato_india.groupby("city"):

    temp_data = {}

    temp_data["city"] = name

    for establi in Zomato_india.establishment.unique():

        value = len(group[group.establishment == establi].establishment)

        col_name = establi.replace("[","").replace("]","").replace("'","")

        temp_data[col_name] = value

    establishment_data = establishment_data.append(temp_data,ignore_index=True)

establishment_data = establishment_data.set_index("city")

establishment_data = establishment_data.astype(int)

establishment_data.head(10)
f = plt.figure(figsize=(19, 15))

plt.matshow(establishment_data.corr(), fignum=f.number)

plt.xticks(range(establishment_data.shape[1]), establishment_data.columns, fontsize=14, rotation=75)

plt.yticks(range(establishment_data.shape[1]), establishment_data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.show()
establishment_data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
#Bakery Distribution in India

plt.figure(figsize =(60,10))

g = sns.barplot(x= establishment_data.index,y=establishment_data.Bakery)

g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.title("Bakery Distribution in India")

plt.xlabel("City")

plt.show()
avg_cost_of_2_by_locality = {}

for name , group in Zomato_india.groupby("locality"):

    avg_cost_of_2_by_locality[name] = group.average_cost_for_two.sum() / (len(group))

avg_cost_of_2_by_locality = sorted(avg_cost_of_2_by_locality.items(), key=lambda x: x[1], reverse=True)

avg_cost_of_2_by_locality = pd.DataFrame(avg_cost_of_2_by_locality,columns=["Locality","Avg_cost_for_2"])

avg_cost_of_2_by_locality = avg_cost_of_2_by_locality.set_index("Locality")

avg_cost_of_2_by_locality.head(10)
#country's most 15 expensive locality according Food of two(data by Zomato)

plt.figure(figsize =(10,10))

g = sns.barplot(y= avg_cost_of_2_by_locality.index[:15],x=avg_cost_of_2_by_locality.Avg_cost_for_2[:15])

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("country's most 15 expensive locality according Food of two(data by Zomato)")

plt.ylabel("Locality")

plt.show()
#find unique heighlights

highlights_dataset = pd.DataFrame()

for sub_highlights in Zomato_india.highlights.unique():

    temp = pd.DataFrame(sub_highlights.replace("'","").strip('][').split(', '))

    highlights_dataset = pd.concat([highlights_dataset,temp])

    highlights_dataset.drop_duplicates(keep='first',inplace=True)

highlights_dataset = highlights_dataset.set_index(0)

highlights_dataset.index.unique()
# restaurants devides by highlights 

highlight_dataset = pd.DataFrame(columns=highlights_dataset.index)

for name,group in Zomato_india.groupby('city'):

    temp = dict((el,0) for el in highlights_dataset.index)

    temp["city"] = name

    for str_hightlight in group.highlights:

        value = str_hightlight.replace("'","").strip('][').split(', ')

        for sub_value in value:

            temp[sub_value] = temp[sub_value] + 1

    highlight_dataset = highlight_dataset.append(temp,ignore_index=True)

highlight_dataset = highlight_dataset.set_index("city")            
highlight_dataset.head(10)
#distribution of Breakfast , Lunch , Dinner

fig, ax = plt.subplots(figsize=(35,10))

ax.plot(highlight_dataset.index,highlight_dataset.Lunch,label="Lunch")

ax.plot(highlight_dataset.index,highlight_dataset.Dinner,label="Dinner")

ax.plot(highlight_dataset.index,highlight_dataset["All Day Breakfast"],label="All Day Breakfast")

ax.plot(highlight_dataset.index,highlight_dataset.Delivery,label="Delivery")

plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

ax.legend()

plt.show()
plt.figure(figsize =(10,10))

g  = sns.barplot(y="establishment",x="average_cost_for_two",data = Zomato_india)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("establishment vs average_cost_for_two")

plt.ylabel("establishment")

plt.show()
Choose_city = "Rajkot"
Zomato_Sub_city = Zomato_india[Zomato_india.city == Choose_city]
Zomato_Sub_city.info()
Zomato_Sub_city.head()
estlablishment_subcity = establishment_data[establishment_data.index == Choose_city]

estlablishment_subcity = estlablishment_subcity.sort_values([Choose_city],axis=1,ascending=False)

estlablishment_subcity = estlablishment_subcity.transpose()

plt.figure(figsize =(10,10))

g = sns.barplot(x= estlablishment_subcity[Choose_city],y=estlablishment_subcity.index)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("ESTLABLISHMENT distribution in {}".format(Choose_city))

plt.xlabel("Count...")

plt.ylabel("ESTLABLISHMENT")

plt.show()

Total_restaurant_inCity = len(Zomato_Sub_city)

QuickBite_data = Zomato_Sub_city[Zomato_Sub_city.establishment == "['Quick Bites']"]

sizes =  [len(QuickBite_data),Total_restaurant_inCity]

plt.figure(figsize=(7,7))

g = plt.pie(sizes,autopct='%1.1f%%',shadow=False,explode=(0.05,0.05),

           wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'dashed', 'antialiased': True},

           labels = ["Quick Bites","Other"])

plt.title(" Quick Bites Restorents in {} :".format(Choose_city))

plt.legend(loc = 4)

plt.show()
Graph_data = QuickBite_data.sort_values(["aggregate_rating"],axis=0,ascending=False)[:10]

plt.figure(figsize =(5,5))

g = sns.barplot(x= "aggregate_rating",y="name",data = Graph_data)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("QuickBite distribution in {}".format(Choose_city))

plt.show()
CasualDining_data = Zomato_Sub_city[Zomato_Sub_city.establishment == "['Casual Dining']"]

Graph_data = CasualDining_data.sort_values(["aggregate_rating"],axis=0,ascending=False)[:10]

plt.figure(figsize =(5,5))

g = sns.barplot(x= "aggregate_rating",y="name",data = Graph_data)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("CasualDining distribution in {}".format(Choose_city))

plt.show()
home_delivery = Zomato_Sub_city.delivery.value_counts()

sizes =[home_delivery[0],home_delivery[1],home_delivery[-1]]

plt.figure(figsize=(7,7))

g = plt.pie(sizes,autopct='%1.1f%%',shadow=False,explode=(0,0.05,0.05),

           wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'dashed', 'antialiased': True},

           labels = ["Not Decide = {}".format(home_delivery[0]),"No = {}".format(home_delivery[-1]),

                    "Yes = {}".format(home_delivery[1])])

plt.title("Delivery At Home By Restorents in {} :".format(Choose_city))

plt.legend(loc = 4)

plt.show()
#top 20 franchises...

number_of_franchised = Zomato_Sub_city.name.value_counts()[:20]

plt.figure(figsize =(10,10))

g = sns.barplot(y= number_of_franchised.index,x=number_of_franchised)

g.set_yticklabels(g.get_yticklabels(),rotation=15)

plt.title("Most 20 franchised Restaurants in India....")

plt.xlabel("franchises...")

plt.show()
# TOP 10 HEIGHEST REATED RESTAURANTS

Top_10_restaurants.loc[[Choose_city]]
plt.figure(figsize =(10,10))

g = sns.barplot(x=Zomato_Sub_city.locality.value_counts().values,y=Zomato_Sub_city.locality.value_counts().index)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("Locality wise restaurant counts in {}".format(Choose_city))

plt.ylabel("Locality")

plt.xlabel("Count")

plt.show()
Top_20_Cuisines.loc[[Choose_city]]
Top_20_Cuisines.loc[[Choose_city]].sum()

plt.figure(figsize =(10,10))

g = sns.barplot(x=Top_20_Cuisines.loc[[Choose_city]].sum().values,y=Top_20_Cuisines.loc[[Choose_city]].sum().index)

g.set_yticklabels(g.get_yticklabels(),rotation=0)

plt.title("number of cuisines restaurant in {}".format(Choose_city))

plt.ylabel("Cuisines")

plt.xlabel("Count...")

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(x="average_cost_for_two",y="locality_verbose",data=Zomato_Sub_city)

plt.title("LOCALITY VS AVG COST FOR 2 PERSON")

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(x="average_cost_for_two",y="rating_text",data=Zomato_Sub_city)

plt.title("Rating Taxt VS AVG COST FOR 2 PERSON")

plt.show()
plt.figure(figsize=(40,10))

sns.barplot(y="average_cost_for_two",x="photo_count",data=Zomato_Sub_city,hue="rating_text")

plt.title("AVG COST FOR 2 PERSON VS PHOTO COUNT")

plt.show()
plt.figure(figsize=(50,10))

sns.barplot(y="average_cost_for_two",x="votes",data=Zomato_Sub_city)

plt.title("AVG COST FOR 2 PERSON VS VOTES")

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(y="establishment",x="votes",data=Zomato_Sub_city)

plt.title("ESTABLISHMENT VS PHOTO COUNT")

plt.show()
Total_restaurant_inCity = len(Zomato_Sub_city)

highlight_subCity = highlight_dataset.loc[Choose_city]

Graph_data = highlight_subCity.sort_values(ascending=False)[:20]

plt.figure(figsize=(10,10))

sns.barplot(y=Graph_data.index,x=Graph_data.values)

plt.title("ESTABLISHMENT VS PHOTO COUNT")

plt.show()