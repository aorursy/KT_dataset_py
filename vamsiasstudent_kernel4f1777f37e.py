import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("../input/googleplaystore.csv")

#data1=pd.read_csv("googleplaystore.csv")

data.dropna(inplace=True)

#data1.dropna(inplace=True)
data.info()
data.tail()
data["Genres"].value_counts().count()
data["Reviews"][10840]
'''data["Reviews"]=data["Reviews"].apply(lambda x :x.replace("M","") if "M" in str(x) else x)



data["Reviews"]=data["Reviews"].apply(lambda x :x.replace(",","") if "M" in str(x) else x)



data["Reviews"]=data["Reviews"].apply(lambda x :int(x))

'''
def filter(per):

    if "M" in str(per) and "," in str(per):

        per = str(per).replace("M","")

        per = per.replace(",","")

        return int(per)*1000000

    elif "M" in str(per):

        per = int(str(per).replace("M",""))

        return per*1000000

    elif "," in str(per):

        per = str(per).replace(",","")

        return int(per)

    

    else:  

        return int(per)
#data["Reviews"] = list(map(filter,data["Reviews"].values))
data["Reviews"] =data["Reviews"].apply(filter)
data.info()
data1.sample(5,random_state=55)
data1.sample(5,random_state=55)
data[data["Reviews"]==data.Reviews.max()]
def filter1(per):

    per = str(per)

    if "M" in per:

        per = per.replace("M","")

        return float(per)

    elif per == "Varies with device":

        return np.NaN

    elif "k" in per:

        return float(per.replace("k",""))/1000

    else:

        return float(per)

#l1= list(map(filter1,data["Size"]))
data["Size"]=data["Size"].apply(filter1) 
data.info()
data["Size"].min()
data["Size"].max()
type(data.iloc[1567].Size)
data.sample(5)
def filter2(per):

    per = str(per)

    if "+" in per:

        per = per.replace("+","")

    if "," in per:

        per = per.replace(",","")

        

    return int(per)
#l2 = list(map(filter2,data["Installs"]))
#filter2('10,000+')
data["Installs"]=data["Installs"].apply(filter2)
data.info()
data[data["Installs"].isnull()]
data[data["App"]=="Home Pony 2"]
def filter3(per):

    per = str(per)

    if "$" in per:

        per='$4.99'.split("$")[1]

    return (float(per)*69.44)
float('$4.99'.split("$")[1])
data["Price"]=data["Price"].apply(filter3)
data.info()
data.to_csv("clean_data.csv",index=False)

cleandata=pd.read_csv("clean_data.csv")
cleandata.sample(5)
cleandata.info()==data.info()
sns.pairplot(cleandata,hue="Type")
temp=pd.DataFrame(cleandata["Content Rating"].value_counts()).reset_index()
temp.columns=['user', 'Content Rating']
temp
plt.figure(figsize=(12,6))

sns.barplot(data=temp,x="user",y="Content Rating")
sns.set_context('talk',font_scale=1)

plt.figure(figsize=(17,13))

sns.countplot(data=cleandata,y="Category",hue="Type")

plt.figure(figsize=(16,12))

sns.boxplot(data=cleandata,x="Size",y="Category",palette='rainbow')
plt.figure(figsize=(17,13))

sns.countplot(data=cleandata[cleandata["Reviews"]>1000000],y="Category",hue="Type")

plt.title("most popular apps with 1000000+ reviews")

plt.xlabel("no of apps")
plt.figure(figsize=(12,6))

sns.distplot(cleandata["Rating"],bins=10,color="red")
sns.countplot(x=cleandata["Type"])
sns.heatmap(cleandata.corr(),cmap='coolwarm')
sns.scatterplot(x="Installs",y="Reviews",data=cleandata,palette="rainbow")
plt.figure(figsize=(16,6))

sns.scatterplot(data=cleandata[cleandata["Reviews"]>100000],x="Size",y="Rating",hue="Type")

plt.title("apps with reviews graterthan 100000")
sns.kdeplot(data=cleandata["Size"])

plt.title("size vs count")

plt.xlabel("")
listcat = cleandata["Category"].unique()

i=0
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")

plt.title(str(listcat[i]))

i+=1
cleandata.columns
cleandata.groupby('Category')["Rating"].mean().index
plt.figure(figsize=(12,6))



sns.scatterplot(x = cleandata.groupby('Category')['Rating'].mean().index, y = cleandata.groupby('Category')['Rating'].mean().values)

plt.ylabel('Category', fontsize=13)

plt.xlabel('Rating', fontsize=13)

plt.xticks(rotation=90)

plt.title("avg rating table based on category")
most_popular_apps = cleandata[(cleandata["Reviews"]>10000000) ][ (cleandata["Rating"]>=4.5)]
sns.countplot(most_popular_apps["Category"])

plt.xticks(rotation=90)
sns.pairplot(most_popular_apps,hue="Type")
sns.heatmap(most_popular_apps.corr())







