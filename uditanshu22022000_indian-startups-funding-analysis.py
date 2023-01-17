import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

y = []

for i in df["Date"].values:

    if '.' in i:

        y.append(i.split('.')[-1])

    else:

        y.append(i.split('/')[-1])

df["Year"] = y

p = df["Year"].value_counts()

year = p.index

q = np.argsort(year)

fund = p.values

for i in q:

    print(year[i],fund[i])
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["CityLocation"].dropna(inplace = True)

d = {}

for i in df["CityLocation"]:

    d[i] = d.get(i,0) + 1

    

d["Bangalore"] += d["bangalore"] - d['SFO / Bangalore'] - d['Seattle / Bangalore']

d["New Delhi"] += d["Delhi"]

d["Hyderabad"] -= d["Goa/Hyderabad"] - d['Dallas / Hyderabad'] 

del d["bangalore"]

del d["Delhi"]

del d["Goa/Hyderabad"]

del d['Dallas / Hyderabad']



d1 = sorted(d, key=d.get, reverse=True)



c = 1

d2 = {}

for i in d1:

    if c == 11:

        break

    d2[i] = 0

    c += 1

    

for i in d2:

    for j in d:

        if i in j:

            d2[i] += d[j]

              

x = []

y = []

for i in d2:

    print(i,d2[i])

    x.append(i)

    y.append(d2[i])

plt.axis("equal")

plt.pie(y,labels = x)

plt.show()
def modified(amount):

    return int(amount.replace(',',''))

def city(c):

    c = str(c)

    return c.split("/")[0].strip()

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["CityLocation"].fillna('',inplace = True)

df["AmountInUSD"].fillna('0',inplace = True)

df["AmountInUSD"] = df["AmountInUSD"].apply(modified)

df["CityLocation"] = df["CityLocation"].apply(city)

df["CityLocation"].replace("bangalore","Bangalore",inplace = True)

df["CityLocation"].replace("Delhi","New Delhi",inplace = True)

df = df[df["CityLocation"] != ""]

a = df["CityLocation"]

b = df["AmountInUSD"]

d = {}

for i in a.index:

    d[a[i]] = d.get(a[i],0) + b[i]

    

d1 = sorted(d, key=d.get , reverse=True) 





c = 1

d2 = {}

for i in d1:

    if c == 11:

        break

    d2[i] = d2.get(i,0) + d[i]

    c += 1

sum = 0

for i in d2.values():

    sum += i

for i in d2:

    print(i,format(d2[i]*100/sum,"0.2f"))
def modified(amount):

    return int(amount.replace(',',''))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["AmountInUSD"].fillna('0',inplace = True)

df["AmountInUSD"] = df["AmountInUSD"].apply(modified)

df["InvestmentType"].fillna("",inplace = True)

df = df[df["InvestmentType"] != ""]

a = df["InvestmentType"]

b = df["AmountInUSD"]

d ={}

for i in a.index:

    d[a[i]] = d.get(a[i],0) + b[i]

d['Seed Funding'] += d['SeedFunding']

d['Private Equity'] += d['PrivateEquity']

d['Crowd Funding'] += d['Crowd funding']

del d['SeedFunding']

del d['PrivateEquity']

del d['Crowd funding']

sum = 0

for i in d:

    sum += d[i]

for i in d:

    print(i,format(d[i]*100/sum,"0.2f"))

plt.pie(d.values(),labels = d.keys())

plt.axis("equal")

plt.show()
def modified(amount):

    return int(amount.replace(',',''))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["AmountInUSD"].fillna('0',inplace = True)

df["AmountInUSD"] = df["AmountInUSD"].apply(modified)

df["IndustryVertical"].fillna("",inplace = True)

df["IndustryVertical"].replace("eCommerce","Ecommerce",inplace = True)

df["IndustryVertical"].replace("ECommerce","Ecommerce",inplace = True)

df["IndustryVertical"].replace("ecommerce","Ecommerce",inplace = True)

df = df[df["IndustryVertical"] != ""]

a = df["IndustryVertical"]

b = df["AmountInUSD"]

d ={}

for i in a.index:

    d[a[i]] = d.get(a[i],0) + b[i]

d1 = sorted(d, key=d.get , reverse=True) 

d1 = d1[:5]

d2 = {}

for i in d1:

    d2[i] = d[i]

sum = 0

for i in d2:

    sum += d2[i]

for i in d2:

    print(i,format(d2[i]*100/sum,"0.2f"))
def modified(amount):

    return int(amount.replace(',',''))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["AmountInUSD"].fillna('0',inplace = True)

df["AmountInUSD"] = df["AmountInUSD"].apply(modified)

df["StartupName"].fillna("",inplace = True)

df["StartupName"].replace("Flipkart.com","Flipkart",inplace = True)

df["StartupName"].replace("Ola Cabs","Ola",inplace = True)

df["StartupName"].replace("Olacabs","Ola",inplace = True)

df["StartupName"].replace("Oyorooms","Oyo",inplace = True)

df["StartupName"].replace("OyoRooms","Oyo",inplace = True)

df["StartupName"].replace("OYO Rooms","Oyo",inplace = True)

df["StartupName"].replace("Oyo Rooms","Oyo",inplace = True)

df["StartupName"].replace("Paytm Marketplace","Paytm",inplace = True)

df = df[df["StartupName"] != ""]

a = df["StartupName"]

b = df["AmountInUSD"]

d = {}

for i in a.index:

    d[a[i]] = d.get(a[i],0) + b[i]

d1 = sorted(d, key=d.get , reverse=True) 

d1 = d1[:5]

for i in d1:

    print(i)
import pandas as pd

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df["InvestorsName"].fillna('',inplace = True)

df = df[df["InvestorsName"] != '']

df["InvestorsName"] = df["InvestorsName"].str.replace(",\xa0 ",", ")

df["InvestorsName"] = df["InvestorsName"].str.replace(", ","^")

df["InvestorsName"] = df["InvestorsName"].str.replace(",","^")

d = {}

for i in df["InvestorsName"]:

    if "^" in i:

        j = i.strip().split('^')

        for a in j:

            d[a] = d.get(a,0) + 1

    else:

        d[i] = d.get(i,0) + 1

d1 = sorted(d, key=d.get , reverse=True) 

print(d1[0],d[d1[0]])
import pandas as pd

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")  #reading data from startup_funding.csv file

df = startup.copy()  #creating a copy of startup dataframe...



df.dropna(subset = ["CityLocation"],inplace = True)  #removing row's having nan's in city location column

df["CityLocation"].replace("bangalore","Bangalore",inplace = True)  #replacing the wrong word with the correct one..

df["CityLocation"].replace("Delhi","New Delhi",inplace = True)



d1 = {}   #dictionary to maintain the number of fundings in the provided locations...



for i in df["CityLocation"]: #traversing through all the locations and maintaining the number of times the provided locations fetched...using if else..

    if "Bangalore" in i:

        d1["Bangalore"] = d1.get("Bangalore",0) + 1  

    elif "Mumbai" in i:

        d1["Mumbai"] = d1.get("Mumbai",0) + 1

    elif "New Delhi" in i:

        d1["New Delhi"] = d1.get("New Delhi",0) + 1

    elif "Noida" in i:

        d1["Noida"] = d1.get("Noida",0) + 1

    elif "Gurgaon" in i:

        d1["Gurgaon"] = d1.get("Gurgaon",0) + 1

        

cities = sorted(d1, key=d1.get , reverse=True)   #sorting the keys in reverse order(descending to aescending) according to the values...

fundings = []  



for i in cities:  #traversing through the cities...and created a new list of values...in sorted format 

    fundings.append(d1[i])

    

print(cities[0]) #printing the location having most number of fundings



plt.bar(cities,fundings,width = 0.5,color = "red" , edgecolor = "yellow")  #ploting the bar graph....cities vs no of fundings..

plt.xlabel("Locations")   #labeling x-axis

plt.ylabel("Number of Fundings")  #labeling y_axis

plt.xticks(rotation = 40) #rotating names in x-axis

plt.show()
import pandas as pd

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()



df.dropna(subset = ["InvestorsName"],inplace = True)  #removing row's having nan's in Investors name column..



d = {}  #creating a dictionary to maintain the number of times the investors name appeared...

for i in df["InvestorsName"].values:  #traversing through the investor names..to see the number of times the names appeared..

    if "," in i:  #if it contains multiple names ..then spliting it ..and traversing through each names separately..

        for j in i.strip().split(','):

            d[j.strip()] = d.get(j.strip(),0) + 1

    else:

        d[i.strip()] = d.get(i.strip(),0) + 1

d1 = sorted(d, key=d.get , reverse=True)[0:5]  #sorting the keys in reverse order(descending to aescending) according to the values...

for i in d1:  #printing the top 5 investors...funded maximum number of times...

    print(i)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()



df.dropna(subset = ["StartupName","InvestorsName"],inplace = True)  #removing row's having nan's in Investors name and startupnames columns..

df["StartupName"].replace("Flipkart.com","Flipkart",inplace = True)  #replacing the wrong word with the correct one..

df["StartupName"].replace("Ola Cabs","Ola",inplace = True)

df["StartupName"].replace("Olacabs","Ola",inplace = True)

df["StartupName"].replace("Oyorooms","Oyo",inplace = True)

df["StartupName"].replace("OyoRooms","Oyo",inplace = True)

df["StartupName"].replace("OYO Rooms","Oyo",inplace = True)

df["StartupName"].replace("Oyo Rooms","Oyo",inplace = True)

df["StartupName"].replace("Paytm Marketplace","Paytm",inplace = True)



#firstly ...created a dictionary ...for each investor names ... maintained a set..means each key(investor's name) having a value set(names of stratup's in which they invested)..

#set is taken as a value to avoid count of multiple investment in a single startup by an investor...

#in the set ..there are startup names in which investor's had invested...

#in case there are multiple investors for a single startup...used split function to split that ..and traversed through each name separately...

d = {}

for i in df.index:

    e = df["InvestorsName"][i].strip()

    if "," in e:

        for j in e.strip().split(','):

            if j.strip() in d:

                d[j.strip()].add(df["StartupName"][i].strip())

            else:

                s = set()

                d[j.strip()] = s

                d[j.strip()].add(df["StartupName"][i].strip())

    else:

        a = e.strip()

        if a in d: 

            d[a].add(df["StartupName"][i].strip())

        else:

            s = set()

            d[a] = s

            d[a].add(df["StartupName"][i].strip())

            

d1 = {}  #created a dictionary where key is investor's name and value is count of startup's in which they had invested..

for i in d:

    if i == "":

        continue

    d1[i] = len(d[i])

    

d2 = sorted(d1, key=d1.get , reverse=True)[0:5]  #sorting the keys according to there values in descending order..and taking the top 5 investor's among all..

for i in d2:

    print(i)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df.dropna(subset = ["InvestorsName"],inplace = True) #removing row's having nan's in Investors name column..

df["InvestorsName"].replace("Undisclosed Investors","",inplace = True) #replacing the undisclosed investors name by null values ..

df["InvestorsName"].replace("Undisclosed investors","",inplace = True)

df = df[(df["InvestmentType"] == "Seed Funding") | (df["InvestmentType"] == "Crowd Funding")]  #keeping only those rows having investment type seed funding and crowd funding..



d = {}  #creating a dictionary to maintain the number of times the investors name appeared...

for i in df["InvestorsName"].values:  #traversing through the investor names..to see the number of times the names appeared..

    if "," in i:  #if it contains multiple names ..then spliting it ..and traversing through each names separately..

        for j in i.strip().split(','):

            d[j.strip()] = d.get(j.strip(),0) + 1

    else:

        d[i.strip()] = d.get(i.strip(),0) + 1

        

del d[""] #deleting the NULL key from dictionary



d1 = sorted(d, key=d.get , reverse=True)[0:5]  #sorting the keys according to there values in descending order..and taking the top 5 investor's among all..

for i in d1:

    print(i)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

startup = pd.read_csv("../input/startup_funding.csv")

df = startup.copy()

df.dropna(subset = ["InvestorsName"],inplace = True)  #removing row's having nan's in Investors name column..

df["InvestorsName"].replace("Undisclosed Investors","",inplace = True)  #replacing the undisclosed investors name by null values ..

df["InvestorsName"].replace("Undisclosed investors","",inplace = True)

df = df[(df["InvestmentType"] == "Private Equity")]  #keeping only those rows having investment type private equity..



d = {}  #creating a dictionary to maintain the number of times the investors name appeared...

for i in df["InvestorsName"].values:  #traversing through the investor names..to see the number of times the names appeared..

    if "," in i:  #if it contains multiple names ..then spliting it ..and traversing through each names separately..

        for j in i.strip().split(','):

            d[j.strip()] = d.get(j.strip(),0) + 1

    else:

        d[i.strip()] = d.get(i.strip(),0) + 1

        

del d[""] #deleting the NULL key from dictionary



d1 = sorted(d, key=d.get , reverse=True)[0:5]  #sorting the keys according to there values in descending order..and taking the top 5 investor's among all..

for i in d1:

    print(i)