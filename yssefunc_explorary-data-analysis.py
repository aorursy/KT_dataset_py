#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read dataset
norway = pd.read_csv('../input/norway_new_car_sales_by_model.csv',encoding="latin-1")
# check the head of data
norway.head()
# Display all informations
norway.info()
# Missing Data Check
all_data_na = (norway.isnull().sum() / len(norway)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
#fixing rows that have some characters in front of Mercedes-Benz
norway["Make"]=norway["Make"].str.replace('\xa0Mercedes-Benz ','Mercedes-Benz ')
#the most selling car make with pie chart visualization
makes = norway.groupby(['Make']).count().index
sizes = norway.groupby(['Make']).count()['Quantity']


fig, ax = plt.subplots(figsize=(10,10))
ax.pie(sizes, labels=makes, autopct='%1.1f%%',
        shadow=False,  startangle=90)
ax.axis('equal')  
sizes
plt.tight_layout()
#Let’s print yearly total car sales.
norway.Make=norway.Make.str.lower()
norway.Model=norway.Model.str.lower()
monthly_total_sales=norway.pivot_table("Quantity",index="Year",aggfunc="sum")
print(monthly_total_sales.mean(axis=1))

#Visualization above the code
monthly_total_sales.mean(axis=1).plot.line()
#Let’s print monthly total car sales.
norway.Make=norway.Make.str.lower()
norway.Model=norway.Model.str.lower()
monthly_total_sales=norway.pivot_table("Quantity",index="Month",columns="Year",aggfunc="sum")
print(monthly_total_sales.mean(axis=1))
#Visualization above the code
monthly_total_sales.mean(axis=1).plot.line()
#Calculate total amount of the sales for each manufacturer from 2007 to 2017. Find the top-10 manufacturers based on the total sale.
make_total = norway.pivot_table("Quantity",index=['Make'],aggfunc='sum')
top10make=make_total.sort_values(by='Quantity',ascending=False)[:10]
print(top10make)
#Visualization above the code
top10make.plot.bar()
#Calculate  which models has highest yearly fluncations
maketotal_1 = norway.pivot_table(values='Quantity',index=['Month','Model','Make'],aggfunc=np.std)
df1 = maketotal_1.reset_index().dropna(subset=['Quantity'])
df2 = df1.loc[df1.groupby('Make')['Quantity'].idxmax()]
for index,row in df2.iterrows():
    print("For Manufacturer",row['Make'],"model",row['Model'],"has the highest yearly fluncation.")

# Comparing Car models with Percentage shares
import seaborn as sns

car_list = list(norway['Make'].unique())
car_selling_quantity = []

for i in car_list:
    x = norway[norway['Make']==i]
    area_car_rate = sum(x.Pct)/len(x)
    car_selling_quantity.append(area_car_rate)

data = pd.DataFrame({'car_list': car_list,'car_selling_quantity':car_selling_quantity})

new_index = (data['car_selling_quantity'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['car_list'], y=sorted_data['car_selling_quantity'])
plt.xticks(rotation= 90)
plt.xlabel('Car Models')
plt.ylabel('Percentage share')
plt.title('Percentage share in Norway')