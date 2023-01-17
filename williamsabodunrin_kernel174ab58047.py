import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sales_data = pd.read_csv( '../input/supermarket-sales/supermarket_sales - Sheet1.csv' , parse_dates=['Date', 'Time'])

det = sales_data.count()

details = pd.DataFrame(det)

details.rename(columns={0: 'counts per column'}, inplace= True)

details['dtypes per column']= sales_data.dtypes

details['unique_values']= sales_data.nunique()

details
sales_data['Product line'].unique()
sales_data
#we print out a list of the various categories in the product line

print(list(sales_data['Product line'].unique()))

#write a function to rename the categories, they are too long for plots

def rename(col):

    if col == 'Health and beauty':

        return 'H & B'

    elif col == 'Electronic accessories':

        return 'E'

    elif col == 'Home and lifestyle':

        return 'H & L'

    elif col == 'Sports and travel':

        return 'S & T'

    elif col == 'Food and beverages':

        return 'F & B'

    elif col == 'Fashion accessories':

        return 'F'

sales_data['Product line(abbr)']= sales_data['Product line'].apply(rename)



#now we use the groupby method to group the dataset by Product line

qdata= sales_data.groupby('Product line')['Quantity'].sum().reset_index()

plt.figure(figsize=(12,8))

sns.barplot(x= 'Product line', y= 'Quantity', data= qdata )
qdata1 = sales_data.groupby(['Product line(abbr)', 'Branch'])['Quantity'].sum().reset_index()

plt.figure(figsize=(12,8))

sns.catplot(x= 'Product line(abbr)', y= 'Quantity', col = 'Branch', kind = 'bar', data= qdata1 )

sales_data.filter(['Product line', 'Product line(abbr)']).drop_duplicates().reset_index(drop = True)
#The X axis will be to cumbersome so we can use the abbr versions of the product line

def frequency(data):

    output = data.count()

    return output

qdata2 = sales_data.groupby(['Product line(abbr)', 'Branch', 'Gender'])['Quantity'].agg([frequency, sum]).reset_index()

qdata2

plt.figure(figsize=(16,10))

for y in [['sum', 'husl'], ['frequency', 'bright']]:

    sns.catplot(x= 'Product line(abbr)', y= y[0], hue = 'Gender', col = 'Branch', kind = 'bar',palette =y[1], data= qdata2 )

sales_data.filter(['Product line', 'Product line(abbr)']).drop_duplicates().reset_index().drop('index', axis =1)

qdata3= sales_data.groupby(['Product line(abbr)', 'Branch'])['Total'].sum().reset_index()

qdata3['Quantity']=sales_data.groupby(['Product line(abbr)', 'Branch'])['Quantity'].sum().reset_index().Quantity

sns.catplot(x= 'Product line(abbr)', y= 'Total', col = 'Branch', kind = 'bar', data= qdata3 )
sns.relplot(x= 'Quantity', y= 'Total', col = 'Branch', kind = 'scatter', data= qdata3 )
sns.scatterplot(x= 'Unit price', y= 'Quantity', data= sales_data)
sns.boxplot(x= 'Payment', y = 'Total', data=sales_data)

sales_data
sales_data2= sales_data.groupby('Payment')['Total'].sum()

payment_type= pd.DataFrame(sales_data2)

payment_type['frequency']= sales_data.groupby('Payment').size()

payment_type ['Quantity']= sales_data.groupby('Payment')['Quantity'].sum()

payment_type
payment_type.plot(kind='bar', figsize=(12,6))
payment = sales_data.groupby(['Payment', 'Branch'])['Total'].agg([frequency, sum]).reset_index()

payment['Quantity']=  sales_data.groupby(['Payment', 'Branch'])['Quantity'].sum().reset_index().Quantity

payment

sns.catplot(x='Payment', y= 'sum', col= 'Branch', kind ='bar', data = payment, palette='Accent_r')

sns.set_style('whitegrid')

payment
for y in ['sum', 'Quantity']:

    sns.relplot(x='frequency', y=y, kind='line', col='Branch', data= payment )
CL= sales_data.groupby(['Customer type', 'Payment',])['Total'].sum().reset_index()

CL
plt.figure(figsize=(10,6), facecolor= 'w', dpi=100)

sns.barplot(x= 'Customer type', y='Total', hue='Payment', data=CL, capsize = 0.1, palette='inferno_r')
CL1= sales_data.groupby(['Customer type', 'Payment','Branch'])['Total'].sum().reset_index()

for hue in ['Payment', None]:

    plt.figure(figsize=(10,6), facecolor= 'w', dpi=100)

    sns.catplot(x= 'Customer type', y='Total', hue=hue, col='Branch', kind='bar', data=CL1, capsize = 0.1)
#First we create three columns, month, day and hour

sales_data['month']= sales_data['Date'].dt.month

sales_data['day']= sales_data['Date'].dt.day

sales_data['hour']= sales_data['Time'].dt.hour

dates= sales_data.sort_values(by= 'Date')

dates
plt.figure(figsize=(16,10), facecolor= 'w', dpi=100)

sns.lineplot(x='Date', y='Total', data=dates)
dates1= dates.groupby(['Date', 'Branch'])['Total'].sum().reset_index()

dates1
for branch in [['A','Blues' ], ['B', 'bone_r'], ['C', 'Greens']]:

    datedata = dates1[dates1['Branch']== branch[0]]

    plt.figure(figsize=(7,3), facecolor= 'w', edgecolor = 'r', dpi=100)

    plt.xticks(rotation = -45) 

    sns.lineplot(x='Date', y= 'Total', hue = 'Branch', data= datedata, palette= branch[1], ci= None)

    sns.set_style('darkgrid')

    #plt.plot(color=branch[1])


sales1= sales_data.groupby(['month', 'Branch'])['Total'].sum().reset_index()

sales2 = sales_data.groupby(['month', 'Branch'])['Total'].count().reset_index()



sales1['count']= sales2['Total']

sales1
time_data=sales_data.groupby(['day', 'month', 'Branch'])['Total'].sum().reset_index()

time_data['People/day']= sales_data.groupby(['day', 'month', 'Branch'])['Total'].count().reset_index().Total

time_data
number =[1,2,3]

for month in number:

    data = time_data[time_data['month']== month]

    plt.figure(figsize=(10,3), facecolor= 'w', dpi=100)

    sns.relplot(x='day', y='Total', kind='line', col= 'Branch', col_wrap=3, data=data)

    
number =[1,2,3]

for month in number:

    data = time_data[time_data['month']== month]

    plt.figure(figsize=(10,3), facecolor= 'w', dpi=100)

    sns.relplot(x='People/day', y='Total', kind='scatter', col='Branch', col_wrap=3,data=data)
time_data['quantity']=sales_data.groupby(['day', 'month', 'Branch'])['Quantity'].sum().reset_index().Quantity

time_data.corr()['Total']