#Import Library ต่างๆ
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# machine learning
import sklearn.datasets as datasets
# เช็คData ที่อยู่ใน folder input
print(os.listdir("../input"))
# อ่าน CSV files
population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')

# ตัวอย่างการ Drop Colums
temp_pop = population
temp_pop.drop(columns=['Country Name','Country Code', 'Indicator Name', 'Indicator Code'],axis =1, inplace=True)
# จำนวนทั้งหมดของประชากรตชในปีนั้นๆ
pop_sum=temp_pop.sum()
pop_sum=pd.DataFrame(pop_sum).reset_index()
pop_sum.columns= ['Year','Total Population']
pop_sum

#  Stack overflow method 
#การทำ Linear Plot ของประชากรทั้งโลกในแต่ละปี
plt.figure(figsize=(30,10))
plt.plot(pop_sum['Year'], pop_sum['Total Population'])
plt.title('Global Population from 1960 to 2016')
plt.xticks(np.arange(1960,2017))

plt.show()

#ลบ Columns ที่ไม่ต้องการ
temp_fert= fertility
temp_fert.head()
temp_fert.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_fert =temp_fert.dropna()
#ดูHead ของ Fertility
# new_fert.isnull().values.any() # no empty values 
new_fert.head()
# ค่าต่างๆและค่าเฉลี่ยของFert
fert_sum = new_fert.mean()
fert_sum = pd.DataFrame(fert_sum).reset_index()
fert_sum.columns=['Year', 'Fertility']
fert_sum.describe()
# fert_sum.plot()
plt.figure(figsize=(25,10))
plt.plot(fert_sum['Year'], fert_sum['Fertility'])
plt.xticks(np.arange(1960,2017))
plt.title('Fertility from 1960 to 2016')
plt.show()

df_life = life
# df_life.head()
df_life.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_life = df_life.dropna()
new_life.head()

# new_life.isnull().values.any()# no empty values 
life_mean =  new_life.mean()
life_mean = pd.DataFrame(life_mean).reset_index()
life_mean.columns= ['Year', 'Life expectancy']
# life_mean.plot()
life_mean.head()
plt.figure(figsize=(25,10))
plt.plot(life_mean['Year'], life_mean['Life expectancy'])
plt.xticks(np.arange(1960,2017))
plt.title('Life Expectancy from 1960 to 2016')
plt.show()

fert_sum.describe()

life_mean.describe()

pop_sum.describe()
# world_data = pd.merge(pd.merge(pop_sum, life_mean, on='Year'), fert_sum, on='Year')
test_data = pd.merge(pop_sum,life_mean, on='Year')
world_data = pd.merge(test_data, fert_sum, on='Year')
world_data.columns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
world_data[['Total Population', 'Life expectancy', 'Fertility']] = scaler.fit_transform(world_data[['Total Population', 'Life expectancy', 'Fertility']])
world_data
# Import CSV file
population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')
    
# เช็ค Missing Values เพื่อน Clean data
population.head()
pop_year = population.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
pop_year= pop_year.T
# population.head()
pop_year.head()
pop_year.shape
# pop_year.isnull().values.any()

t = pop_year.isnull().apply(sum, axis=0)
t

pop_year = pop_year.fillna(pop_year.mean())        
pop_year.isnull().values.any() # The answer is true
pop_year.shape
fertility = pd.read_csv('../input/fertility_rate.csv')

fertility=fertility.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
fert_year = fertility

        
fert_year= fert_year.dropna()
fert_year


fert_temp = fert_year

fert_temp= fert_temp.dropna()
# fert_temp=fert_temp.drop('mean of year', axis =1)
fert_temp

country_names = pd.DataFrame(fert_temp['Country Name'])
country_names
fert_ignore =pd.DataFrame(fert_temp.iloc[:,:]) #Ignores the names of the country

fert_ignore
if 'mean of year' in fert_ignore.columns:
    fert_ignore =fert_ignore.drop('mean of year',axis =1)
fert_ignore.T
fert_mean = fert_ignore.mean(axis =0)
type(fert_mean)
fert_mean

idx= 0
fert_add = fert_ignore.T
fert_add.insert(loc=idx, column='mean of year', value=fert_mean)
fert_add.columns[0]

fert_add = fert_add.T
fert_add.iloc[1]
column_drop = fert_add.columns[1:41] #1960 ถึง 1999
column_interest= fert_add.columns[41:] #2000 ถึง 2016
mil_fert = fert_add.drop(column_drop,axis=1) #fertility data since 2000

mil_fert.iloc[:,1:]#.mean().mean() = 3.0017
mil_filter = mil_fert[mil_fert[column_interest]>= mil_fert.iloc[:,1:].mean().mean()]

count_try= mil_filter.drop('Country Name', axis=1).dropna()
count_try

# ค่า Mean เก่ามีค่าประมาณ 3.0017 เราจึงกำกับค่าใหม่ให้สูงกว่าเดิม เพื่อลดจำนวนลง
more_filter = count_try[count_try[column_interest]>=5.0]
more_filter= more_filter.dropna()
more_filter
# ปรับเพิ่มเป็น 5.5
filter_six = more_filter[more_filter[column_interest]>=5.5]
filter_six = filter_six.dropna()
filter_six
country_indices=filter_six.index.values
country_indices
type(country_indices)
insert_names = country_names[country_names.index.isin(country_indices)]
insert_names['Country Name']

idx= 0
country_fert = filter_six
# country_fert.insert(loc=idx, column='Country Name', value=insert_names)
country_fert
country_fert.plot(figsize= (10,10), title= '10 countries with highest fertility rate 2000-2016')
fertility[column_interest].dropna().plot(kind='line',figsize=(10,10), title = 'World fertility 2000-2016')
population[column_interest].plot(figsize=(10,10), title='World population 2000-2016')
population = population = pd.read_csv('../input/country_population.csv')
population= population.drop(['Country Code','Indicator Name','Indicator Code'], axis=1)
world_16 = population['2016'].sum()

country_pop= population[column_interest].iloc[country_indices]
ten_16= country_pop['2016'].sum()
country_pop
#population of the 10 countries in 2016 was 414258372.0

idx= 0
country_pop.insert(loc=idx, column='Country Name', value=insert_names)
country_pop.plot(figsize= (10,10), title = 'population of 10 mos fertile countries')



percent = np.multiply(np.divide(ten_16,world_16),100)
print('The top 10 most fertile countries in 2016, form ', percent,'%  of the world population in 2016')
# percent 
population = population.transpose()
population
X = population["Country Name"].dropna()
y = population["Aruba"].dropna()
X_train, X_test, y_train, y_test = train_test_split(X[:1459], y[:1459], test_size = 1)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)

plt.scatter(X_test, y_test, color = "black")
plt.plot(X_test, predictions, color = "blue")
plt.title("Regression Model")
plt.xlabel("Year")
plt.ylabel('Population')
model.summary()