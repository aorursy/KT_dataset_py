import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



%matplotlib inline



print("Setup Complete")
fire_filepath = "../input/forest-fires-in-brazil/amazon.csv"



fire_data = pd.read_csv(fire_filepath, parse_dates=True, encoding = "cp1252")



fire_data.head()
fire_data.loc[fire_data['month'] == 'Janeiro','month']='January'

fire_data.loc[fire_data['month'] == 'Fevereiro','month']='February'

fire_data.loc[fire_data['month'] == 'Março','month']='March'

fire_data.loc[fire_data['month'] == 'Abril','month']='April'

fire_data.loc[fire_data['month'] == 'Maio','month']='May'

fire_data.loc[fire_data['month'] == 'Junho','month']='June'

fire_data.loc[fire_data['month'] == 'Julho','month']='July'

fire_data.loc[fire_data['month'] == 'Agosto','month']='August'

fire_data.loc[fire_data['month'] == 'Setembro','month']='September'

fire_data.loc[fire_data['month'] == 'Outubro','month']='October'

fire_data.loc[fire_data['month'] == 'Novembro','month']='November'

fire_data.loc[fire_data['month'] == 'Dezembro','month']='December'
fire_data2=fire_data.loc[(fire_data['year'] == 1998)&(fire_data['month'] == 'July')]



fire_data2.head()
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(fire_data2['state'], fire_data2['number'], width=0.95)

ax.set_xticklabels(fire_data2['state'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil for July 1998')

ax.set_xlabel('State')

ax.set_ylabel('Number')
f, ax = plt.subplots(figsize=(16, 6))

sns.set(style="whitegrid")

sns.set_color_codes("pastel")



ax = sns.barplot(x='state', y='number', data=fire_data2, color="b", ci=None)

ax.set_xticklabels(fire_data2['state'], rotation = 45, ha="right")
states=fire_data['state'].unique()

print(states)
years=fire_data['year'].unique()

print(years)
fire_data_Acre=fire_data.loc[fire_data['state'] == 'Acre']



fire_data_Acre.shape[0]
total=[]

state_total=0

data={}

years_data=[]

year_num=0



data['state']=states



for j in range(len(states)):   

    for i in range(len(fire_data.index)):

        if fire_data.at[i,'state'] == states[j]:

            state_total += fire_data.at[i,'number']



    total.append(state_total)

    state_total=0

    



for y in range(len(years)): 

    for s in range(len(states)):  

        for i in range(len(fire_data.index)):

            if (fire_data.at[i,'year'] == years[y]) and (fire_data.at[i,'state'] == states[s]):

                year_num += fire_data.at[i,'number']

                

        years_data.append(year_num)



        year_num=0

    

    data[years[y]]=years_data

    years_data=[]





data['total']=total



states_data = pd.DataFrame(data)

states_data
fire_data.loc[(fire_data['year'] == 1998)&(fire_data['state'] == 'Tocantins')]['number'].sum()
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(states_data['state'], states_data['total'], width=0.95)

ax.set_xticklabels(states_data['state'], rotation = 45, ha="right")



ax.set_title('Total Forest Fire in Brazil by state')

ax.set_xlabel('State')

ax.set_ylabel('Number of forest fires')
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(states_data['state'], states_data[2016], width=0.95)

ax.set_xticklabels(states_data['state'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil in 2016 by state')

ax.set_xlabel('State')

ax.set_ylabel('Number of forest fires')
a=states_data.loc[states_data['state'] == 'Acre'].values.tolist()

a=a[0][1:-1]

    

list_of_tuples = list(zip(years,a))

new_years_data = pd.DataFrame(list_of_tuples, columns = ['Year', 'State'])



new_years_data['Year']=new_years_data['Year'].astype('str')
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(new_years_data['Year'], new_years_data['State'], width=0.95)

ax.set_xticklabels(new_years_data['Year'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil in Acre')

ax.set_xlabel('Year')

ax.set_ylabel('Number of forest fires')
states_data_years=states_data.transpose()



new_header = states_data_years.iloc[0]

states_data_years = states_data_years[1:]

states_data_years.columns = new_header



states_data_years.drop(states_data_years.tail(1).index,inplace=True)



states_data_years.reset_index(drop=False, inplace=True)

states_data_years['index']=states_data_years['index'].astype('str')

states_data_years=states_data_years.rename(columns={"index": "Year"}, errors="raise")

states_data_years
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(states_data_years['Year'], states_data_years['Mato Grosso'], width=0.95)

ax.set_xticklabels(states_data_years['Year'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil in Mato Grosso')

ax.set_xlabel('Year')

ax.set_ylabel('Number of forest fires')
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(states_data_years['Year'], states_data_years['Sergipe'], width=0.95)

ax.set_xticklabels(states_data_years['Year'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil in Sergipe')

ax.set_xlabel('Year')

ax.set_ylabel('Number of forest fires')
fig, ax = plt.subplots()

plt.figure(figsize=(16,6))

plt.rcParams["figure.figsize"] = [20,8]



ax.bar(states_data_years['Year'], states_data_years['Mato Grosso'], width=0.95, color='r', align='center')

ax.bar(states_data_years['Year'], states_data_years['Acre'], width=0.95, color='g', align='center')

ax.bar(states_data_years['Year'], states_data_years['Sergipe'], width=0.95, color='b', align='center')

ax.set_xticklabels(states_data_years['Year'], rotation = 45, ha="right")



ax.set_title('Forest Fire in Brazil')

ax.set_xlabel('Year')

ax.set_ylabel('Number of forest fires')
f, ax = plt.subplots(figsize=(16, 6))

sns.set(style="whitegrid")

sns.set_color_codes("pastel")



ax = sns.barplot(x='Year', y='Mato Grosso', data=states_data_years, color="b", ci=None)

ax = sns.barplot(x='Year', y='Acre', data=states_data_years, color="r", ci=None)

ax = sns.barplot(x='Year', y='Sergipe', data=states_data_years, color="g", ci=None)

ax.set_xticklabels(states_data_years['Year'], rotation = 45, ha="right")
