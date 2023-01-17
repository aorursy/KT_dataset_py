import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns



class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1', thousands = '.')

data.head()
print(color.BOLD + "Number of lines : " + color.END + str(data.count()['year']))

print(color.BOLD + 'States in the file : ' + color.END)

states = pd.unique(data['state'])

print(states)

print(color.BOLD + "Years from : " + color.END + str(data['year'].min())+" to "+str(data['year'].max()))
nb_fires_per_year = data.groupby(['year']).sum().reset_index()
sns.set()

plt.figure(figsize=(15,3))

locator = matplotlib.ticker.MultipleLocator()

plt.gca().xaxis.set_major_locator(locator)

formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")

plt.gca().xaxis.set_major_formatter(formatter)

ax = sns.lineplot(x="year", y="number", data=nb_fires_per_year, color='Red')

ax.set_ylabel('')    

ax.set_xlabel('')

ax.set_title("Number of fires by year in Brazil",fontdict={'fontsize': '17', 'fontweight' : 'bold'})
nb_fires_per_month = data.groupby(['month']).sum().reindex(['Janeiro','Fevereiro','Março','Abril','Maio','Julho','Julho','Agosto',

                                                           'Setembro','Outubro','Novembro','Dezembro']).reset_index()

nb_fires_per_month
sns.set()

plt.figure(figsize=(15,3))



ax = sns.barplot(x="month", y="number", data=nb_fires_per_month,palette="Reds")

ax.set_ylabel('')    

ax.set_xlabel('')

ax.set_title("Number of fires by month in Brazil since 1998",fontdict={'fontsize': '17', 'fontweight' : 'bold'})
fires_states_2017 = data.groupby('state').sum().reset_index()

fires_states_2017 = fires_states_2017.sort_values(by=['number'],ascending = True)

sns.set()

plt.figure(figsize=(30,3))



ax = sns.barplot(x="state", y="number", data=fires_states_2017,palette="Reds")

ax.set_ylabel('')    

ax.set_xlabel('')

ax.set_title("Number of fires by states in Brazil since 1998",fontdict={'fontsize': '17', 'fontweight' : 'bold'})
evolution_nb_fire = pd.DataFrame(columns =['state','mean_3_first_years','mean_3_last_years','diff','diff_percentage'])

nb_fires_per_year_and_state = data.groupby(['state','year']).sum()

for state in states:

    init_val = int((nb_fires_per_year_and_state.loc[state,1998]['number']+nb_fires_per_year_and_state.loc[state,1999]['number']+nb_fires_per_year_and_state.loc[state,2000]['number'])/3)

    final_val = int((nb_fires_per_year_and_state.loc[state,2015]['number']+nb_fires_per_year_and_state.loc[state,2016]['number']+nb_fires_per_year_and_state.loc[state,2017]['number'])/3)

    evolution_nb_fire = evolution_nb_fire.append({'state':state,

                                                    'mean_3_first_years':init_val,

                                                    'mean_3_last_years':final_val,

                                                    'diff':final_val-init_val,

                                                    'diff_percentage':((final_val-init_val)/init_val)*100},

                                                    ignore_index=True)



evolution_nb_fire = evolution_nb_fire.set_index('state').loc[['Acre','Amazonas','Mato Grosso','Roraima','Pará','Tocantins']]

evolution_nb_fire
fires_states = nb_fires_per_year_and_state.reset_index()



few_fires_states = fires_states[fires_states['state'].isin(['Acre','Amazonas','Roraima'])]

big_fires_states = fires_states[fires_states['state'].isin(['Mato Grosso','Pará','Tocantins'])]
sns.set()

plt.figure(figsize=(15,10))

locator = matplotlib.ticker.MultipleLocator()

plt.gca().xaxis.set_major_locator(locator)

formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")

plt.gca().xaxis.set_major_formatter(formatter)

ax = sns.lineplot(x="year", y="number", hue="state", data=few_fires_states)

ax.set_ylabel('')    

ax.set_xlabel('')

ax.set_title("Number of fires per year and states",fontdict={'fontsize': '17', 'fontweight' : 'bold'})
sns.set()

plt.figure(figsize=(15,10))

locator = matplotlib.ticker.MultipleLocator()

plt.gca().xaxis.set_major_locator(locator)

formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")

plt.gca().xaxis.set_major_formatter(formatter)

ax = sns.lineplot(x="year", y="number", hue="state", data=big_fires_states)

ax.set_title("Number of fires per year and states",fontdict={'fontsize': '17', 'fontweight' : 'bold'})

ax.set_ylabel('')    

ax.set_xlabel('')