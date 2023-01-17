# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
%matplotlib inline
data = pd.read_csv('/kaggle/input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')
data.shape
data
data.describe()
data.dtypes
data.columns
data = data.astype({
    'date':np.datetime64
})
def get_periodo_resampled(df, periodo, group_by="state"):
    """Método que retorna um dataframe com o periodo e grupo especificado

    Args:
        df (pandas.DataFrame): dataframe de incidentes
        periodo (string): "M" etc.
        group_by (string, optional): pode ser qualquer nome de coluna do df. Defaults to "state".

    Returns:
        [pandas.DataFrame]: dataframe resampled
    
    periodo opções:
    B         business day frequency
    C         custom business day frequency (experimental)
    D         calendar day frequency
    W         weekly frequency
    M         month end frequency
    SM        semi-month end frequency (15th and end of month)
    BM        business month end frequency
    CBM       custom business month end frequency
    MS        month start frequency
    SMS       semi-month start frequency (1st and 15th)
    BMS       business month start frequency
    CBMS      custom business month start frequency
    Q         quarter end frequency
    BQ        business quarter endfrequency
    QS        quarter start frequency
    BQS       business quarter start frequency
    A         year end frequency
    BA, BY    business year end frequency
    AS, YS    year start frequency
    BAS, BYS  business year start frequency
    BH        business hour frequency
    H         hourly frequency
    T, min    minutely frequency
    S         secondly frequency
    L, ms     milliseconds
    U, us     microseconds
    N         nanoseconds

    group_by opções:
    
    ['incident_id', 'date', 'state', 'city_or_county', 'address', 'n_killed',
       'n_injured', 'incident_url', 'source_url',
       'incident_url_fields_missing', 'congressional_district', 'gun_stolen',
       'gun_type', 'incident_characteristics', 'latitude',
       'location_description', 'longitude', 'n_guns_involved', 'notes',
       'participant_age', 'participant_age_group', 'participant_gender',
       'participant_name', 'participant_relationship', 'participant_status',
       'participant_type', 'sources', 'state_house_district',
       'state_senate_district']
    """    
    df['date'] = df['date'].astype(np.datetime64)
    df = df.set_index(df.date)
    resampled = df.resample(periodo, on='date')
    lista = []
    for name, group in resampled:
        grouped =  group.groupby([group_by])
        grouped = pd.DataFrame(grouped.sum()).reset_index()
        grouped['date'] = name
        if not grouped.empty:
            lista.append(grouped)

    # adicional code to return to normal dataframe
    dataframe_concated = pd.DataFrame([])
    for df in lista:
        dataframe_concated = pd.concat([dataframe_concated, df], ignore_index=True)

    return dataframe_concated
data_resampled = get_periodo_resampled(df=data, periodo='M', group_by='state')
# pegando os valores únicos da coluna
data_resampled.date.unique()
def filtrar_date(df, date):
    date_ = datetime.date.fromisoformat(date)
    return df[df.date.map(lambda x: x == date_)]
df_filtered = filtrar_date(data_resampled, "2018-02-28")
df_filtered.iloc[0]
x = df_filtered.state.values # pegar somente primeira letra
y = df_filtered.n_killed.values
x.sort()
y.sort()
xlabel = df_filtered.state
n_labels = range(df_filtered.shape[0])
plt.xticks( n_labels, xlabel, fontsize=14,rotation=65)  # Set text labels and properties.
plt.suptitle('N Killed por estado')
plt.title("Em " + str(df_filtered.iloc[0].date))
plt.legend(loc='best')

# plt.subplots_adjust(left=0.4, right=0.5)
plt.subplots_adjust(left=0, bottom=None, right=3, top=None, wspace=30, hspace=0.5)
plt.bar(x, y) ## talvez esse gráfico seja melhor em barras
plt.plot(x, y)
df_grouped_by_state = data_resampled.groupby("state")
df_alabama = df_grouped_by_state.get_group("Alabama")
x = df_alabama.date
y = df_alabama.n_killed
plt.xticks(rotation=90)  # Set text labels and properties.
plt.suptitle('N Killed no Alabama')
plt.plot(x, y)
plt.plot(x,y, 'b--')
plt.scatter(x, y)
plt.legend(loc='best')
plt.show()
df_sumed = data.groupby("n_guns_involved").sum()

x = df_sumed.index.values[0:10]

y = df_sumed.n_injured.values[0:10]


plt.xticks(rotation=90)  # Set text labels and properties.
plt.suptitle('N guns vs  N feridas')
plt.plot(x, y)
coluna = "n_killed"
df_sumed = data.groupby("gun_type").sum()
df_sumed = df_sumed.sort_values(coluna, ascending=False)
x = df_sumed.index.values[0:10]
y = df_sumed[coluna].values[0:10]

plt.xticks(rotation=90)  # Set text labels and properties.
plt.suptitle('Gun type vs  N killed')
plt.bar(x, y)
coluna = "n_injured"
df_sumed = data.groupby("gun_type").sum()
df_sumed = df_sumed.sort_values(coluna, ascending=False)
x = df_sumed.index.values[0:10]
y = df_sumed[coluna].values[0:10]

plt.xticks(rotation=90)  # Set text labels and properties.
plt.suptitle('Gun type vs  N injured')
plt.bar(x, y)
data_anual = get_periodo_resampled(df=data, periodo='A', group_by='state')
data_anual = data_anual.sort_values("date")

group_states = data_anual.groupby("state")

df_alabama = group_states.get_group('Alabama') # testando pro Alabama

df_alabama.index = df_alabama.date # mudando o index para date
df_alabama.n_killed.plot(color='#17a589', label=' numero de mortos')

df_alabama.n_injured.plot(label=' numero de feridos')

df_alabama.n_guns_involved.plot(label=' numero de armas envolvidas')
plt.title('Alabama dados')
plt.legend(loc=0)
# pegando lista de estados que diminuitam de 2017 pra 2018
lista = []

for name, group in group_states:
    size = group.shape[0]
    if group.n_killed.iloc[size-1] < group.n_killed.iloc[size-2]:
        group['state'] = name
        lista.append(group)

len(lista)

# acho que houve uma subnotificação a partir de 2018, todos os estados diminuiram o número de acidentes

lista_2 = []

for name, group in group_states:
    size = group.shape[0]
    if group.n_killed.iloc[size-3] < group.n_killed.iloc[size-2]:
        group['state'] = name
        lista.append(group)

len(lista_2)
data_anual = data_anual.sort_values("date")

group_states = data_anual.groupby("state")

df_alabama = group_states.get_group('Wyoming') # testando pro Wyoming

df_alabama.index = df_alabama.date # mudando o index para date
df_alabama.n_killed.plot(color='#17a589', label=' numero de mortos')

df_alabama.n_injured.plot(label=' numero de feridos')

df_alabama.n_guns_involved.plot(label=' numero de armas envolvidas')
plt.title('Wyoming dados')
plt.legend(loc=0)
data.columns
data.participant_age
groups = data.groupby("participant_gender")
data.participant_gender
data_anual = get_periodo_resampled(df=data, periodo='A', group_by='state')
df_grouped = data_anual.groupby('date') # agrupando por date
df_2017 = df_grouped.get_group("2017-12-31") # pegando somente o ano de 2017
# Vendo o maximo e mínimo dos estado
df_2017.index = df_2017.n_guns_involved
df_2017.sort_index()
data_anual = get_periodo_resampled(df=data, periodo='A', group_by='state')
df_alabama = group_states.get_group('Alabama') # testando pro Alabama
df_california = group_states.get_group('California') # testando pro California
df_california.index = df_california.date
df_california.n_killed.plot(label=' California')
df_alabama.index = df_alabama.date
df_alabama.n_killed.plot(label=' Alabama')

plt.title('N de mortes')
plt.legend(loc=0)
df_california = group_states.get_group('California') # testando pro California
df_alabama = group_states.get_group('Alabama') # testando pro Alabama
df_Illinois = group_states.get_group('Illinois') # testando pro Illinois
df_Florida = group_states.get_group('Florida') # testando pro Florida
plt.figure(figsize=(30, 8))

plt.subplot(spec2[0, 0])
df_alabama.n_killed.plot(label=' Alabama')
plt.legend(loc=0)
plt.subplot(spec2[0, 1])
df_california.n_killed.plot(label=' California')
plt.legend(loc=0)
plt.subplot(spec2[1, 0])
df_Illinois.n_killed.plot(label=' Illinois')
plt.legend(loc=0)
plt.subplot(spec2[1, 1])
df_Florida.n_killed.plot(label=' Florida')
plt.suptitle('N de mortes por estado e tempo')
plt.legend(loc=0)
plt.show()
df_alabama.n_killed.plot(label=' Alabama')
df_california.n_killed.plot(label=' California')
df_Illinois.n_killed.plot(label=' Illinois')
df_Florida.n_killed.plot(label=' Florida')
plt.suptitle('N de mortes por estado e tempo')
plt.legend(loc=0)
plt.show()
#Mostrar o numero de incidentes por estado
state_crime = data['state'].value_counts().head(30)
plt.figure(figsize=(30, 8))
# state_crime
plt.pie(state_crime, labels=state_crime.index,autopct='%1.1f%%', shadow=True)
#As 10 principais cidades com maior número de incidentes
plt.figure(figsize=(30, 8))
top_10_city = data['city_or_county'].value_counts().keys().tolist()[0:9]
top_10_values = data['city_or_county'].value_counts().tolist()[0:9]
x=top_10_city
y=top_10_values
plt.bar(x,y)
#  Por Ano - contagem de incidentes criminais
Yearly_incidents_label = data['date'].value_counts().keys()
Yearly_incidents_count = data['date'].value_counts().tolist()

x=Yearly_incidents_label
y=Yearly_incidents_count
plt.figure(figsize=(30, 3))
plt.scatter(x, y)
groups = data.groupby("participant_gender")
data.participant_gender
data['n_female'] = data.participant_gender.map(lambda x: str(x).count("Female"))
data['n_male'] = data.participant_gender.map(lambda x: str(x).count("Male"))
data['n_male']
data['n_female']
data_resampled = get_periodo_resampled(df=data, periodo='A', group_by='state')
groups = data_resampled.groupby("date")
keys = groups.groups.keys()
yearly_data_state = data[["state"]]
lista = [groups.get_group(key) for key in keys]
lista[0]
D = list(keys)[0]
D.year
for index, df in enumerate(lista):
    result = df.sum(axis=0)
    result.n_female
    
    if (index >= 3):
        plt.subplot(2, 3, index+1)
    else:
        plt.subplot(2, 3, index+1)

    plt.title(list(keys)[index].year)
    plt.pie([result.n_female, result.n_male], labels=["Female", "Male"], autopct='%1.1f%%', shadow=True )
df2 = df_2017.sort_values([ 'n_killed'])
df2
df_california = group_states.get_group('California') # testando pro California
df_california
# for index, df in (df_california):
#     result = df.sum(axis=0)
#     result.n_female
    
#     if (index >= 3):
#         plt.subplot(2, 3, index+1)
#     else:
#         plt.subplot(2, 3, index+1)

#     plt.title(list(keys)[index].year)
#     plt.pie([result.n_female, result.n_male], labels=["Female", "Male"], autopct='%1.1f%%', shadow=True )
data_anual = get_periodo_resampled(df=data, periodo='M', group_by='state')
D.month
df_jan = data_anual[data_anual.date.map(lambda x: x.month == 1)]
df_fev = data_anual[data_anual.date.map(lambda x: x.month == 2)]
df_mar = data_anual[data_anual.date.map(lambda x: x.month == 3)]
df_jan.groupby("date").sum() # pegando o total de todos os estados juntos
df_fev.groupby("date").sum() # pegando o total de todos os estados juntos
df_mar.groupby("date").sum() # pegando o total de todos os estados juntos
plt.figure(figsize=(30, 8))
ano = df_mar.date
n_mortos = df_mar['n_killed']
plt.bar(ano,n_mortos)
df_jan.groupby("state").get_group("Alabama")
df_jan.groupby("state").get_group("California")