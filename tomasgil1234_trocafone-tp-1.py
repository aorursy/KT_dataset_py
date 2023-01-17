import pandas as pd
import numpy as np
%matplotlib nbagg
%pylab
# ===> Importamos el dataset de Trocafone
df = pd.read_csv('../input/events.csv')
# ===> Parsear fechas de string a datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

# ===> Para saber cuantos NaN hay al hacer value_counts()
df = df.fillna("-")

# ===> Mostrar todas las columnas del data set
pd.set_option('display.max_columns', None) 
# ===> Vemos como esta conformado el dataset
df.head()
conversions = df[df.event == 'conversion'].set_index("timestamp")
views = df[df.event == 'viewed product'].set_index("timestamp")
figure()
title("Conversions and views by day")
conversions_by_date = conversions.resample("D").event.count()
conversions_by_date.plot()
ylabel('Numbers of conversions')

views_by_date = views.resample("D").event.count()
views_by_date.plot(secondary_y=True) 
ylabel('Views by date')
tight_layout()


figure()
title("Conversion percentage by day")
(conversions_by_date / views_by_date * 100).plot()
ylabel('Conversion percentage')
tight_layout()
# map to the weekday names: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#anchored-offsets

import calendar

def fecha_a_dia(fecha):
    return fecha.weekday()

figure(figsize= (8,8))
subplot(311)
title("Conversions by day of week")

conversions_by_date = conversions.groupby(fecha_a_dia).event.count()
conversions_by_date.plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" ,fontsize=13) 
xticks([0,1,2,3,4,5,6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']) 
ylabel('Conversions')

subplot(312)
title("Views by day of week")
views_by_date = views.groupby(fecha_a_dia).event.count()
views_by_date.plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" ,fontsize=13) 
xticks([0,1,2,3,4,5,6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']) 
ylabel('Views')


subplot(313)
title("Conversion percentage by week")
(conversions_by_date / views_by_date * 100).plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" ,fontsize=13) 
xticks([0,1,2,3,4,5,6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']) 

tight_layout()

df_number_convertions_per_user = pd.DataFrame(df[(df['event']=='conversion')]['person'].value_counts())

number_user_with_convertions = len(df_number_convertions_per_user.index)
print(number_user_with_convertions)
number_users_without_convertions = len(df[~((df['event']=='conversion'))]['person'].unique())
print(number_users_without_convertions)
sizes = [number_user_with_convertions, number_users_without_convertions]
labels = ['Usuarios CON conversiones', 'Usuarios SIN conversiones']

fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
ax1.set_title("Usuarios con al menos una compra")
plt.show()
convs_per_person = conversions.person.value_counts()
convs_distr = convs_per_person.value_counts().sort_index()

figure()
convs_distr.plot(kind="bar", rot=0, color=(0.1, 0.1, 0.1, 0.2), edgecolor="b")
title('Number of conversions per user')
xlabel('Number of conversions')
ylabel('Number of users')
number_users_with_more_than_two_convertions = len(df_number_convertions_per_user[df_number_convertions_per_user['person']>2].index)
print(number_users_with_more_than_two_convertions)
number_users_with_two_convertions = len(df_number_convertions_per_user[df_number_convertions_per_user['person']==2].index)
print(number_users_with_two_convertions)
number_users_with_only_one_convertion = len(df_number_convertions_per_user[df_number_convertions_per_user['person']<2].index)
print(number_users_with_only_one_convertion)
sizes_2 = [number_users_with_more_than_two_convertions, number_users_with_two_convertions, number_users_with_only_one_convertion]
labels_2 = ['Usuarios con mas de 2 conversiones', 'Usuarios con 2 conversiones', 'Usuarios con solo una conversion']
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.pie(sizes_2, labels=labels_2, autopct='%1.1f%%', shadow=True)
ax2.axis('equal')
ax2.set_title("Usuarios y conversiones")
plt.show()
number_purchases_users_with_more_than_two_convertions = df_number_convertions_per_user[df_number_convertions_per_user['person']>2]['person'].sum()
print(number_purchases_users_with_more_than_two_convertions)
number_purchases_users_with_two_convertions = df_number_convertions_per_user[df_number_convertions_per_user['person']==2]['person'].sum()
print(number_purchases_users_with_two_convertions)
number_purchases_users_with_only_one_convertions = df_number_convertions_per_user[df_number_convertions_per_user['person']<2]['person'].sum()
print(number_purchases_users_with_only_one_convertions)
sizes_3 = [number_purchases_users_with_more_than_two_convertions, number_purchases_users_with_two_convertions, number_purchases_users_with_only_one_convertions]
labels_3 = ['Purchases usuarios con mas de 2 conversiones','Purchases usuarios con 2 conversiones', 'Purchases usuarios con solo una conversiones']
fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.pie(sizes_3, labels=labels_3, autopct='%1.1f%%', shadow=True)
ax3.axis('equal')
ax3.set_title("Compras por grupo de usuarios")
plt.show()
# ===> Construimos un dataframe con los usuarios que realizaron al menos una conversion

df_number_convertions_per_user_time = pd.DataFrame(df[(df['event']=='conversion') ][['timestamp','person']])
print(df_number_convertions_per_user_time.head())
df_number_convertions_per_user_time['timestamp'] =  pd.to_datetime(df_number_convertions_per_user_time['timestamp'])
# ===> Construimos un dataframe solo con los usuarios

df_number_convertions_per_user_time_unique = pd.DataFrame(df_number_convertions_per_user_time['person'].unique())

df_number_convertions_per_user_time_unique = df_number_convertions_per_user_time_unique.rename(columns={0: 'person_id'})

print(df_number_convertions_per_user_time_unique.head())
df_number_convertions_per_user_time['id'] = df_number_convertions_per_user_time.index
from datetime import datetime
import time
# ===> Construimos el dataframe con los eventos que cada usuario realizo

df_number_convertions_per_user_time = df_number_convertions_per_user_time.sort_values(['person', 'timestamp'], ascending=[True, True])

df_number_convertions_per_user_time['time_diff'] = df_number_convertions_per_user_time.groupby('person')['timestamp'].diff()

df_number_convertions_per_user_time['time_diff'] = df_number_convertions_per_user_time['time_diff'].dt.total_seconds()/60/60/24

print(df_number_convertions_per_user_time.head())
import featuretools as ft

es = ft.EntitySet('app_data')

es.entity_from_dataframe(
    entity_id='users', # define entity id
    dataframe=df_number_convertions_per_user_time_unique, # select underlying data
    index='person_id', # define unique index column
    #time_index="timestamp",
)

es.entity_from_dataframe(
    entity_id='users_logs', # define entity id
    dataframe=df_number_convertions_per_user_time, # select underlying data
    index='id', # define unique index column
    time_index="timestamp",
)

users_to_users_logs = ft.Relationship(
    es['users']['person_id'],
    es['users_logs']['person']
)

# ===> Agregamos la relationship al entity set

es = es.add_relationship(users_to_users_logs)

feature_matrix, features = ft.dfs(entityset=es, 
                                  target_entity="users", 
                                  agg_primitives=["count", "mean"], 
                                  trans_primitives=[],
                                  features_only=False)
print(feature_matrix.head(3))
data_user_with_two_or_more_convertions = feature_matrix[feature_matrix['COUNT(users_logs)']>2]['MEAN(users_logs.time_diff)']
del data_user_with_two_or_more_convertions.index.name
import seaborn as sns
#import matplotlib.pyplot as plt
print(data_user_with_two_or_more_convertions.mean())
print(data_user_with_two_or_more_convertions.mode())
print(data_user_with_two_or_more_convertions.min())
print(data_user_with_two_or_more_convertions.max())
data_user_with_two_or_more_convertions = pd.DataFrame({'avg_days_btw_conv':data_user_with_two_or_more_convertions.values})
data_user_with_two_or_more_convertions['avg_days_btw_conv-range'] = pd.cut(data_user_with_two_or_more_convertions['avg_days_btw_conv'], [0,1,7,78])
print(data_user_with_two_or_more_convertions['avg_days_btw_conv-range'].value_counts())
data_avg_days = data_user_with_two_or_more_convertions['avg_days_btw_conv-range'].value_counts()

labels_avg_days = ['Un dia o menos', 'Entre 1 y 7 dias', 'Mas de 7 dias']
fig_avg_days, ax_avg_days = plt.subplots()
ax_avg_days.pie(data_avg_days , labels=labels_avg_days, autopct='%1.1f%%', shadow=True)
ax_avg_days.axis('equal')
ax_avg_days.set_title("Dias promedio entre conversiones - Usuarios con mas de dos conversiones")
plt.show()
# Cuales son los modelos más buscados? Armamos un tagcloud con los modelos mas buscados

search_frecuencies = df.search_term.map(lambda s: s.lower()).value_counts().to_dict()
search_frecuencies.pop('-')
# pip install wordcloud
from wordcloud import WordCloud

tagcloud = WordCloud( background_color='white').generate_from_frequencies(search_frecuencies)


figure()
plt.imshow(tagcloud, interpolation='bilinear')
plt.axis("off") 
# ===> En las conversiones no se encuentran por ejemplo los datos de device_type, operating_system_version y otros que necesitamos. 
# ===> Debemos buscarlos en los eventos anteriores de usurario para obtener esta información.

df['converted'] = (df.event == 'conversion')
convs_df = df[df.event == 'conversion'].copy()
convs_df["conv_id"] = range(len(convs_df))
device_df = df[df.device_type != '-']
# Cruzamos la data de conversiones con los registros que tienen un device.
# Las columnas que se repiten entre los data frames quedan con los sufijos _conv y _dev.
convs_with_device = convs_df.merge(device_df, on="person", suffixes=("_conv", "_dev"))

# El registro de device tiene que preceder a la conversion
convs_with_device = convs_with_device[(convs_with_device.timestamp_conv > convs_with_device.timestamp_dev)]

# Calculo el rank para cada conversion, para elegir el último de los eventos con los device asociados a la conversion. 
convs_with_device['prev_event_id'] = convs_with_device.groupby("conv_id").timestamp_dev.rank(ascending=False)
# Me quedo con el ultimo de los eventos con device.
convs_with_device = convs_with_device[convs_with_device.prev_event_id == 1]
convs_with_device.head()
#Calculamos el porcentaje de conversiones segun el channel.
(convs_with_device.channel_dev.value_counts()/df.channel.value_counts()*100).sort_values()
figure()
title("Convertion by channel")
convs_with_device.channel_dev.value_counts().plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" )
xlabel('Channels')
ylabel('Number of conversion')
tight_layout()

#Calculamos cuantos usurarios realizaron conversiones desde smartphone o computadora.
convs_with_device.device_type_dev.value_counts()
figure()
title("Convertions by device type")
(convs_with_device.device_type_dev.value_counts()).plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" )
xlabel('Device type')
ylabel('Amout of convertions')
tight_layout()
# Calculamos el numero de conversiones segun sistema operativo
convs_with_device.operating_system_version_dev.map(lambda s: s.split()[0]).value_counts()
figure()
title("Convertions by os_type")
(convs_with_device.operating_system_version_dev.map(lambda s: s.split()[0]).value_counts()).plot(kind='bar', rot=0, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" )
xlabel('Os type')
ylabel('Amout of convertions')
tight_layout()
convs_with_device["brand"] = convs_with_device.model_conv.map(lambda model: model.split()[0])
convs_with_device["os_type"] = convs_with_device.operating_system_version_dev.map(lambda os: os.split()[0])
heatmap_data =(convs_with_device.groupby(["brand", "os_type"]).converted_conv.count().reset_index().pivot("os_type", "brand", "converted_conv").fillna(0)
)
heatmap_data=heatmap_data.div(heatmap_data.sum(1), axis=0)
import seaborn as sns

figure()
title('Brand purchased based on OS')
sns.heatmap(
    heatmap_data.div(heatmap_data.sum(1), axis=0).loc[["Android", "Windows", "iOS"]]*100, 
    cmap="Purples", 
    annot = True,
    cbar=False,
    square=True
  
)
tight_layout()
users_returning = df[df['new_vs_returning']=='Returning']['person'].unique()
users_returning = pd.DataFrame(users_returning)
users_returning = users_returning.rename(columns={0: 'person'})
user_new = df[~df['person'].isin(users_returning['person'])]['person'].unique()
user_new = pd.DataFrame(user_new)
user_new = user_new.rename(columns={0: 'person'})
print(len(user_new))
df_events_users_returning = df[df['person'].isin(users_returning['person'])]

convertions_returning_users = len(df_events_users_returning[df_events_users_returning['event']=='conversion'].index)
print(convertions_returning_users)
df_events_users_new = df[df['person'].isin(user_new['person'])]
convertions_new_users = len(df_events_users_new[df_events_users_new['event']=='conversion'].index)
print(len(df_events_users_new[df_events_users_new['event']=='conversion'].index))
data_r_n = [convertions_returning_users, convertions_new_users]
label_r_n = ['Conversiones returning users', 'Conversiones nuevos usuarios']
fig_r_n, ax_r_n = plt.subplots(figsize=(8,4))
ax_r_n.pie(data_r_n , labels=label_r_n , autopct='%1.1f%%', shadow=True)
ax_r_n.axis('equal')
ax_r_n.set_title("Conversiones usuarios segun new o returning")
plt.show()
total_usuarios = len(df['person'].unique())
print(total_usuarios)
df_number_checkouts_per_user = pd.DataFrame(df[df['event']=='checkout'][['person','model']])
total_usuarios_checkout = len(df_number_checkouts_per_user['person'].unique())
number_checkouts = len(df[df['event']=='checkout'].index)
print(number_checkouts)
number_convertions = len(df[df['event']=='conversion'].index)
print(number_convertions)
sizes_conv_check = [number_convertions, number_checkouts-number_convertions]

labels_conv_check = ['Checkouts con conversiones', 'Checkouts sin conversiones']
fig2, ax_conv_check = plt.subplots(figsize=(8,4))
ax_conv_check.pie(sizes_conv_check, labels=labels_conv_check, autopct='%1.1f%%', shadow=True)
ax_conv_check.axis('equal')
ax_conv_check.set_title("Checkouts y conversiones")
plt.show()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].dt.month
df_month_checkout = pd.DataFrame(df[df['event']=='checkout'].groupby('month')['event'].size())
df_month_checkout['month'] = df_month_checkout.index
del df_month_checkout.index.name
df_month_convertions = pd.DataFrame(df[df['event']=='conversion'].groupby('month')['event'].size())
df_month_convertions['month'] = df_month_convertions.index
del df_month_convertions.index.name
df_month_final = pd.merge(df_month_checkout, df_month_convertions, on='month' )
df_month_final['ratio_conv_checkout'] = df_month_final['event_y']/df_month_final['event_x']
df_month_final = df_month_final.rename(columns={'event_y': 'Conversiones', 'event_x': 'Checkouts'})
print(df_month_final)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6,16))

barplot = sns.barplot(x=df_month_final['month'], y=df_month_final['Conversiones'], ax=ax1)
barplot.set_title("Total checkouts between January and June")

barplot_2 = sns.barplot(x=df_month_final['month'], y=df_month_final['Checkouts'], ax=ax2)
barplot_2.set_title("Total conversions between January and June")

barplot_3 = sns.barplot(x=df_month_final['month'], y=df_month_final['ratio_conv_checkout'], ax=ax3)
barplot_3.set_title("Ratio conversions/ checkouts ratio between January and June")
def get_brand(item):
    
    return str(item).split(' ', 1)[0]
df_number_checkouts_per_user['brand'] = df_number_checkouts_per_user['model'].apply(get_brand)
df_value_counts_brand_checkout = pd.DataFrame(df_number_checkouts_per_user['brand'].value_counts())
df_value_counts_brand_checkout = df_value_counts_brand_checkout.rename(columns={'brand': 'count'})
df_value_counts_brand_checkout['brand'] = df_value_counts_brand_checkout.index
df_convertions = pd.DataFrame(df[(df['event']=='conversion')][['person', 'model']])
df_convertions['brand'] = df_convertions['model'].apply(get_brand)
df_value_counts_brand_convertions = pd.DataFrame(df_convertions['brand'].value_counts())
df_value_counts_brand_convertions = df_value_counts_brand_convertions.rename(columns={'brand': 'count'})
df_value_counts_brand_convertions['brand'] = df_value_counts_brand_convertions.index
df_bradn_conv_check_relation = pd.merge(df_value_counts_brand_checkout , df_value_counts_brand_convertions, on='brand')
df_bradn_conv_check_relation['Conv_por_checkout'] = df_bradn_conv_check_relation['count_y']/df_bradn_conv_check_relation['count_x']
fig = plt.subplots(figsize=(6,4))
barplot = sns.barplot(x=df_bradn_conv_check_relation['brand'], y=df_bradn_conv_check_relation['Conv_por_checkout'])
barplot.set_title("Checkouts y conversions")
convs_with_device.city_dev.value_counts().head(10)
figure()
title("Convertions by city")
(convs_with_device.city_dev.value_counts().head(10)).plot(kind='bar', rot=45, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b" )
xlabel('City')
ylabel('Amout of convertions')
tight_layout()
def make_dataframe_sequence_events(dataframe):
    
    df_first_event_users = pd.DataFrame(dataframe.groupby('person').nth(1)['event'])
    
    df_first_event_users = df_first_event_users.rename(columns={'event': 'first_event'})
    
    del df_first_event_users.index.name
    
    
    df_second_event_users = pd.DataFrame(dataframe.groupby('person').nth(2)['event'])
    
    df_second_event_users = df_second_event_users.rename(columns={'event': 'second_event'})
    
    del df_second_event_users.index.name
    
    
    df_final = pd.merge(df_first_event_users, df_second_event_users, left_index=True, right_index=True)
    
    df_thrid_event_users = pd.DataFrame(dataframe.groupby('person').nth(3)['event'])
    
    df_thrid_event_users = df_thrid_event_users.rename(columns={'event': 'thrid_event'})
    
    del df_thrid_event_users.index.name
    
    
    df_final = pd.merge(df_final, df_thrid_event_users, left_index=True, right_index=True)
    
    df_final['combination'] = df_final['first_event'].map(str) + '_'+ df_final['second_event'].map(str) + '_'+ df_final['thrid_event'].map(str)
    
    return df_final
df_number_convertions_per_user['person'] = df_number_convertions_per_user.index
df_users_without_convertions = df[~df['person'].isin(df_number_convertions_per_user['person'])]
df_users_with_convertions = df[df['person'].isin(df_number_convertions_per_user['person'])]
df_sequence_events_users_with_convertions = make_dataframe_sequence_events(df_users_with_convertions)
df_sequence_events_users_with_convertions = pd.DataFrame(df_sequence_events_users_with_convertions['combination'].value_counts())
df_sequence_events_users_with_convertions['flow'] = df_sequence_events_users_with_convertions.index
df_sequence_events_users_without_convertions = make_dataframe_sequence_events(df_users_without_convertions)
make_dataframe_sequence_events(df_users_without_convertions).head()
figure(figsize= (8,8))
title("Most common flows - Zero conversions users")
(df_sequence_events_users_without_convertions['combination'].value_counts().head(10)).plot(kind='bar', rot=90, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b", fontsize=8 )
xlabel('Flows')
ylabel('Total Users')
tight_layout()
figure(figsize= (8,8))
title("Most common Flows - Users with Conversions")
(df_sequence_events_users_with_convertions['combination'].head(10)).plot(kind='bar', rot=90, color =(0.1, 0.1, 0.1, 0.2), edgecolor="b", fontsize=8 )
xlabel('Flows')
ylabel('Total Users')
tight_layout()
