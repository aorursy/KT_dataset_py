# magic function para hacer que los gráficos de matplotlib se renderizen en el notebook.

%matplotlib notebook



import datetime as datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math



plt.style.use('default') # Make the graphs a bit prettier

plt.rcParams['figure.figsize'] = (8, 11)
df1 = pd.read_csv('../input/propiedadesproperati/properati-AR-2013-08-01-properties-sell.csv', low_memory=False)

fecha = df1['created_on'] < '2013-08-01'

caba = df1.place_with_parent_names.str.contains('Capital')

gba = df1.place_with_parent_names.str.contains('G.B.A')

    

df1 = df1.loc[fecha & (caba | gba),:]

df1.info()
df2 = pd.read_csv('../input/propiedadesproperati/properati-AR-2014-02-01-properties-sell.csv', low_memory=False)

fecha = (df2['created_on']>= '2013-08-01') & (df2['created_on']<= '2014-01-31')

caba = df2.place_with_parent_names.str.contains('Capital')

gba = df2.place_with_parent_names.str.contains('G.B.A')



df2 = df2.loc[fecha & (caba | gba),:]

df2.info()
df3 = pd.read_csv('../input/propiedadesproperati/properati-AR-2014-09-01-properties-sell.csv', low_memory=False)

fecha = (df3['created_on']>= '2014-02-01') & (df3['created_on']<= '2014-08-31')

caba = df3.place_with_parent_names.str.contains('Capital')

gba = df3.place_with_parent_names.str.contains('G.B.A')



df3 = df3.loc[fecha & (caba | gba),:]

df3.info()
df4 = pd.read_csv('../input/propiedadesproperati/properati-AR-2015-01-01-properties-sell.csv', low_memory=False)

fecha = (df4['created_on']>= '2014-09-01') & (df4['created_on']<= '2014-12-31')

caba = df4.place_with_parent_names.str.contains('Capital')

gba = df4.place_with_parent_names.str.contains('G.B.A')



df4 = df4.loc[fecha & (caba | gba),:]

df4.info()
df5 = pd.read_csv('../input/propiedadesproperati/properati-AR-2015-07-01-properties-sell-six_months.csv', low_memory=False)



fecha = (df5['created_on']>= '2015-01-01') & (df5['created_on']<= '2015-07-31')

caba = df5.place_with_parent_names.str.contains('Capital')

gba = df5.place_with_parent_names.str.contains('G.B.A')



df5 = df5.loc[fecha & (caba | gba),:]

df5.info()
df6 = pd.read_csv('../input/propiedadesproperati/properati-AR-2015-12-01-properties-sell-six_months.csv', low_memory=False)

fecha = (df6['created_on']>= '2015-08-01') & (df6['created_on']<= '2015-11-30')

caba = df6.place_with_parent_names.str.contains('Capital')

gba = df6.place_with_parent_names.str.contains('G.B.A')



df6 = df6.loc[fecha & (caba | gba),:]

df6.info()
df7 = pd.read_csv('../input/propiedadesproperati/properati-AR-2016-06-01-properties-sell.csv', low_memory=False)

fecha = (df7['created_on']>= '2015-12-01') & (df7['created_on']<= '2016-06-30')

caba = df7.place_with_parent_names.str.contains('Capital')

gba = df7.place_with_parent_names.str.contains('G.B.A')



df7 = df7.loc[fecha & (caba | gba),:]

df7.info()
df8 = pd.read_csv('../input/propiedadesproperati/properati-AR-2017-01-01-properties-sell.csv', low_memory=False)

fecha = (df8['created_on']>= '2016-07-01') & (df8['created_on']<= '2017-01-31')

caba = df8.place_with_parent_names.str.contains('Capital')

gba = df8.place_with_parent_names.str.contains('G.B.A')



df8 = df8.loc[fecha & (caba | gba),:]

df8.info()
df9 = pd.read_csv('../input/propiedadesproperati/properati-AR-2017-08-01-properties-sell-six_months.csv', low_memory=False)

fecha = (df9['created_on']>= '2017-02-01') & (df9['created_on']<= '2017-07-31')

caba = df9.place_with_parent_names.str.contains('Capital')

gba = df9.place_with_parent_names.str.contains('G.B.A')



df9 = df9.loc[fecha & (caba | gba),:]

df9.info()
df1a4 = pd.concat([df1,df2,df3,df4])

df5a9 = pd.concat([df5,df6,df7,df8,df9])
df1a4 = df1a4.drop(df1a4[['operation','geonames_id', 'lat-lon', 'properati_url','image_thumbnail']], axis = 1)

df5a9 = df5a9.drop(df5a9[['id','operation','place_with_parent_names', 'geonames_id', 'lat-lon', 'properati_url','image_thumbnail','country_name', 'title', 'description']], axis = 1)
#Obtenemos el place_name a traves del place_with_parent_names, y lo eliminamos de df1a4

def obtener_state_name(x):

    return x.split('|')[2]



df1a4['state_name'] = df1a4['place_with_parent_names'].apply(obtener_state_name)

df1a4 = df1a4.drop(df1a4[['place_with_parent_names']], axis = 1)
df1a4.info()

df5a9.info()
surface_covered_is_invalid = (df5a9['surface_covered_in_m2'] < 10) | (df5a9['surface_covered_in_m2'] > 10000) | np.isnan(df5a9['surface_covered_in_m2'])

surface_total_is_invalid = (df5a9['surface_total_in_m2'] < 10) | (df5a9['surface_total_in_m2'] > 10000) | np.isnan(df5a9['surface_total_in_m2'])

invalid_en_simultaneo = surface_covered_is_invalid & surface_total_is_invalid

df5a9 = df5a9[np.logical_not(invalid_en_simultaneo)]



ambos_iguales = df5a9['surface_covered_in_m2'] == df5a9['surface_total_in_m2']

ambos_iguales.value_counts()
def obtener_superficie_final(covered, total):

    if (covered == total): return total

    if (math.isnan(covered) or covered < 10 or covered > 10000): return total

    if (math.isnan(total) or total < 10 or total > 10000): return covered

    return (covered + total)/2



df5a9['surface_in_m2'] = df5a9[['surface_covered_in_m2','surface_total_in_m2']].apply(lambda x: obtener_superficie_final(x[0],x[1]), axis = 1)

df5a9 = df5a9.drop(df5a9[['surface_covered_in_m2','surface_total_in_m2']],axis=1)
price_usd_per_m2_is_invalid = (df5a9['price_usd_per_m2'] < 100) | (df5a9['price_usd_per_m2'] > 10000) | np.isnan(df5a9['price_usd_per_m2'])

price_per_m2_is_invalid = (df5a9['price_per_m2'] < 100) | (df5a9['price_per_m2'] > 10000) | np.isnan(df5a9['price_per_m2'])

invalid_en_simultaneo = price_usd_per_m2_is_invalid & price_per_m2_is_invalid

df5a9 = df5a9[np.logical_not(invalid_en_simultaneo)]



ambos_iguales = df5a9['price_per_m2'] == df5a9['price_usd_per_m2']

ambos_iguales.value_counts()
def obtener_precio_por_m2_final(price_usd_per_m2, price_per_m2):

    if (price_usd_per_m2 == price_per_m2): return price_per_m2

    if (math.isnan(price_usd_per_m2) or price_usd_per_m2 < 100 or price_usd_per_m2 > 10000): return price_per_m2

    if (math.isnan(price_per_m2) or price_per_m2 < 100 or price_per_m2 > 10000): return price_usd_per_m2

    return (price_usd_per_m2 + price_per_m2)/2



df5a9.loc[:,['price_usd_per_m2']] = df5a9.loc[:,['price_usd_per_m2','price_per_m2']].apply(lambda x: obtener_superficie_final(x[0],x[1]), axis = 1)

df5a9 = df5a9.drop(df5a9[['price_per_m2']],axis=1)
columns1a4 = df1a4.columns.tolist().sort()

columns5a9 = df5a9.columns.tolist().sort()

columns1a4 == columns5a9
properties = pd.concat([df1a4,df5a9])
properties.info()

properties.describe()
properties[np.isnan(properties.price)][['price','currency','price_aprox_local_currency','price_aprox_usd','price_usd_per_m2']].info()
properties.dropna(subset=['price'], inplace=True)

properties.info()
surface_invalid = np.isnan(properties['surface_in_m2'])

properties = properties[np.logical_not(surface_invalid)]

properties.info()
surface_cota_inf = properties.surface_in_m2 < 10

surface_cota_sup = properties.surface_in_m2 > 10000

price_usd_per_m2_cota_inf = properties.price_usd_per_m2 < 100

price_usd_per_m2_cota_sup = properties.price_usd_per_m2 > 10000



condicion_final = ((surface_cota_inf | surface_cota_sup) | (price_usd_per_m2_cota_inf | price_usd_per_m2_cota_sup))



properties = properties[np.logical_not(condicion_final)]



properties.info()
apartments = properties.loc[properties.property_type.str.contains('apartment'),:]

houses = properties.loc[properties.property_type.str.contains('house'),:]

PHs = properties.loc[properties.property_type.str.contains('PH'),:]

stores = properties.loc[properties.property_type.str.contains('store'),:]
plot_superficie_total = properties.plot.scatter('price_usd_per_m2', 'surface_in_m2', alpha=0.32, color = 'purple', title = "Relación entre la superficie total de la propiedad y su valor por m2")

plot_superficie_total.set_xlabel("Precio por m2 (US$)")

plot_superficie_total.set_ylabel("Superficie total (m2)")

plot_superficie_total.axis([0,3000,0,1000])
apartments[['price_usd_per_m2', 'surface_in_m2']].describe()
plot1 = properties[['property_type', 'price_usd_per_m2']].groupby(['property_type']).mean().sort_values('price_usd_per_m2', ascending = False).plot(kind = 'barh',color='green',legend=None,title='Tipo de propiedad vs Precio por m2', grid = True)

plot1.set_xlabel("Precio por m2 (US$)")

plot1.set_ylabel("Tipo de Propiedad")

#Mas caro a mas barato: departamento - store - PH - casa

df_por_piso = apartments[['floor', 'price_usd_per_m2']]



df_por_piso = df_por_piso.groupby(['floor']).agg([np.mean, np.size]).reset_index()



df_por_piso = df_por_piso.loc[df_por_piso[('price_usd_per_m2', 'size')] >= 200,:]

df_por_piso_plot = df_por_piso.plot(kind='bar',color=['red','blue'],x = 'floor', y = [('price_usd_per_m2', 'mean'),('price_usd_per_m2', 'size')],title='Comparación precio y cantidad por número de pisos')

df_por_piso_plot.set_xlabel("Pisos")

df_por_piso_plot.set_ylabel("Precio por m2 (US$)")

df_por_piso_plot.legend(labels = ['Precio por m2 (US$)', 'Cantidad de muestras'])
df_por_piso = apartments

df_por_piso = df_por_piso[df_por_piso['floor'] < 50]

df_por_piso = df_por_piso.groupby(['place_name']).agg([np.mean, np.size]).reset_index()



df_por_piso = df_por_piso[['place_name','floor']].loc[df_por_piso[('floor','size')]>= 100,:].sort_values([('floor','mean')],ascending=False).head(15)

df_por_piso_plot = df_por_piso.plot(kind='barh',color='fuchsia',alpha=0.7,x = ['place_name'], y = [('floor', 'mean')],title='Barrios con los deptos más altos', legend = False, rot = 45)

df_por_piso_plot.set_xlabel("Cantidad de pisos")

df_por_piso_plot.set_ylabel("Barrio")
df_por_barrio = properties[['place_name', 'price_usd_per_m2']].groupby('place_name').agg([np.mean,np.size]).sort_values(('price_usd_per_m2', 'mean'), ascending = False).reset_index()



df_por_barrio = df_por_barrio.loc[df_por_barrio[('price_usd_per_m2', 'size')] > 50,:].head(15)

df_por_barrio_plot = df_por_barrio.plot(kind='bar',color='maroon',x = ['place_name'], y = [('price_usd_per_m2', 'mean')],title='Barrios más caros por m2', legend = False, rot = 45)

df_por_barrio_plot.set_xlabel("Barrio")

df_por_barrio_plot.set_ylabel("Precio por m2 (US$)")
df_barrios = apartments.groupby('place_name').agg([np.mean,np.size,np.std,np.max,np.min]) 

df_barrios = df_barrios.sort_values([('price_usd_per_m2','mean')],ascending=False).loc[df_barrios[('price_usd_per_m2','size')]>50]

df_barrios_plot = df_barrios.head(15).plot(kind='bar',y = [('price_usd_per_m2', 'mean')],title='Barrios con los departamentos más caros',rot = 45,stacked=True,color = ['chocolate'])

df_barrios_plot.set_xlabel("Barrio")

df_barrios_plot.set_ylabel("Precio final por m2 (US$)")
prop_house_complete = properties.loc[properties.property_type.str.contains('house'),:]

prop_house = prop_house_complete[['place_name', 'price_usd_per_m2']].groupby('place_name').agg([np.mean, np.size])

df_house= prop_house.loc[prop_house[('price_usd_per_m2','size')] > 50,:].reset_index().sort_values([('price_usd_per_m2','mean')],ascending=False).head(10)

df_house = df_house.plot(kind='bar',color='olive',x = ['place_name'], y = [('price_usd_per_m2', 'mean')],title='Barrios con las casas más caras', legend = False, rot = 45)

df_house.set_xlabel("Barrio")

df_house.set_ylabel("Precio final por m2 (US$)")
prop_stores_complete = properties.loc[properties.property_type.str.contains('store'),:]

prop_stores = prop_stores_complete[['place_name', 'price_usd_per_m2']].groupby('place_name').agg([np.mean, np.size])

df_stores= prop_stores.loc[prop_stores[('price_usd_per_m2','size')] > 15,:].reset_index().sort_values([('price_usd_per_m2','mean')],ascending=False).head(10)

df_stores = df_stores.plot(kind='bar',color='purple',x = ['place_name'], y = [('price_usd_per_m2', 'mean')],title='Barrios con los locales más caros', legend = False, rot = 45)

df_stores.set_xlabel("Barrio")

df_stores.set_ylabel("Precio final por m2 (US$)")
sacar_stores = properties.property_type.str.contains('stores')

df_por_cuartos = properties[np.logical_not(sacar_stores)]

df_por_cuartos = df_por_cuartos[['rooms', 'price_usd_per_m2']].groupby(['rooms']).agg([np.mean,np.size]).reset_index()



df_por_cuartos = df_por_cuartos.loc[df_por_cuartos[('price_usd_per_m2', 'size')] >= 100,:]

df_por_cuartos_plot = df_por_cuartos.plot(kind='bar',color='darkgreen', x = ['rooms'], y = [('price_usd_per_m2', 'mean')],title='Relacion Precio por m2 y cantidad de cuartos', legend = False, rot = 45)

df_por_cuartos_plot.set_xlabel('Cantidad de cuartos')

df_por_cuartos_plot.set_ylabel('Precio por m2 (US$)')
df_por_creacion = properties[['created_on', 'price_usd_per_m2']].groupby('created_on').agg(np.mean).reset_index()

df_por_creacion['mes'] = df_por_creacion['created_on'].apply(lambda x: x[:7]) #Me quedo solo con anio-mes

df_por_creacion = df_por_creacion.groupby('mes').agg(np.mean).reset_index().rename(columns = {'mes': "Fecha de publicacion", 'price_usd_per_m2': "Precio promedio en dolares (US$)"})

df_por_creacion_plot = df_por_creacion.plot(kind = 'line',color='red',x="Fecha de publicacion", y = "Precio promedio en dolares (US$)", grid=True, title = "Relacion entre la fecha de publicacion y el precio por m2", rot = 45)
df_100_mil_dolares_por_barrio = properties[['place_name', 'price_usd_per_m2']].groupby('place_name').agg([np.mean, np.size])

df_100_mil_dolares_por_barrio = df_100_mil_dolares_por_barrio.loc[df_100_mil_dolares_por_barrio[('price_usd_per_m2','size')] > 50,:]

df_100_mil_dolares_por_barrio['m2 con 100mil usd'] = 100000/df_100_mil_dolares_por_barrio[('price_usd_per_m2','mean')]

df_100_mil_dolares_por_barrio = df_100_mil_dolares_por_barrio.reset_index().sort_values('m2 con 100mil usd')

df_100_mil_dolares_por_barrio_plot = df_100_mil_dolares_por_barrio[:15].plot(kind='bar',color='goldenrod', x='place_name', y='m2 con 100mil usd', rot=45, legend = False, title = "Barrios en los que menos m2 compro con US$100.000 ")

df_100_mil_dolares_por_barrio_plot.set_xlabel('Barrio')

df_100_mil_dolares_por_barrio_plot.set_ylabel('Cantidad de metros')
df_por_barrio_apartment = properties.loc[properties['property_type'] == 'apartment',['place_name', 'property_type']].groupby('place_name').agg(np.size).reset_index()

df_por_barrio_apartment.columns = ['place_name','apartment']

df_por_barrio_house = properties.loc[properties['property_type'] == 'house',['place_name', 'property_type']].groupby('place_name').agg(np.size).reset_index()

df_por_barrio_house.columns = ['place_name','house']

df_por_barrio_ph = properties.loc[properties['property_type'] == 'PH',['place_name', 'property_type']].groupby('place_name').agg(np.size).reset_index()

df_por_barrio_ph.columns = ['place_name','PH']

df_por_barrio_store = properties.loc[properties['property_type'] == 'store',['place_name', 'property_type']].groupby('place_name').agg(np.size).reset_index()

df_por_barrio_store.columns = ['place_name','store']



df_por_barrio1 = pd.merge(df_por_barrio_apartment, df_por_barrio_house, how='outer')

df_por_barrio2 = pd.merge(df_por_barrio_ph, df_por_barrio_store, how='outer')

df_por_barrio = pd.merge(df_por_barrio1, df_por_barrio2, how='outer')

df_por_barrio = df_por_barrio.fillna(value = 0)



df_por_barrio['total'] = df_por_barrio['apartment'] +  df_por_barrio['house'] + df_por_barrio['PH'] + df_por_barrio['store']



#Me quedo con los barrios que tienen mas de 50 propiedades

df_por_barrio = df_por_barrio.loc[df_por_barrio['total'] > 50,:].sort_values('total',ascending=False)

df_por_barrio.info()

df_por_barrio.head(5)
df_por_barrio_plot = df_por_barrio[:10].plot(kind = 'barh',color='darkblue', x='place_name', y = 'total', rot=45, title='Cantidad de propiedades por barrio', legend = False)

df_por_barrio_plot.set_xlabel('Total de propiedades publicadas')

df_por_barrio_plot.set_ylabel('Barrio')
df_por_barrio_porcentaje_departamentos = df_por_barrio

df_por_barrio_porcentaje_departamentos['Porcentaje Apartments'] = df_por_barrio_porcentaje_departamentos['apartment'] / df_por_barrio_porcentaje_departamentos['total']



df_por_barrio_porcentaje_departamentos = df_por_barrio_porcentaje_departamentos.sort_values('Porcentaje Apartments', ascending=False)

df_por_barrio_porcentaje_departamentos_plot = df_por_barrio_porcentaje_departamentos[:10].plot(kind = 'bar',color='saddlebrown', x='place_name',y='Porcentaje Apartments', rot = 45, title = 'Porcentaje de departamentos por barrio', legend = False)

df_por_barrio_porcentaje_departamentos_plot.set_xlabel('Barrio')

df_por_barrio_porcentaje_departamentos_plot.set_ylabel('Porcentaje de departamentos')

df_por_barrio_porcentaje_stores = df_por_barrio

df_por_barrio_porcentaje_stores['Porcentaje Stores'] = df_por_barrio_porcentaje_stores['store'] / df_por_barrio_porcentaje_stores['total']



df_por_barrio_porcentaje_stores = df_por_barrio_porcentaje_stores.sort_values('Porcentaje Stores', ascending=False)

df_por_barrio_porcentaje_stores_plot = df_por_barrio_porcentaje_stores[:10].plot(kind = 'bar',color='crimson', x='place_name', y='Porcentaje Stores', legend = False, rot = 45, title = 'Porcentaje de locales por Barrio')

df_por_barrio_porcentaje_stores_plot.set_xlabel('Barrio')

df_por_barrio_porcentaje_stores_plot.set_ylabel('Porcentaje de locales')
df_por_barrio_porcentaje_houses = df_por_barrio

df_por_barrio_porcentaje_houses['Porcentaje Houses'] = df_por_barrio_porcentaje_houses['house'] / df_por_barrio_porcentaje_houses['total']



df_por_barrio_porcentaje_houses = df_por_barrio_porcentaje_houses.sort_values('Porcentaje Houses', ascending=True)

df_por_barrio_porcentaje_houses_plot = df_por_barrio_porcentaje_houses[:10].plot(kind = 'bar',color='red',alpha=0.75, x='place_name', y='Porcentaje Houses', rot = 35, title = 'Porcentaje de casas publicadas por barrio', legend = False)

df_por_barrio_porcentaje_houses_plot.set_xlabel('Barrio')

df_por_barrio_porcentaje_houses_plot.set_ylabel('Porcentaje de casas')

df_por_barrio_porcentaje_phs = df_por_barrio

df_por_barrio_porcentaje_phs['Porcentaje PHs'] = df_por_barrio_porcentaje_phs['PH'] / df_por_barrio_porcentaje_phs['total']



df_por_barrio_porcentaje_phs = df_por_barrio_porcentaje_phs.sort_values('Porcentaje PHs', ascending=False)

df_por_barrio_porcentaje_phs_plot = df_por_barrio_porcentaje_phs[:10].plot(kind = 'bar', color='darkmagenta',alpha=0.8, x='place_name', y='Porcentaje PHs', rot = 45, title= 'Porcentaje de PHs por barrio', legend = False)

df_por_barrio_porcentaje_phs_plot.set_xlabel('Barrio')

df_por_barrio_porcentaje_phs_plot.set_ylabel('Porcentaje de PHs')
df_barrios = properties.groupby('place_name').agg([np.mean,np.size,np.std,np.max,np.min]) 

df_barrios = df_barrios.sort_values([('price_usd_per_m2','std')],ascending=False).loc[df_barrios[('price_usd_per_m2','size')]>100]

df_barrios_plot = df_barrios.head(10).plot(kind='bar',y = [('price_usd_per_m2', 'std')],title='Diferencia de precios por barrio',rot = 45,stacked=True,color = ['darkslategrey'], legend = False)

df_barrios_plot.set_xlabel("Barrio")

df_barrios_plot.set_ylabel("Precio final por m2 (US$)")
df_arboles = pd.read_csv('../input/datasetsextrasgobiernociudad/arbolado-publico-lineal.csv', low_memory=False, delimiter=';')

df_arboles.info()
df_arboles = df_arboles[['ID_ARBOL','BARRIO']].groupby(['BARRIO']).agg(np.size).reset_index()

df_arboles.columns = ['BARRIO', 'CANTIDAD']

df_arboles = df_arboles.sort_values('CANTIDAD', ascending = False)
df_por_barrio = properties.loc[properties['state_name']=='Capital Federal', ['place_name', 'price_usd_per_m2']]

df_por_barrio.columns = ['BARRIO', 'price_usd_per_m2']

df_por_barrio['BARRIO'] = df_por_barrio['BARRIO'].apply(lambda x: x.upper())
df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('ABASTO'), ['BARRIO']] = 'BALVANERA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('BARRIO NORTE'), ['BARRIO']] = 'RECOLETA'

df_por_barrio = df_por_barrio[df_por_barrio['BARRIO'] != 'CAPITAL FEDERAL']

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('CATALINAS'), ['BARRIO']] = 'BOCA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('CENTRO / MICROCENTRO'), ['BARRIO']] = 'SAN NICOLAS'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('CONGRESO'), ['BARRIO']] = 'BALVANERA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('DISTRITO DE LAS ARTES'), ['BARRIO']] = 'SAN NICOLAS'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('LAS CAÑITAS'), ['BARRIO']] = 'PALERMO'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('POMPEYA'), ['BARRIO']] = 'NUEVA POMPEYA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('ONCE'), ['BARRIO']] = 'BALVANERA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('PALERMO'), ['BARRIO']] = 'PALERMO'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('PARQUE CENTENARIO'), ['BARRIO']] = 'CABALLITO'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('TRIBUNALES'), ['BARRIO']] = 'SAN NICOLAS'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('VILLA DEL PARQUE'), ['BARRIO']] = 'VILLA DEL PARQUE'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('VILLA GENERAL MITRE'), ['BARRIO']] = 'VILLA GRAL. MITRE'





# Ahora unifiquemos los acentos

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('SAN NICOLÁS'), ['BARRIO']] = 'SAN NICOLAS'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('AGRONOMÍA'), ['BARRIO']] = 'AGRONOMIA'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('VILLA PUEYRREDÓN'), ['BARRIO']] = 'VILLA PUEYRREDON'

df_por_barrio.loc[df_por_barrio.BARRIO.str.contains('CONSTITUCIÓN'), ['BARRIO']] = 'CONSTITUCION'
df_por_barrio = df_por_barrio.groupby('BARRIO').agg(np.mean).reset_index()
df_unificado = pd.merge(df_por_barrio, df_arboles, how='outer')

plot_arboles = df_unificado.plot.scatter('price_usd_per_m2', 'CANTIDAD', color = 'green', title = "Relación entre cantidad de árboles y precio en dólares por m2 promedio por barrio")

plot_arboles.set_xlabel("Precio por m2 (US$)")

plot_arboles.set_ylabel("Cantidad de Arboles")
df_area = pd.read_csv('../input/datasetsextrasgobiernociudad/barrios.csv', low_memory=False, delimiter=',')

df_area.info()
df_unificado = pd.merge(df_unificado, df_area, how='outer')

df_unificado['arboles_proporcional_al_area'] = df_unificado['CANTIDAD']/df_unificado['AREA']
plot_arboles = df_unificado.plot.scatter('price_usd_per_m2', 'arboles_proporcional_al_area', color = 'green', title = "Relación entre árboles por m2 y precio en dólares por m2 promedio por barrio")

plot_arboles.set_xlabel("Precio por m2 (US$)")

plot_arboles.set_ylabel("Árboles/Área")

plot_arboles.axis([1000, 6000,0,0.005])
df_unificado.sort_values('arboles_proporcional_al_area', ascending = False)[['BARRIO']].head(10)
df_unificado.sort_values('arboles_proporcional_al_area', ascending = True)[['BARRIO']].head(10)