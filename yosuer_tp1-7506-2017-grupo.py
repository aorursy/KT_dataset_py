# Alumnos:

# Jhonatan Zubieta, Padron: 91256

# Felipe Gonzalez, Padron: 91387

# Gonzalo Fernandez, Padron: 94667

# Link a repositorio de Github: https://github.com/yosuer/TP1-OrgaDeDatos
import numpy as np 

import pandas as pd

import calendar



#Plots

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Concatenamos los DataFrame y eliminamos los elementos duplicados



##Podriamos poner las ruta de los archivos que vamos a utilizar en un archivo de texto y cargar todo de ahi sin definir uno por uno



propiedades0 = pd.read_csv('DataSetsSell/properati-AR-2017-08-01-properties-sell-six_months.csv') #datasetTP1.csv



propiedades1 = pd.read_csv('DataSetsSell/properati-AR-2017-08-01-properties-sell.csv')



propiedades2 = pd.read_csv('DataSetsSell/properati-AR-2017-07-03-properties-sell.csv')



propiedades3 = pd.read_csv('DataSetsSell/properati-AR-2017-06-06-properties-sell.csv')



                        

frames = [propiedades0, propiedades1,propiedades2,propiedades3]    

    

propiedades = pd.concat(frames)



propiedadesSinDuplicados = propiedades.drop_duplicates()
#Revisamos el DataFrame resultante

propiedadesSinDuplicados.head()
#Seleccionamos solo las columnas con las que vamos a trabajar y verificamos sus tipos

props = propiedadesSinDuplicados[['created_on','operation','property_type','place_name','place_with_parent_names','state_name','lat-lon','lat','lon','price','currency','surface_total_in_m2','rooms','title']]



props.info()
#Convertimos la fecha created_on en datetime y agregamos el campo año, mes y dia

props['created_on'] = pd.to_datetime(props['created_on'])

props['year'] = props['created_on'].map(lambda x:x.year)

props['month'] = props['created_on'].map(lambda x:x.month)
#Filtramos los datos que nos interesan



#Limpiamos las publicaciones sin precio

props.dropna(subset=['price'],inplace=True)

#Utilizamos solo las que estan en dolares

props.dropna(subset=['currency'],inplace=True)

props = props.loc[props.currency.str.contains('USD'),:]

#Y eliminamos las que no poseen place_name

props.dropna(subset=['place_name'],inplace=True)

props = props.loc[~props.place_name.str.contains('Capital Federal'),:]
#Imprimimos por pantala cuantos registros tenemos ahora consistentes



props.info()
#Excluimos las provincias que no tienen relevancia (Trabajamos solo con CABA y Gran Buenos Aires)

propsBA = props.loc[props.state_name.str.contains('G.B.A|Capital Federal'),:]
#Spliteamos por tipo de operacion (Solo nos interesa las ventas)

propsBASell =  propsBA.loc[propsBA.operation.str.contains('sell'),:]
#Incremento de las ventas con el paso de los anios

propsBASell.groupby('year').count()['created_on'].plot(rot=0, figsize=(14,4), color='blue' ,fontsize=12)

plt.title('Incremento de las ventas con el paso de los anios', fontsize=20);

plt.xlabel('Anio', fontsize=16);

plt.ylabel('Cantidad de operaciones', fontsize=16);
#Barrios con mayor cantidad de ventas

propsBASell.loc[propsBASell.state_name.str.contains('Capital Federal'),:].groupby('place_name').count()['created_on'].sort_values(ascending=False)[0:10].plot(kind='bar',rot=0, figsize=(14,4), color='green' ,fontsize=12)

plt.title('Los 10 barrios de CABA con mayor cantidad de ventas', fontsize=20);

plt.xlabel('Barrio', fontsize=16);

plt.ylabel('Cantidad de operaciones', fontsize=16);
#Mayor cantidad de ventas por Localidad

propsBASell['state_name'].value_counts().plot(kind='bar',rot=0, figsize=(14,4), color='red' ,fontsize=12);

plt.title('Mayor cantidad de ventas por Localidad', fontsize=20);

plt.xlabel('Localidad', fontsize=16);

plt.ylabel('Cantidad de operaciones', fontsize=16);
#Propiedades mas caras en promedio por Barrio (Tener en cuenta la cantidad de propiedades que existen en cada lugar)

propsBASell.groupby('place_name')['price'].agg([np.mean]).sort_values('mean', ascending=False)[0:10].plot(kind='bar',rot=90, figsize=(14,4), color='black' ,fontsize=12);

plt.title('Lugar o barrio con las propiedades mas caras', fontsize=20);

plt.xlabel('Barrio', fontsize=16);

plt.ylabel('Precio en Dolares', fontsize=16);
#Evolucion del precio de la propiedad con el paso de los años

evolucionPropiedad = propsBASell[['created_on', 'price']]

evolucionPropiedad.set_index('created_on',inplace=True)

evolucionPropiedad['price'].sample(len(evolucionPropiedad)).sort_index().rolling(window=10000,center=False).mean().plot(figsize=(14,4))



plt.title('Evolucion del precio de la Propiedad', fontsize=20);

plt.xlabel('Paso de los Anios', fontsize=16); plt.xlim('2015-09','2017-09');

plt.ylabel('Precio en Dolares', fontsize=16);

# Precio promedio de la propiedad por zonas geograficas

# Cancha de River alrededor de 10 cuadras

# Cantidad de operaciones por año



propsriver=props.loc[props.state_name.str.contains('Capital Federal'),:]



propsriverX=propsriver.loc[propsriver.lat.between(-34.558123865,-34.542465451,inclusive=True),:]

propsriverXY=propsriverX.loc[propsriverX.lon.between(-58.467156887,-58.433983326,inclusive=True),:]



propsriverXY.groupby('year').count()['created_on'].plot(kind='bar', rot=0, figsize=(14,4), color='green' ,fontsize=12)



plt.title('Cantidad de operaciones por anio', fontsize=20);

plt.xlabel('Anio', fontsize=16); 

plt.ylabel('Operaciones', fontsize=16);

# Precio promedio de la propiedad por zonas geograficas (Continua de la anterior)

# Cancha de river alrededor de 10 cuadras

# Operaciones mas caras realizadas

# FALTA ARREGLAR LA AGRUPACION



#propsriverXY.groupby('year')['price'].head(10).plot(kind='bar',figsize=(14,4))



#propsriverXY['price'].sort_values('price',ascending=[False]).drop_duplicates().plot()



#propsriverXY.sort_values(['price'],ascending=[False]).drop_duplicates().head(10)

#propsriverXY.sort_values(['price'],ascending=[False])[0:10].plot(kind='bar',figsize=(14,4))



#propsriverporprecio.groupby('created_on')['price'].plot(kind='bar',figsize=(14,4))



#propsriverXY.groupby()
#Estadisticas operaciones en el primer anillo del GBA



operacionesEnGBA = propsBASell



operacionesEnGBA['partido'] = operacionesEnGBA.place_with_parent_names.str.split('|').str.get(3)



propsPrimerCordon = operacionesEnGBA.loc[operacionesEnGBA.partido.str.contains('Avellaneda|Lanús|Lomas de Zamora|La Matanza|Morón|Tres de Febrero|San Martín|Vicente López|San Isidro'),:]



propsPrimerCordon.groupby('partido').count()['created_on'].sort_values(ascending=False).plot(kind='bar', rot=90, figsize=(14,4), color='orange' ,fontsize=12)



plt.title('Cantidad de operaciones en el primer anillo del GBA', fontsize=20);

plt.xlabel('Partido', fontsize=16); 

plt.ylabel('Operaciones', fontsize=16);
#Estadisticas operaciones en el segundo anillo del GBA



propsSegundoCordon = operacionesEnGBA.loc[operacionesEnGBA.partido.str.contains('Quilmes|Berazategui|Florencio Varela|Almirante Brown|Esteban Echeverría|Ezeiza|Moreno|Merlo|Hurlingham|Ituzaingó|La Matanza|Tigre|San Fernando|José C. Paz|San Miguel|Malvinas Argentinas'),:]



propsSegundoCordon.groupby('partido').count()['created_on'].sort_values(ascending=False).plot(kind='bar', rot=90, figsize=(14,4), color='purple' ,fontsize=12)



plt.title('Cantidad de operaciones en el segundo anillo del GBA', fontsize=20);

plt.xlabel('Partido', fontsize=16); 

plt.ylabel('Operaciones', fontsize=16);
#Estadisticas operaciones en el tercer anillo del GBA



propsTercerCordon = operacionesEnGBA.loc[operacionesEnGBA.partido.str.contains('San Vicente|Presidente Perón|Marcos Paz|General Rodríguez|Escobar|Pilar'),:]



propsTercerCordon.groupby('partido').count()['created_on'].sort_values(ascending=False).plot(kind='bar', rot=90, figsize=(14,4), color='yellow' ,fontsize=12)



plt.title('Cantidad de operaciones en el primer anillo del GBA', fontsize=20);

plt.xlabel('Partido', fontsize=16); 

plt.ylabel('Operaciones', fontsize=16);
# Evolucion del precio de la propiedad con el crecimiento del polo tecnologico de Parque Patricios

evolucionPropiedadParquePatricios = propsBASell[['created_on', 'price','place_name']]



parquePatricios = evolucionPropiedadParquePatricios.loc[evolucionPropiedadParquePatricios.place_name.str.contains('Parque Patricios'), :]

parquePatricios.set_index('created_on',inplace=True)

parquePatricios['price'].sample(len(parquePatricios)).sort_index().rolling(window=10,center=False).mean().plot(figsize=(14,4))



plt.title('Evolucion del precio en Parque Patricios', fontsize=20);

plt.xlabel('Paso de los Anios', fontsize=16); 

plt.ylabel('Precio en Dolares', fontsize=16);
#Porcentaje de ventas en CABA y GBA



caba=propsBASell.loc[propsBASell.state_name.str.contains('Capital Federal'),:]

gba=propsBASell.loc[propsBASell.state_name.str.contains('G.B.A'),:]

sizes = [len(caba), len(gba)]

nombres = ['CABA', 'GBA']



plt.figure(figsize=(6, 6))

plt.title('Porcentaje de ventas en CABA y GBA', fontsize=20)

plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['orange', 'green'], explode=(0.1, 0))

plt.show()
#Dia de la semana con mayor cantidad de ventas

diaConMayorVentas = propsBASell

diaConMayorVentas['dia_de_semana'] = diaConMayorVentas['created_on'].map(lambda x:x.weekday_name)



diaConMayorVentas['dia_de_semana'].value_counts().plot(kind='bar', rot=0, figsize=(14,4), color='blue' ,fontsize=12)



plt.title('Cantidad de operaciones segun el dia', fontsize=20);

plt.xlabel('Dia de la semana', fontsize=16); 

plt.ylabel('Cantidad de operaciones', fontsize=16);
#Cantidad de ventas para cada mes durante el 2017



propsBA2017 = propsBASell.loc[(propsBASell.year == 2017) ,:]



propsBA2017.groupby('month').count()['created_on'].plot(kind='line',rot=0, figsize=(14,4), color='purple' ,fontsize=12)

plt.title('Cantidad de ventas durante 2017', fontsize=20);

plt.xlabel('Mes', fontsize=16);

plt.xlim(1,7);

plt.ylabel('Cantidad de operaciones', fontsize=16);