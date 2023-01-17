# Importamos librerías de análisis de datos

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)

pd.set_option('display.float_format', '{:.2f}'.format)

sns.set(rc={'figure.figsize':(10,8)})

# pd.set_option('mode.chained_assignment', None) # Deshabilita SettingWithCopyWarning. Ojo.
# Cargamos el dataframe

df = pd.read_csv('data/train_dollar.csv', index_col='id', parse_dates=['fecha'])
df.shape
df.head(2)
# Veo la cantidad de elementos nulos de cada columna

display(df.isnull().sum())
df[['antiguedad','habitaciones','garages','banos','idzona']] = df[['antiguedad','habitaciones','garages','banos','idzona']].fillna(-1)
df[['antiguedad', 'habitaciones', 'garages', 'banos', 'idzona', 'Precio_USD']] = pd.DataFrame(df, columns=['antiguedad', 'habitaciones', 'garages', 'banos', 'idzona', 'Precio_USD'], dtype=int)
df[['antiguedad', 'habitaciones', 'garages', 'banos', 'idzona', 'Precio_USD']] = df[['antiguedad', 'habitaciones', 'garages', 'banos', 'idzona', 'Precio_USD']].apply(pd.to_numeric, downcast='integer').replace(-1, np.nan)
df[['metroscubiertos','metrostotales']] = df[['metroscubiertos','metrostotales']].fillna(-1)
df[['metroscubiertos', 'metrostotales']] = df[['metroscubiertos', 'metrostotales']].apply(pd.to_numeric,downcast='float').replace(-1, np.nan)
df[['gimnasio', 'usosmultiples', 'piscina', 'escuelascercanas', 'centroscomercialescercanos']] = pd.DataFrame(df, columns=['gimnasio', 'usosmultiples', 'piscina', 'escuelascercanas', 'centroscomercialescercanos'], dtype=int).apply(pd.to_numeric, downcast='integer')

df = df.astype({'tipodepropiedad': 'category'})
df.dtypes
display(df.isnull().sum())
df = df.sort_values(by=["fecha","id","Precio_USD"])
publicacionesXanios = pd.DataFrame(df[['fecha']], columns=['fecha'])

publicacionesXanios = publicacionesXanios["fecha"].groupby(publicacionesXanios['fecha'].dt.year).agg({'count'})

publicacionesXanios
print('________________________________________________________________________________')

print('Cantidad de "metrostotales" nulos ' + str(df['metrostotales'].isnull().sum()))

nullosLimpiables = df[ df['metrostotales'].isnull() & df['metroscubiertos'].notnull()]

coincidencias = nullosLimpiables.shape[0]

print('Cantidad de "metrostotales" nullos, con valores en "metroscubiertos" : ' + str(coincidencias))

print('________________________________________________________________________________')

print('Cantidad de "metroscubiertos" nulos ' + str(df['metroscubiertos'].isnull().sum()))

nullosLimpiables = df[ df['metroscubiertos'].isnull() & df['metrostotales'].notnull()]

coincidencias = nullosLimpiables.shape[0]

print('Cantidad de "metroscubiertos" nullos, con valores en "metrostotales" : ' + str(coincidencias))

print('________________________________________________________________________________')
metros = df.copy()

metros['totalesNull'] = np.where(metros['metrostotales'].isnull(), True, False)

metros['cubiertosNull'] = np.where(metros['metroscubiertos'].isnull(), True, False)

display(metros[['metrostotales','metroscubiertos']].isnull().sum())
metros['metrostotales'] = np.where(metros['metrostotales'].isnull() & metros['metroscubiertos'].notnull(), metros['metroscubiertos'], metros['metrostotales'])

print('Cantidad de "metrostotales" nulos ' + str(metros['metrostotales'].isnull().sum()))
metros['metrostotales'].corr(metros['metroscubiertos'])
propiedadesExteriores = metros.loc[(metros['tipodepropiedad'] == 'Terreno') | 

              (metros['tipodepropiedad'] == 'Terreno comercial') |

              (metros['tipodepropiedad'] == 'Rancho') |

              (metros['tipodepropiedad'] == 'Otros') |

              (metros['tipodepropiedad'] == 'Terreno industrial') |

              (metros['tipodepropiedad'] == 'Huerta') |

              (metros['tipodepropiedad'] == 'Garage'), ['metroscubiertos','metrostotales', 'tipodepropiedad']]

propiedadesExteriores = propiedadesExteriores.dropna()

pe = propiedadesExteriores
print('Cantidad de "metroscubiertos" de valor 0: ' 

      + str(len(pe[pe['metroscubiertos'] == 0])))
print("Terreno: " + str(pe[pe['tipodepropiedad'] == 'Terreno']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Terreno']['metroscubiertos'])))

print("Terreno comercial: " + str(pe[pe['tipodepropiedad'] == 'Terreno comercial']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Terreno comercial']['metroscubiertos'])))

print("Otros: " + str(pe[pe['tipodepropiedad'] == 'Otros']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Otros']['metroscubiertos'])))

print("Terreno industrial: " + str(pe[pe['tipodepropiedad'] == 'Terreno industrial']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Terreno industrial']['metroscubiertos'])))

print("Huerta: " + str(pe[pe['tipodepropiedad'] == 'Huerta']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Huerta']['metroscubiertos'])))

print("Garage: " + str(pe[pe['tipodepropiedad'] == 'Garage']['metrostotales'].corr(pe[pe['tipodepropiedad'] == 'Garage']['metroscubiertos'])))
pe[pe['tipodepropiedad'] == 'Garage']
metros['metroscubiertos'] = np.where(metros['metroscubiertos'].isnull() & metros['metrostotales'].notnull(), metros['metrostotales'], metros['metroscubiertos'])

print('Cantidad de "metroscubiertos" nulos ' + str(metros['metroscubiertos'].isnull().sum()))
display(metros[['metrostotales','metroscubiertos']].isnull().sum())
Noroeste = ['Baja California Norte', 'Baja California Sur','Chihuahua', 'Durango', 'Sinaloa', 'Sonora']

Noreste = ['Coahuila','Nuevo León', 'Tamaulipas']

Oeste = ['Colima', 'Jalisco', 'Michoacán', 'Nayarit']

Este = ['Hidalgo', 'Puebla', 'Tlaxcala', 'Veracruz']

Centronorte = ['Aguascalientes', 'Guanajuato', 'Querétaro', 'San luis Potosí', 'Zacatecas']

Centrosur = ['Edo. de México', 'Distrito Federal', 'Morelos']

Suroeste = ['Chiapas', 'Guerrero', 'Oaxaca']

Sureste = ['Campeche', 'Quintana Roo', 'Tabasco', 'Yucatán']
def region(provincia):

    if(provincia in Noroeste):

        return 'Noroeste'

    if(provincia in Noreste):

        return 'Noreste'

    if(provincia in Oeste):

        return 'Oeste'

    if(provincia in Este):

        return 'Este'

    if(provincia in Centronorte):

        return 'Centronorte'

    if(provincia in Centrosur):

        return 'Centrosur'

    if(provincia in Suroeste):

        return 'Suroeste'

    return 'Sureste'
df['region'] = df['provincia'].apply(region)

df = df.astype({'region': 'category'})
df.groupby(['region']).agg({'Precio_USD':'mean'}).sort_values('Precio_USD', ascending = False)
reg_cs = df[ df['region'] == 'Centrosur']['Precio_USD']

reg_cn = df[ df['region'] == 'Centronorte']['Precio_USD']

reg_o = df[ df['region'] == 'Oeste']['Precio_USD']

reg_ne = df[ df['region'] == 'Noreste']['Precio_USD']

reg_e = df[ df['region'] == 'Este']['Precio_USD']

reg_se = df[ df['region'] == 'Sureste']['Precio_USD']

reg_no = df[ df['region'] == 'Noroeste']['Precio_USD']

reg_so = df[ df['region'] == 'Suroeste']['Precio_USD']



data = {'Centrosur':reg_cs, 'Noreste':reg_ne, 'Suroeste':reg_so, 'Oeste':reg_o, 'Sureste':reg_se, 'Centronorte':reg_cn, 'Este':reg_e, 'Noroeste':reg_no}
precioXregion = pd.DataFrame(data)

sns.set(rc={'figure.figsize':(10,8)})

ax = sns.boxplot(x="variable", y="value", data=pd.melt(precioXregion), dodge=True)

ax.set(xlabel='', ylabel='Valor U$D', title="Valor en U$D de las propiedades por region")

plt.show()
#cantidad de propiedades por region

df['region'].value_counts().plot(kind='bar', figsize=(10,8), title='Cantidad de publicaciones por region')
df['region'].value_counts()
import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML
regXmes = df[['region','fecha']].copy()

regXmes['anio_mes'] = pd.to_datetime(regXmes['fecha']).dt.to_period('M')

regXmes = regXmes.groupby(['anio_mes', 'region']).agg({'count'})

regXmes.columns = regXmes.columns.droplevel(0)

regXmes = regXmes.rename_axis(None, axis=1)

regXmes.reset_index(inplace=True)

regXmes.head(8)
dfc_cn = regXmes[regXmes['region'] == 'Centronorte'].copy()

dfc_cs = regXmes[regXmes['region'] == 'Centrosur'].copy()

dfc_e = regXmes[regXmes['region'] == 'Este'].copy()

dfc_o = regXmes[regXmes['region'] == 'Oeste'].copy()

dfc_se = regXmes[regXmes['region'] == 'Sureste'].copy()

dfc_so = regXmes[regXmes['region'] == 'Suroeste'].copy()

dfc_ne = regXmes[regXmes['region'] == 'Noreste'].copy()

dfc_no = regXmes[regXmes['region'] == 'Noroeste'].copy()
dfc_cn = dfc_cn.sort_values(['anio_mes'], ascending = True)

dfc_cs = dfc_cs.sort_values(['anio_mes'], ascending = True)

dfc_e = dfc_e.sort_values(['anio_mes'], ascending = True)

dfc_o = dfc_o.sort_values(['anio_mes'], ascending = True)

dfc_se = dfc_se.sort_values(['anio_mes'], ascending = True)

dfc_so = dfc_so.sort_values(['anio_mes'], ascending = True)

dfc_ne = dfc_ne.sort_values(['anio_mes'], ascending = True)

dfc_no = dfc_no.sort_values(['anio_mes'], ascending = True)
dfc_cn['suma'] = dfc_cn['count'].rolling(dfc_cn.shape[0], min_periods=1).sum()

dfc_cs['suma'] = dfc_cs['count'].rolling(dfc_cs.shape[0], min_periods=1).sum()

dfc_e['suma'] = dfc_e['count'].rolling(dfc_e.shape[0], min_periods=1).sum()

dfc_o['suma'] = dfc_o['count'].rolling(dfc_o.shape[0], min_periods=1).sum()

dfc_se['suma'] = dfc_se['count'].rolling(dfc_se.shape[0], min_periods=1).sum()

dfc_so['suma'] = dfc_so['count'].rolling(dfc_so.shape[0], min_periods=1).sum()

dfc_ne['suma'] = dfc_ne['count'].rolling(dfc_ne.shape[0], min_periods=1).sum()

dfc_no['suma'] = dfc_no['count'].rolling(dfc_no.shape[0], min_periods=1).sum()
dfc_cn['crecimiento'] = dfc_cn['suma'].pct_change()

dfc_cs['crecimiento'] = dfc_cs['suma'].pct_change()

dfc_e['crecimiento'] = dfc_e['suma'].pct_change()

dfc_o['crecimiento'] = dfc_o['suma'].pct_change()

dfc_se['crecimiento'] = dfc_se['suma'].pct_change()

dfc_so['crecimiento'] = dfc_so['suma'].pct_change()

dfc_ne['crecimiento'] = dfc_ne['suma'].pct_change()

dfc_no['crecimiento'] = dfc_no['suma'].pct_change()
frames = [dfc_cn, dfc_cs, dfc_e, dfc_o, dfc_se, dfc_so, dfc_ne, dfc_no]

regXmes = pd.concat(frames)

regXmes = regXmes.sort_values(['anio_mes','suma'], ascending = False)

regXmes.head(9)
def draw_barchart(anio_mes):

    dff = regXmes[regXmes['anio_mes'].eq(anio_mes)].sort_values(by='suma', ascending=True).tail(10)

    ax.clear()

    ax.barh(dff['region'], dff['suma'])

    dx = dff['suma'].max() / 200

    for i, (suma, region) in enumerate(zip(dff['suma'], dff['region'])):

        ax.text(suma-dx, i,     region,           size=14, weight=600, ha='right', va='bottom')

        ax.text(suma+dx, i,     f'{suma:,.0f}',  size=14, ha='left',  va='center')

    

    ax.text(1, 0.4, (str(anio_mes.month)+' - '+str(anio_mes.year)), transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Publicaciones', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Cantidad de publicaciones por region desde 2012 a 2016',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    plt.box(False)
import matplotlib.animation as animation

from IPython.display import HTML

fig, ax = plt.subplots(figsize=(15, 8))

animator = animation.FuncAnimation(fig, draw_barchart, frames=pd.date_range(start='1/1/2012', end='1/1/2017', freq='M'))

HTML(animator.to_jshtml()) 
metros = df.copy()
sns.set(rc={'figure.figsize':(10,8)})

g = sns.PairGrid(metros, vars=['metrostotales', 'metroscubiertos', 'Precio_USD'],

                 hue='region', palette='RdBu_r')

g.map(plt.scatter, alpha=0.3)

g.add_legend();
bad = df[df['metrostotales'] < df['metroscubiertos']].copy()

bad['anio_mes'] = pd.to_datetime(bad['fecha']).dt.to_period('M')
bad = bad[['anio_mes','titulo']].groupby(['anio_mes']).agg({'count'})

bad.columns = bad.columns.droplevel(0)

bad.reset_index(inplace=True)
bad.columns = ['Año','Cantidaad']

bad.plot(x='Año', y='Cantidaad', title="Metrostotales menores a Metroscubiertos", figsize=(10,8))
metros_filtrados = metros[metros['metrostotales'] >= metros['metroscubiertos']]

sns.set(rc={'figure.figsize':(10,8)})

g = sns.PairGrid(metros_filtrados, vars=['metrostotales','Precio_USD'],

                 hue='region', palette='RdBu_r')

g.map(plt.scatter, alpha=0.2)

g.add_legend();
# ax = sns.scatterplot(data=metros_filtrados, x="metrostotales", y="metroscubiertos",alpha=0.3)
sns.set_style("whitegrid")

g = sns.FacetGrid(metros_filtrados, col="region",col_wrap=4,hue='region', palette='RdBu_r')

g.map(plt.scatter, "metrostotales", "Precio_USD", alpha=.2)

g.add_legend();
dfc_no.head()
plt.rcParams['figure.figsize'] = (10, 8)

ax = dfc_no.plot(x='anio_mes',y="crecimiento", legend=False)

ax.set_prop_cycle(None)

ax = dfc_ne.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

ax = dfc_e.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

ax = dfc_o.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

ax = dfc_se.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

ax = dfc_so.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

ax = dfc_cn.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False)

dfc_cs.plot(ax=ax,x='anio_mes',y="crecimiento", legend=False, title='Porcentaje de crecimeinto de publicaciones por año')

ax.set_xlabel("Porcentaje")

ax.set_ylabel("Años")

plt.rcParams['figure.figsize'] = (10, 8)

ax = dfc_no.plot(x='anio_mes',y="suma")

ax.set_prop_cycle(None)

ax = dfc_ne.plot(ax=ax,x='anio_mes',y="suma")

ax = dfc_e.plot(ax=ax,x='anio_mes',y="suma")

ax = dfc_o.plot(ax=ax,x='anio_mes',y="suma")

ax = dfc_se.plot(ax=ax,x='anio_mes',y="suma")

ax = dfc_so.plot(ax=ax,x='anio_mes',y="suma")

ax = dfc_cn.plot(ax=ax,x='anio_mes',y="suma")

dfc_cs.plot(ax=ax,x='anio_mes',y="suma", legend=False, title='Crecimeinto de publicaciones por año')

ax.set_xlabel("Cantidad total")

ax.set_ylabel("Años")

plt.rcParams['figure.figsize'] = (10, 8)