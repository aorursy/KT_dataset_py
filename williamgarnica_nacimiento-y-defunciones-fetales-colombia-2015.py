# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import svm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename)) 
        

# Any results you write to the current directory are saved as output.
DF_defunciones_fetales_2015=pd.read_csv('/kaggle/input/nacimientos-y-defuncionesfetales-colombia20152018/defunciones_fetales_colombia_2015_original.txt', encoding = "ISO-8859-1", delimiter='\t');
DF_nacimientos_2015=pd.read_csv('/kaggle/input/nacimientos-y-defuncionesfetales-colombia20152018/nacimientos_colombia_2015_original.txt', encoding = "ISO-8859-1", delimiter='\t');

# describe
print('info DF_defunciones_fetales_2015')
DF_defunciones_fetales_2015.head()
DF_defunciones_fetales_2015.info()
DF_defunciones_fetales_2015.describe()
DF_defunciones_fetales_2015.shape
DF_defunciones_fetales_2015.columns


# describe
print('\n\ninfo DF_nacimientos_2015')
DF_nacimientos_2015.head()
DF_nacimientos_2015.info()
DF_nacimientos_2015.info()
DF_nacimientos_2015.describe()
DF_nacimientos_2015.shape
DF_nacimientos_2015.columns
# NULOS
DF_defunciones_fetales_2015.isnull().sum()       
# elimina columnas con muchos nulos identificadas en la sentencia anterior
DF_defunciones_fetales_2015=DF_defunciones_fetales_2015.drop(['OTRSITIODE','CODOCUR','MAN_MUER','CODMUNOC','C_MUERTE','C_MUERTEC','C_MUERTED','C_MUERTEE','C_DIR12','C_ANT1','C_ANT12','C_ANT2','C_ANT22','C_ANT3','C_ANT32','C_PAT1','C_PAT2','C_MCM1'], axis=1)
#se eliminan columnas no relevantes
    # TIPO_DEFUN: Tipo de defuncion 1= fetal, 2=  no fetal 
    # CONS_EXP: Certificado de defuncin expedido por 
    # ULTCURMAD: se elimina ya que ya se tiene la columna NIV_EDUM
DF_defunciones_fetales_2015=DF_defunciones_fetales_2015.drop(['TIPO_DEFUN','CONS_EXP','ULTCURMAD','HORA','MINUTOS','MU_PARTO','C_MUERTEB','IDPROFCER','CAUSA_666','C_BAS1','C_DIR1','CAU_HOMOL','ASIS_MED','N_HIJOSM','IDADMISALU','PMAN_MUER'], axis=1)



# NULOS
DF_nacimientos_2015.isnull().sum()       
# elimina columnas con muchos nulos identificadas en la sentencia anterior
DF_nacimientos_2015=DF_nacimientos_2015.drop(['OTRO_SIT','OTRPARATX'], axis=1)
#se eliminan columnas no relevantes
    # APGAR1: tiempo en que se hizo la prueba de APGAR
    # APGAR2: tiempo en que se hizo la prueba de APGAR 
    # IDHEMOCLAS: se elimina ya que el dataframe de DF_defunciones_fetales_2015 no la tiene
    # IDFACTORRH: se elimina ya que el dataframe de DF_defunciones_fetales_2015 no la tiene
    # IDPUEBLOIN: De acuerdo con la cultura, pueblo o rasgos fsicos, el nacido vivo es reconocido por sus padres como, se elimina ya que el dataframe de DF_defunciones_fetales_2015 no la tiene
DF_nacimientos_2015=DF_nacimientos_2015.drop(['APGAR1','APGAR2','IDHEMOCLAS','IDFACTORRH','IDPUEBLOIN','EDAD_PADRE','NIV_EDUP','N_EMB','NUMCONSUL','IDCLASADMI'], axis=1);



# MERGE

# PRIMERO IGUALAMOS LAS COLUMNAS DE LOS DOS DATAFRAME

# SE ELIMINAN COLUMNAS QUE NO EXISTEN EN UNO U OTRO DATAFRAME
DF_nacimientos_2015=DF_nacimientos_2015.drop(['TALLA_NAC','ATEN_PAR','PROFESION','ULTCURPAD','FECHA_NACM','ULTCURMAD'], axis=1);

# SE AGREGA COLUMNA DE 'ES_DEFUNCION_FETAL' =1
DF_defunciones_fetales_2015['ES_DEFUNCION_FETAL'] = 1
# SE AGREGA COLUMNA DE 'ES_DEFUNCION_FETAL' =0
DF_nacimientos_2015['ES_DEFUNCION_FETAL'] = 0

# SE CAMBIA EL NOMBRE DE ALGUNAS COLUMNAS
DF_defunciones_fetales_2015 = DF_defunciones_fetales_2015.rename(columns={'A_DEFUN':'AREA',
                                   'SIT_DEFUN':'SITIO',
                                   'TIPO_EMB':'MUL_PARTO',
                                   'T_PARTO':'TIPO_PARTO',
                                   'IDCLASADMI':'NOMCLASAD'
                                    });

DF_defunciones_fetales_2015.MUL_PARTO[DF_defunciones_fetales_2015.MUL_PARTO==5]=9;
DF_nacimientos_2015 = DF_nacimientos_2015.rename(columns={'AREANAC':'AREA',
                                   'SIT_PARTO':'SITIO'
                                    });

DF_defunciones_fetales_2015 = DF_defunciones_fetales_2015.drop(DF_defunciones_fetales_2015[DF_defunciones_fetales_2015['SEG_SOCIAL']==0].index)
DF_nacimientos_2015 = DF_nacimientos_2015.drop(DF_nacimientos_2015[DF_nacimientos_2015['SEG_SOCIAL']==0].index)


#  SE ORDENAN LAS COLUMNAS DE NACIMIENTOS PARA QUE CONSUERDE CON LAS DE DEFUNCIONES
DF_nacimientos_2015 = DF_nacimientos_2015[['COD_DPTO', 'COD_MUNIC', 'AREA', 'SITIO', 'COD_INST', 'NOM_INST', 'ANO',
       'MES', 'SEXO', 'CODPRES', 'CODPTORE', 'CODMUNRE', 'AREA_RES',
       'SEG_SOCIAL', 'NOMCLASAD', 'TIPO_PARTO', 'MUL_PARTO', 'T_GES',
       'PESO_NAC', 'EDAD_MADRE', 'N_HIJOSV', 'EST_CIVM', 'NIV_EDUM','ES_DEFUNCION_FETAL']]

# SE CONCATENAN LAS FILAS
DF_NAC_DEF_2015=pd.concat([DF_nacimientos_2015, DF_defunciones_fetales_2015],sort=True)


# GRAFICAS
# DEFUNCIONES SEGUN NIVEL EDUCATIVO DE LA MADRE 2015

plt.figure(3,figsize=(30,7)) 
labels=['Preescolar','Basica primaria','Basica secundaria','Media acadmica o clasica','Media tecnica','Normalista','Tecnica profesional','Tecnolgica','Profesional','Especializacin','Maestra','Doctorado','Ninguno','Sin informacion']

plt.subplot(1,3,1)
nacimientos=DF_nacimientos_2015.groupby('NIV_EDUM')['NIV_EDUM'].value_counts()
plt.bar(labels,nacimientos, color='b')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun nivel academico madre 2015')
plt.xlabel(r'Nivel academico de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,2)
defunciones=DF_defunciones_fetales_2015.groupby('NIV_EDUM')['NIV_EDUM'].value_counts()
plt.bar(labels,defunciones, color='red')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun nivel academico madre 2015')
plt.xlabel(r'Nivel academico de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,3)
X = np.arange(len(labels))
plt.bar(X+0.00,nacimientos, color = "b", label='Nacimientos',width = 0.40)
plt.bar(X+0.40,defunciones, color = "r", label='Defunciones',width = 0.40)
plt.grid(True) 
plt.legend(loc='upper right')
plt.xticks(X+0.20,labels,rotation='vertical')
plt.ylabel('Cantidad')

# GRAFICAS
# DEFUNCIONES SEGUN EDAD DE LA MADRE


plt.figure(5,figsize=(30,7)) 
labels=['10 - 14 ','15 - 19','20 - 24','25 - 29','30 - 34','35- 39','40 - 44','45-49','50-54','Sin informacion']

plt.subplot(1,3,1)
nacimientos=DF_nacimientos_2015.groupby('EDAD_MADRE')['EDAD_MADRE'].value_counts()
plt.bar(labels,nacimientos, color='b')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun edad de la madre 2015')
plt.xlabel(r'Edad de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,2)
defunciones=DF_defunciones_fetales_2015.groupby('EDAD_MADRE')['EDAD_MADRE'].value_counts()
plt.bar(labels,defunciones, color='red')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun edad de la madre 2015')
plt.xlabel(r'Edad de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,3)
X = np.arange(len(labels))
plt.bar(X+0.00,nacimientos, color = "b", label='Nacimientos',width = 0.40)
plt.bar(X+0.40,defunciones, color = "r", label='Defunciones',width = 0.40)
plt.grid(True) 
plt.legend(loc='upper right')
plt.xticks(X+0.20,labels,rotation='vertical')
plt.ylabel('Cantidad')

# GRAFICA DE BARRAS
# DEFUNCIONES Y NACIMIENTOS SEGUN REGIMENES DE SEGURIDAD SOCIAL DE LA MADRE 2015

plt.figure(1,figsize=(30,7)) 
labels=['Contributivo','Subsidiado','Excepcion','Especial','No asegurado','Sin Informacion']

plt.subplot(1,3,1)
nacimientos=DF_nacimientos_2015.groupby('SEG_SOCIAL')['SEG_SOCIAL'].value_counts()
plt.bar(labels,nacimientos, color='b')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun regimenes 2015')
plt.xlabel(r'Régimen de seguridad social de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,2)
defunciones=DF_defunciones_fetales_2015.groupby('SEG_SOCIAL')['SEG_SOCIAL'].value_counts()
plt.bar(labels,defunciones, color='red')
plt.xticks(rotation='vertical')
plt.title('Defunciones fetales segun regimenes 2015')
plt.xlabel(r'Régimen de seguridad social de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,3)
X = np.arange(len(labels))
plt.bar(X+0.00,nacimientos, color = "b", label='Nacimientos',width = 0.40)
plt.bar(X+0.40,defunciones, color = "r", label='Defunciones',width = 0.40)
plt.grid(True) 
plt.legend(loc='upper right')
plt.xticks(X+0.20,labels,rotation='vertical')
plt.ylabel('Cantidad')






# GRAFICAS
# DEFUNCIONES SEGUN SEMANAS DE GESTACION 2015
#print(DF_NAC_DEF_2015['T_GES'].value_counts());
plt.figure(2) 
df = pd.DataFrame({'lab':['Menos de 22','De 22 a 27','De 28 a 37','De 38 a 41','De 42 y mas','Ignorado','Sin informacion'], 
                   'val':DF_defunciones_fetales_2015.groupby('T_GES')['T_GES'].value_counts()})
ax = df.plot.bar(x='lab', y='val')
plt.title('Semanas de gestacion al momento de la defuncion del feto 2015')
plt.xlabel(r'Rango semanas')
plt.ylabel('Cantidad defunciones')
plt.grid(True) # grilla
plt.show()
# GRAFICAS
# DEFUNCIONES SEGUN ESTADO CIVIL DE LA MADRE


plt.figure(4,figsize=(30,7)) 
labels=['Union libre 2 o mas años','Union libre menos de 2 anos','Separada, divorciada','Viuda','Soltera','Casada','Sin informacion']

plt.subplot(1,3,1)
nacimientos=DF_nacimientos_2015.groupby('EST_CIVM')['EST_CIVM'].value_counts()
plt.bar(labels,nacimientos, color='b')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun estado civil madre 2015')
plt.xlabel(r'Estado civil de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,2)
defunciones=DF_defunciones_fetales_2015.groupby('EST_CIVM')['EST_CIVM'].value_counts()
plt.bar(labels,defunciones, color='red')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun estado civil madre 2015')
plt.xlabel(r'Estado civil de la madre')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,3)
X = np.arange(len(labels))
plt.bar(X+0.00,nacimientos, color = "b", label='Nacimientos',width = 0.40)
plt.bar(X+0.40,defunciones, color = "r", label='Defunciones',width = 0.40)
plt.grid(True) 
plt.legend(loc='upper right')
plt.xticks(X+0.20,labels,rotation='vertical')
plt.ylabel('Cantidad')


# GRAFICAS
# DEFUNCIONES SEGUN EL TIPO DE EMBARAZO


plt.figure(5,figsize=(30,7)) 
labels=['Simple','Doble','Triple','Cuadruple a mas','Sin informacion']
plt.subplot(1,3,1)
nacimientos=DF_nacimientos_2015.groupby('MUL_PARTO')['MUL_PARTO'].value_counts()
plt.bar(labels,nacimientos, color='b')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun tipo embarazo 2015')
plt.xlabel(r'Tipo embarazo')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,2)
defunciones=DF_defunciones_fetales_2015.groupby('MUL_PARTO')['MUL_PARTO'].value_counts()
plt.bar(labels,defunciones, color='red')
plt.xticks(rotation='vertical')
plt.title('Nacimientos segun tipo embarazo 2015')
plt.xlabel(r'Tipo embarazo')
plt.grid(True) 
plt.ylabel('Cantidad')

plt.subplot(1,3,3)
X = np.arange(len(labels))
plt.bar(X+0.00,nacimientos, color = "b", label='Nacimientos',width = 0.40)
plt.bar(X+0.40,defunciones, color = "r", label='Defunciones',width = 0.40)
plt.grid(True) 
plt.legend(loc='upper right')
plt.xticks(X+0.20,labels,rotation='vertical')
plt.ylabel('Cantidad')

# coeficiente de pearsonr
# asume una distribucion normal
# linealmente relacionada
pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['T_GES'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('T_GES - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))

pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['SEG_SOCIAL'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('SEG_SOCIAL - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))

pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['NIV_EDUM'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('NIV_EDUM - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))

pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['EST_CIVM'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('EST_CIVM - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))

pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['EDAD_MADRE'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('EDAD_MADRE - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))

pearsonr_coefficient,p_value=stats.pearsonr(DF_NAC_DEF_2015['MUL_PARTO'],DF_NAC_DEF_2015['ES_DEFUNCION_FETAL'])
print ('MUL_PARTO - ES_DEFUNCION_FETAL pearsonr_coefficient: %0.3f   , p_value: %0.3f' %(pearsonr_coefficient ,p_value))


# correlacion
corr=DF_NAC_DEF_2015.corr()
corr

sns.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)
#Trazar relaciones por pares en un conjunto de datos.
#De manera predeterminada, esta función creará una cuadrícula de Ejes de manera que cada variable numérica 
#datase compartirá en el eje y en una sola fila y en el eje x en una sola columna. Los ejes diagonales se tratan 
#de manera diferente, dibujando una gráfica para mostrar la distribución univariada de los datos para la variable en esa columna.
#También es posible mostrar un subconjunto de variables o trazar diferentes variables en las filas y columnas.

sns.pairplot(DF_NAC_DEF_2015[['MUL_PARTO','SEG_SOCIAL','NIV_EDUM','EST_CIVM','EDAD_MADRE','T_GES','ES_DEFUNCION_FETAL']])
#Prueba de chi-cuadrado de independencia de variables en una tabla de contingencia.

#Esta función calcula la estadística de chi-cuadrado y el valor p para la prueba de hipótesis de independencia de las 
#frecuencias observadas en la tabla de contingencia [1] observada . 
#Las frecuencias esperadas se calculan en función de las sumas marginales bajo el supuesto de independencia	

variables=DF_NAC_DEF_2015[['ES_DEFUNCION_FETAL','SEG_SOCIAL','T_GES','NIV_EDUM','EDAD_MADRE','EST_CIVM']]
stat, p, dof, expected = stats.chi2_contingency(variables)
print('dof=%d' % dof)
print(expected)

# interpret test-statistic
prob =0.95
critical =stats.chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# scikit-learn 0.22.2 Entrenamiento  SVM ( Soporte de máquinas vectoriales )


# se filtran solos los de bogota #############################################################
DF_NAC_DEF_2015_BOGOTA = DF_NAC_DEF_2015[(DF_NAC_DEF_2015['COD_DPTO'] == 11)]

X = DF_NAC_DEF_2015_BOGOTA[['MUL_PARTO','SEG_SOCIAL','NIV_EDUM','EST_CIVM','EDAD_MADRE','T_GES']]
y = DF_NAC_DEF_2015_BOGOTA['ES_DEFUNCION_FETAL']
clf = svm.SVC()
clf.fit(X, y)


# PREDICCIONES

print(clf.predict([[1,5,3,2,1,4]]))
print(clf.predict([[1,2,4,1,5,1]]))
print(clf.predict([[1,2,3,2,1,4]]))
print(clf.predict([[1,3,4,2,4,4]]))
print(clf.predict([[1,1,4,1,3,9]]))
print(clf.predict([[1,4,3,2,5,4]]))
print(clf.predict([[1,2,4,2,1,4]]))

#Machine Learning con Regresión
#LinearRegression ajusta un modelo lineal con coeficientes 
# w = (w1, ..., wp) para minimizar la suma residual de cuadrados entre los objetivos 
# observados en el conjunto de datos y los objetivos predichos por la aproximación lineal.
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression

x_ent, x_test, y_ent, y_test = tts(DF_NAC_DEF_2015_BOGOTA[['COD_DPTO','COD_MUNIC','AREA','SITIO','MUL_PARTO','SEG_SOCIAL','NIV_EDUM','EST_CIVM','EDAD_MADRE','T_GES']], DF_NAC_DEF_2015_BOGOTA['ES_DEFUNCION_FETAL'])
x_ent.head()
lm = LinearRegression()
lm.fit(x_ent, y_ent)
lm.score(x_test,y_test )
print(lm.predict([[11,1,1,1,1,1,11,6,2,4]]))
print(lm.predict([[11,1,1,1,1,2,7,6,3,4]]))
print(lm.predict([[11,1,1,1,1,1,3,6,4,4]]))
print(lm.predict([[11,1,1,1,1,3,7,6,5,4]]))
print(lm.predict([[11,1,1,1,1,1,4,6,6,4]]))
print(lm.predict([[11,1,1,1,1,4,2,6,7,4]]))