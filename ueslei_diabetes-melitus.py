# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#entrada de dados

# O banco de dados em questão, apresenta os missing como sendo '?'. ao carregar os dados com a biblioteca pandas, substituímos por NAN

df = pd.read_csv('/kaggle/input/diabetes/diabetic_data.csv')

df.replace('?', np.nan, inplace=True)
#quantidade de linhas e colunas

df.shape
#exibindo nomes de colunas

df.columns
# EXCLUÍNDO DADOS DUPLICADOS

# Pegarei apenas a primeira visita de cada paciente para que não tenhamos interferências nos resultados

df['duplicado'] = df.patient_nbr.duplicated()

df = df[df['duplicado'] == False]



df.drop(['duplicado'], axis = 1, inplace = True)
df.shape
#Para possibilitar os cálculos de predição, todos os dados precisam ser numéricos



# Alterando variável race

df.loc[df['race'] == 'Caucasian', ['race']] = 1

df.loc[df['race'] == 'AfricanAmerican', ['race']] = 0

#df.loc[df['race'] == '?', ['race']] = 2

df.loc[df['race'] == 'Other', ['race']] = 2

df.loc[df['race'] == 'Asian', ['race']] = 2

df.loc[df['race'] == 'Hispanic', ['race']] = 2





# Alterando variável gender

df.loc[df['gender'] == 'Female', ['gender']] = 0

df.loc[df['gender'] == 'Male', ['gender']] = 1

df.replace('Unknown/Invalid', np.nan, inplace=True)

#df.loc[df['gender'] == 'Unknown/Invalid', ['gender']] = -1000 #ausente



# Alterando variável age

df.loc[df['age'] == '[0-10)', ['age']] = 0

df.loc[df['age'] == '[10-20)', ['age']] = 0

df.loc[df['age'] == '[20-30)', ['age']] = 1

df.loc[df['age'] == '[30-40)', ['age']] = 1

df.loc[df['age'] == '[40-50)', ['age']] = 2

df.loc[df['age'] == '[50-60)', ['age']] = 2

df.loc[df['age'] == '[60-70)', ['age']] = 3

df.loc[df['age'] == '[70-80)', ['age']] = 3

df.loc[df['age'] == '[80-90)', ['age']] = 4

df.loc[df['age'] == '[90-100)', ['age']] = 4



# Alterando variável weight (excluida)

#df.loc[df['weight'] == '?', ['weight']] = 0

df.loc[df['weight'] == '[0-25)', ['weight']] = 1

df.loc[df['weight'] == '[25-50)', ['weight']] = 2

df.loc[df['weight'] == '[50-75)', ['weight']] = 3

df.loc[df['weight'] == '[75-100)', ['weight']] = 4

df.loc[df['weight'] == '[100-125)', ['weight']] = 5

df.loc[df['weight'] == '[125-150)', ['weight']] = 6

df.loc[df['weight'] == '[150-175)', ['weight']] = 7

df.loc[df['weight'] == '[175-200)', ['weight']] = 8

df.loc[df['weight'] == '>200', ['weight']] = 9



#Admission type

df['admission_type_id_II'] = ""

df.loc[df['admission_type_id'] == 1 , ['admission_type_id_II']] = 1

df.loc[df['admission_type_id'] == 2 , ['admission_type_id_II']] = 2

df.loc[df['admission_type_id'] == 3 , ['admission_type_id_II']] = 3

df.loc[df['admission_type_id'] == 5 , ['admission_type_id_II']] = 4 

df.loc[df['admission_type_id'] == 6 , ['admission_type_id_II']] = 4

df.loc[df['admission_type_id'] == 8 , ['admission_type_id_II']] = 4

df.loc[df['admission_type_id'] == 4 , ['admission_type_id_II']] = 5

df.loc[df['admission_type_id'] == 7 , ['admission_type_id_II']] = 5

df['admission_type_id'] = df['admission_type_id_II']

df = df.drop(['admission_type_id_II'],axis = 1)



#Discharge disposition

# Discharge to Home = 1 -> agrupar 1, 6, 8

df['discharge_disposition_id_II'] = ""

df.loc[df['discharge_disposition_id'] == 1 , ['discharge_disposition_id_II']] = 1

df.loc[df['discharge_disposition_id'] == 6 , ['discharge_disposition_id_II']] = 1

df.loc[df['discharge_disposition_id'] == 8 , ['discharge_disposition_id_II']] = 1

# Agrupados por valors nulos, não mapeados etc.

df.loc[df['discharge_disposition_id'] == 18 , ['discharge_disposition_id_II']] = 3

df.loc[df['discharge_disposition_id'] == 25 , ['discharge_disposition_id_II']] = 3

df.loc[df['discharge_disposition_id'] == 26 , ['discharge_disposition_id_II']] = 3

# Outros

df.loc[(df['discharge_disposition_id'] != 1) & (df['discharge_disposition_id'] > 1) , ['discharge_disposition_id_II']] = 2

df['discharge_disposition_id'] = df['discharge_disposition_id_II']

df = df.drop(['discharge_disposition_id_II'],axis = 1)



#Admission source

df['admission_source_id_II'] = ""

df.loc[df['admission_source_id'] == 7, ['admission_source_id_II']] = 0

df.loc[df['admission_source_id'] == 1, ['admission_source_id_II']] = 1

df.loc[(df['admission_source_id'] > 1) & (df['admission_source_id'] < 7) | (df['admission_source_id'] > 7), ['admission_source_id_II']] = 2

df['admission_source_id'] = df['admission_source_id_II']

df = df.drop(['admission_source_id_II'],axis = 1)



# Variaveis (time_in_hospital_II); 

#Classes: 0: 0 - 4 ; 1: 4-8 ; 2: >=8

df['time_in_hospital_II'] = ""

df.loc[(df['time_in_hospital'] >= 0) & (df['time_in_hospital'] < 4), ['time_in_hospital_II']] = 0

df.loc[(df['time_in_hospital'] >= 4) & (df['time_in_hospital'] < 8), ['time_in_hospital_II']] = 1

df.loc[(df['time_in_hospital'] >= 8), ['time_in_hospital_II']] = 2  

df['time_in_hospital'] = df['time_in_hospital_II']

df = df.drop(['time_in_hospital_II'],axis = 1)



# Alterando variável payer_code (excluida)

df.loc[df['payer_code'] == 'MC', ['payer_code']] = 1

df.loc[df['payer_code'] == 'MD', ['payer_code']] = 2

df.loc[df['payer_code'] == 'HM', ['payer_code']] = 3

df.loc[df['payer_code'] == 'UN', ['payer_code']] = 4

df.loc[df['payer_code'] == 'BC', ['payer_code']] = 5

df.loc[df['payer_code'] == 'SP', ['payer_code']] = 6

df.loc[df['payer_code'] == 'CP', ['payer_code']] = 7

df.loc[df['payer_code'] == 'SI', ['payer_code']] = 8

df.loc[df['payer_code'] == 'DM', ['payer_code']] = 9

df.loc[df['payer_code'] == 'CM', ['payer_code']] = 10

df.loc[df['payer_code'] == 'CH', ['payer_code']] = 11

df.loc[df['payer_code'] == 'PO', ['payer_code']] = 12

df.loc[df['payer_code'] == 'WC', ['payer_code']] = 13

df.loc[df['payer_code'] == 'OT', ['payer_code']] = 14

df.loc[df['payer_code'] == 'OG', ['payer_code']] = 15

df.loc[df['payer_code'] == 'MP', ['payer_code']] = 16

df.loc[df['payer_code'] == 'FR', ['payer_code']] = 17



#Medical specialty



# Variaveis (num_lab_procedures_II); 

#Classes: 0: 0 - 30 ; 1: 30-60 ; 2: >=60

df['num_lab_procedures_II'] = ""

df.loc[(df['num_lab_procedures'] >= 0) & (df['num_lab_procedures'] < 30), ['num_lab_procedures_II']] = 0

df.loc[(df['num_lab_procedures'] >= 30) & (df['num_lab_procedures'] < 60), ['num_lab_procedures_II']] = 1

df.loc[(df['num_lab_procedures'] >= 60), ['num_lab_procedures_II']] = 2

df['num_lab_procedures'] = df['num_lab_procedures_II']

df = df.drop(['num_lab_procedures_II'],axis = 1)



# Variaveis (num_procedures_II); 

#Classes: 0: 0 - 2 ; 1: 2-4 ; 2: >=4

df['num_procedures_II'] = ""

df.loc[(df['num_procedures'] >= 0) & (df['num_procedures'] < 2), ['num_procedures_II']] = 0

df.loc[(df['num_procedures'] >= 2) & (df['num_procedures'] < 4), ['num_procedures_II']] = 1

df.loc[(df['num_procedures'] >= 4), ['num_procedures_II']] = 2

df['num_procedures'] = df['num_procedures_II']

df = df.drop(['num_procedures_II'],axis = 1)



# Variaveis (num_medications_II); 

#Classes: 0: 0 - 10 ; 1: 10-20 ; 2: >=20

df['num_medications_II'] = ""

df.loc[(df['num_medications'] >= 0) & (df['num_medications'] < 10), ['num_medications_II']] = 0

df.loc[(df['num_medications'] >= 10) & (df['num_medications'] < 20), ['num_medications_II']] = 1

df.loc[(df['num_medications'] >= 20), ['num_medications_II']] = 2

df['num_medications'] = df['num_medications_II']

df = df.drop(['num_medications_II'],axis = 1)



# Variaveis (number_outpatient_II); 

#Classes: 0: 0 - 10 ; 1: 10-20 ; 2: >= a 20

df['number_outpatient_II'] = ""

df.loc[(df['number_outpatient'] >= 0) & (df['number_outpatient'] < 10), ['number_outpatient_II']] = 0

df.loc[(df['number_outpatient'] >= 10) & (df['number_outpatient'] < 20), ['number_outpatient_II']] = 1

df.loc[(df['number_outpatient'] >= 20) , ['number_outpatient_II']] = 2

df['number_outpatient'] = df['number_outpatient_II']

df = df.drop(['number_outpatient_II'],axis = 1)

             

# Variaveis (number_emergency_II); 

#Classes: 0: 0 - 5 ; 1: 5-10 ; 2: >= 10

df['number_emergency_II'] = ""

df.loc[(df['number_emergency'] >= 0) & (df['number_emergency'] < 5), ['number_emergency_II']] = 0

df.loc[(df['number_emergency'] >= 5) & (df['number_emergency'] < 10), ['number_emergency_II']] = 1

df.loc[(df['number_emergency'] >= 10), ['number_emergency_II']] = 2

df['number_emergency'] = df['number_emergency_II']

df = df.drop(['number_emergency_II'],axis = 1)

        

# Variaveis (number_inpatient_II); 

#Classes: 0: 0 - 3 ; 1: 3-6 ; 2: >=6

df['number_inpatient_II'] = ""

df.loc[(df['number_inpatient'] >= 0) & (df['number_inpatient'] < 3), ['number_inpatient_II']] = 0

df.loc[(df['number_inpatient'] >= 3) & (df['number_inpatient'] < 6), ['number_inpatient_II']] = 1

df.loc[(df['number_inpatient'] >= 6), ['number_inpatient_II']] = 2

df['number_inpatient'] = df['number_inpatient_II']

df = df.drop(['number_inpatient_II'],axis = 1)



#Diagnosis 1

#Diagnosis 2

#Diagnosis 3



# Variaveis (number_diagnoses_II); 

#Classes: 0: 0 - 4 ; 1: 4-8 ; 2: >=8

df['number_diagnoses_II'] = ""

df.loc[(df['number_diagnoses'] >= 0) & (df['number_diagnoses'] < 4), ['number_diagnoses_II']] = 0

df.loc[(df['number_diagnoses'] >= 4) & (df['number_diagnoses'] < 8), ['number_diagnoses_II']] = 1

df.loc[(df['number_diagnoses'] >= 8), ['number_diagnoses_II']] = 2

df['number_diagnoses'] = df['number_diagnoses_II']

df = df.drop(['number_diagnoses_II'],axis = 1)





# Alterando variável MAX_GLU_SERUM

df.loc[df['max_glu_serum'] == 'None', ['max_glu_serum']] = 0

df.loc[df['max_glu_serum'] == 'Norm', ['max_glu_serum']] = 1

df.loc[df['max_glu_serum'] == '>200', ['max_glu_serum']] = 2

df.loc[df['max_glu_serum'] == '>300', ['max_glu_serum']] = 3



# Alterando variável A1Cresult

df.loc[df['A1Cresult'] == 'None', ['A1Cresult']] = 0

df.loc[df['A1Cresult'] == 'Norm', ['A1Cresult']] = 1

df.loc[df['A1Cresult'] == '>7', ['A1Cresult']] = 2

df.loc[df['A1Cresult'] == '>8', ['A1Cresult']] = 3



# Alterando diversas variáveis que têm classes iguais

metricas = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 

           "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 

           "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

for i in metricas:

    df.loc[df[i] == 'Up', [i]] = 1  #sim o medicamento foi prescito-aumentou a doze

    df.loc[df[i] == 'Down', [i]] = 1 #sim o medicamento foi prescito-baixou a doze

    df.loc[df[i] == 'Steady', [i]] = 1 #sim o medicamento foi prescito -manteve a doze

    df.loc[df[i] == 'No', [i]] = 0 #nao foi prescrito

    



# Alterando variável change

df.loc[df['change'] == 'No', ['change']] = 0

df.loc[df['change'] == 'Ch', ['change']] = 1

    

# Alterando variável DIABETESMED

df.loc[df['diabetesMed'] == 'Yes', ['diabetesMed']] = 1

df.loc[df['diabetesMed'] == 'No', ['diabetesMed']] = 0



# Transformando variável: readmitted; 

df.loc[df['readmitted'] == '<30', ['readmitted']] = 1 #0,087

df.loc[df['readmitted'] == '>30', ['readmitted']] = 0 #0,912

df.loc[df['readmitted'] == 'NO', ['readmitted']] = 0
df.readmitted.value_counts()
# Excluindo os pacientes que MORRERAM ou estão no HOSPICIO

#df = df[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]

df = df[~df.discharge_disposition_id.isin([4])]

# Mostrando quantidade de casa registro com as seguintes variáveis

print('4: ', df['discharge_disposition_id'][df['discharge_disposition_id'] == 4].count())
#OVER SAMPLING, BALANCEANDO OS DADOS



# Class count

count_class_0, count_class_1 = df.readmitted.value_counts()



# Divide by class

df_class_0 = df[df['readmitted'] == 0]

df_class_1 = df[df['readmitted'] == 1]

df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_b = pd.concat([df_class_0, df_class_1_over], axis=0)
#Under-sumpling

# Class count

#count_class_0, count_class_1 = df.readmitted.value_counts()



# Divide by class

#df_class_0 = df[df['readmitted'] == 0]

#df_class_1 = df[df['readmitted'] == 1]



#df_class_0_under = df_class_0.sample(count_class_1)

#df= pd.concat([df_class_0_under, df_class_1], axis=0)



#print('Random under-sampling:')

#print(df.readmitted.value_counts())



#df.readmitted.value_counts().plot(kind='bar', title='Count (target)');
# Corpo do dataset após tratamento dos dados

df_b.shape
#Visualizando variavel Readmitted

#df.readmitted.count()

df_b["readmitted"].value_counts()
# Importando as bibliotecas necessárias

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error
removed_cols = ['encounter_id',

                'patient_nbr', 

                'weight',

                'payer_code', 

                'medical_specialty',

                'number_outpatient', 

                'number_emergency',

                'diag_1',

                'diag_2',

                'diag_3',

                'chlorpropamide',

                'acetohexamide', 

                'tolbutamide',

                'miglitol', 

                'troglitazone',

                'tolazamide',

                'examide',

                'citoglipton', 

                'glipizide-metformin',

                'glimepiride-pioglitazone',

                'metformin-rosiglitazone',

                'metformin-pioglitazone', 

                'readmitted',

                'duplicado',

                'random'

               ]





#feats = [c for c in df.columns if c in colunas_transformadas]

feats = [c for c in df_b.columns if c not in removed_cols]
# Preenchendos os valores nulos com -1

df_b.fillna(-1, inplace=True)
# Dividindo o df em treino e validação

train, valid = train_test_split(df_b, random_state=42)
# Random Forest



# Carregando o modelo

rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=150, min_samples_split=3, oob_score = True)



# Treinando o modelo

rf.fit(train[feats], train['readmitted'])

preds = rf.predict(valid[feats])



print('Erro médio quadrado:', mean_squared_error(valid['readmitted'], preds)**(1/2))



print('Acurácia: ', accuracy_score(valid['readmitted'],preds))



fi = pd.DataFrame({'feature': feats, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)

fi = fi.reset_index()

#fi[:11]

fi.loc[fi['importance']>0.005, 'feature']
#analiusando a importancia das caracteristicas.

fig, ax = plt.subplots(figsize=(15,12))

# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+500, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

    

# Add Text watermark

fig.text(0.9, 0.15, '@bmanohar16', fontsize=12, color='grey',

         ha='right', va='bottom', alpha=0.5)





pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(color=(0.2, 0.4, 0.6, 0.6))



ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
def plot_feature_importance(fi):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24,8))

    ax1.plot(np.arange(0, len(fi.index)), fi['importance'])

    label_nrs = np.arange(0, len(fi.index), 5 )

    ax1.set_xticks(label_nrs)

    ax1.set_xticklabels(fi['feature'][label_nrs], rotation=90)

    

    num_bar = min(len(fi.index), 30)

    ax2.barh(np.arange(0, num_bar), fi['importance'][:num_bar], align='center', alpha=0.5)

    ax2.set_yticks(np.arange(0, num_bar))

    ax2.set_yticklabels(fi['feature'][:num_bar])



plot_feature_importance(fi)
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc

from sklearn import metrics

#auc_roc



# Logistic regression

false_positive_rate, true_positive_rate, thresholds = roc_curve(valid['readmitted'], preds)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('random forest classifier: ', roc_auc)
plt.figure(figsize=(10,10))

plt.title('Graph')

plt.plot(false_positive_rate,true_positive_rate, color='green',label = 'Random Forest Classifier = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')