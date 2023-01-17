# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import pandas as pd

import seaborn as sns

import shap

import matplotlib.pyplot as plt

import numpy as np





# sklearn libs compatilhadas

import sklearn

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, train_test_split, RandomizedSearchCV

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler



## Tratando dados desbalanceados

from imblearn.over_sampling import SMOTE









### Clustenização 1

from sklearn.cluster import KMeans 

from sklearn import metrics 

from scipy.spatial.distance import cdist 







## Classificacao 1

from sklearn.linear_model import LogisticRegression



## Classificação 2

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from lightgbm import LGBMClassifier





from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

shap.initjs()


def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred, normalize="true")

    df = pd.DataFrame(cm, index=["no", "yes"], columns=["no", "yes"])

    ax = sns.heatmap(df, annot=True)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    return ax



def plot_roc(y_true, y_score, figsize=(8, 8)):

    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)

    

    plt.figure(figsize=figsize)

    plt.plot(fpr, tpr, color='darkorange',

             lw=2, label=f'ROC curve (AUC = {100*roc_auc:.2f}%)')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")

    plt.show()

    

    return roc_auc





def plot_precision_recall(precisions, recalls, thresholds):

    fig, ax = plt.subplots(figsize=(12,8))

    ax.plot(thresholds, precisions[:-1], "r--", label="Precisions")

    ax.plot(thresholds, recalls[:-1], "#424242", label="Recalls")

    ax.set_title("Precision and Recall \n Tradeoff", fontsize=18)

    ax.set_ylabel("Level of Precision and Recall", fontsize=16)

    ax.set_xlabel("Thresholds", fontsize=16)

    ax.legend(loc="best", fontsize=14)

    ax.set_xlim([0, 1])

    ax.set_ylim([0, 1])

    return ax



def plot_confusion_matrix2(y_test, y_pred, figsize=(16,16),names=False):

    fig, ax = plt.subplots(figsize = figsize)

    cm = confusion_matrix(y_test, y_pred, normalize="true")

    if names:

        df = pd.DataFrame(cm, index=names, columns=names)

    else:

        df = pd.DataFrame(cm)#, index=["no", "yes"], columns=["no", "yes"])    

    ax = sns.heatmap(df, annot=True,ax=ax)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    return ax
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.columns = ['Patient ID',

 'Patient_age_quantile',

 'SARS-Cov-2_result',

 'Patient_addmited_regular_ward_care_bool',

 'Patient_addmited_semi-intensive_care_bool',

 'Patient_addmited_intensive_care_bool',

 'Hematocrit',

 'Hemoglobin',

 'Platelets',

 'Mean_platelet_volume',

 'Red_blood_Cells',

 'Lymphocytes',

 'MCHC',

 'Leukocytes',

 'Basophils',

 'MCH',

 'Eosinophils',

 'MCV',

 'Monocytes',

 'RDW',

 'Serum Glucose',

 'Respiratory_Syncytial Virus',

 'Influenza_A',

 'Influenza_B',

 'Parainfluenza_1',

 'CoronavirusNL63',

 'Rhinovirus_Enterovirus',

 'Mycoplasma_pneumoniae',

 'Coronavirus_HKU1',

 'Parainfluenza_3',

 'Chlamydophila_pneumoniae',

 'Adenovirus',

 'Parainfluenza 4',

 'Coronavirus229E',

 'CoronavirusOC43',

 'Inf_A_H1N1_2009',

 'Bordetella_ertussis',

 'Metapneumovirus',

 'Parainfluenza_2',

 'Neutrophils',

 'Urea',

 'Proteina_C',

 'Creatinine',

 'Potassium',

 'Sodium',

 'Influenza_B_test',

 'Influenza_A_test',

 'Alanine_ransaminase',

 'Aspartate_transaminase',

 'Gamma-glutamyltransferase',

 'Total_Bilirubin',

 'Direct_Bilirubin',

 'Indirect_Bilirubin',

 'Alkaline_phosphatase',

 'Ionized_calcium',

 'Strepto_A',

 'Magnesium_',

 'pCO2_',

 'Hb_saturation_',

 'Bas_excess_',

 'pO2_',

 'Fio2_',

 'Total_CO2_',

 'pH_',

 'HCO3_venon',

 'Rods',

 'Segmented',

 'Promyelocytes',

 'Metamyelocytes',

 'Myelocytes',

 'Myeloblasts',

 'Urine_Esterase',

 'Urine_Aspect',

 'Urine_pH',

 'Urine_Hemoglobin',

 'Urine_Bile pigments',

 'Urine_Ketone Bodies',

 'Urine_Nitrite',

 'Urine_Density',

 'Urine_Urobilinogen',

 'Urine_Protein',

 'Urine_Sugar',

 'Urine_Leukocytes',

 'Urine_Crystals',

 'Urine_Red blood cells',

 'Urine_Hyaline cylinders',

 'Urine_Granular cylinders',

 'Urine_Yeasts',

 'Urine_Color',

 'Partial_thromboplastin_time',

 'Relationship',

 'INR',

 'Lactic_Dehydrogenase',

 'Prothrombin_time (PT)',

 'Vitamin_B12',

 'Creatine_phosphokinase',

 'Ferritin',

 'Lactic_Acid',

 'Lipase_dosage',

 'D-Dimer',

 'Albumin',

 'Hb_saturation',

 'pCO2',

 'Base_excess',

 'pH',

 'Total_CO2',

 'HCO3_artery',

 'pO2',

 'Arteiral_Fio2',

 'Phosphor',

 'ctO2']
# sorted([[df.shape[0]-j,i] for i,j in df.isna().sum().items() if j > 0])





# [0, 'D-Dimer'],

#  [0, 'Mycoplasma_pneumoniae'],

#  [0, 'Partial_thromboplastin_time'],

#  [0, 'Prothrombin_time (PT)'],

#  [0, 'Urine_Sugar'],

#  [1, 'Fio2_'],

#  [1, 'Urine_Nitrite'],

#  [3, 'Vitamin_B12'],

#  [8, 'Lipase_dosage'],
## 

df.drop(['Prothrombin_time (PT)', 'D-Dimer', 'Mycoplasma_pneumoniae', 'Urine_Sugar', 'Partial_thromboplastin_time', 'Fio2_', 'Urine_Nitrite', 'Vitamin_B12'], axis = 1, inplace = True)
## Preenchendo as colunas  dos testes que não foram feitos.

columns_to_fill = pd.DataFrame(df.isna().sum()/df.shape[0], columns=['Missing'])

columns_to_fill = columns_to_fill[(columns_to_fill.Missing < 0.87)].index



for col in columns_to_fill:

    df[col] = df[col].fillna('not_done')
## Map CATEGORICO

fullMapper={'negative': 0, 'positive': 1,

           'not_detected': 0, 'detected': 1,

            'not_done': -1, 'absent': -1,

            'Não Realizado': -1,

               ## Urine Aspects

              'clear': 0,

              'cloudy': 1,

              'lightly_cloudy': 2,

              'altered_coloring': 3,

               #Urine_Leukocytes

               '<1000': 1000,

                #Urine_urobilinogen

                'normal':0,

            #'Urine_Crystals': {

              'Ausentes': 0,

              'Urato Amorfo --+': 1,

              'Urato Amorfo +++': 2,

              'Oxalato de Cálcio +++': 3,

              'Oxalato de Cálcio -++': 4,

            # Urine_Color

              'yellow': 0,

              'light_yellow': 1,

              'orange': 2,

              'citrus_yellow': 3,

            # Urine_Hemoglobin

            'present': 1,

               }



df.replace(fullMapper, inplace=True)
## Preenchendo os exames de sangue com base na média por idade

columns_to_fill = pd.DataFrame(df.isna().sum()/df.shape[0], columns=['Missing'])

columns_to_fill = columns_to_fill[columns_to_fill.Missing > 0.87].index



#Prenche quando tiver o dado da idade

for value in df.Patient_age_quantile.unique():

    df_aux = df[df.Patient_age_quantile == value].copy()

    

    for col in columns_to_fill:

        df_aux[col] = df_aux[col].fillna(df_aux[col].median())

        

    df.loc[df_aux.index] = df_aux



#Preenche com a media geral

for col in columns_to_fill:

    df[col] = df[col].fillna(df[col].median())
df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals']]  = df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals']].astype('category')
df.isna().sum().sum()
X = df.drop(['Patient ID','SARS-Cov-2_result', 'Patient_addmited_regular_ward_care_bool',

       'Patient_addmited_semi-intensive_care_bool',

       'Patient_addmited_intensive_care_bool'], axis=1)

y = df['SARS-Cov-2_result']

X_scaled = StandardScaler().fit_transform(X)
os = SMOTE(random_state=0)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

columns = X.columns



os_data_X, os_data_y = os.fit_sample(X_train_all, y_train_all)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns)

os_data_y = pd.DataFrame(data=os_data_y)



X_train_all = os_data_X

y_train_all = os_data_y['SARS-Cov-2_result']
print(f"""

tamanho do dataset de treino:

X_train:{X_train_all.shape}

y_train:{y_train_all.shape}

~~~~~~~

tamanho do dataset de teste:

X_test:{X_test_all.shape}

y_test:{y_test_all.shape}

""")
X_train = X_train_all.copy()

y_train = y_train_all.copy()

X_test= X_test_all.copy()

y_test=y_test_all.copy()



# params = dict(

#     n_estimators=[150,500,1000],

#     max_depth=[3, 5, 10],

#     min_samples_split=[2,50],

#     min_samples_leaf=[1,5,10],

# )

# model = RandomForestClassifier(n_jobs=-1, random_state=42)

# grid = GridSearchCV(model, param_grid = params,verbose=True, n_jobs=-1, return_train_score= True)

# grid.fit(X_train, y_train)

best_params = {'max_depth': 10,

 'min_samples_leaf': 1,

 'min_samples_split': 2,

 'n_estimators': 500}



model = RandomForestClassifier(**best_params,n_jobs=-1,verbose=0, random_state=42).fit(X_train, y_train)

y_pred = model.predict_proba(X_test)
pred_train = model.predict(X_train)

scores = sklearn.metrics.accuracy_score(y_train, pred_train)

print('Accuracy on training data: {:.2f}%'.format(scores))   

 

pred_test = model.predict(X_test)

scores2 = sklearn.metrics.accuracy_score(y_test, pred_test)

print('Accuracy on test data: {:.2f}%'.format(scores2))    
_ =plot_roc(y_test, y_pred[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred[:,1])

_ =plot_precision_recall(precisions, recalls, thresholds)
threshold = 0.6



y_pred_ = y_pred[:,1] > threshold



_ =plot_confusion_matrix(y_test, y_pred_)

confusion_matrix(y_test, y_pred_)
X_train = X_train_all.copy()

y_train = y_train_all.copy()

X_test= X_test_all.copy()

y_test_=y_test_all.copy()

# one hot encode outputs

y_train = to_categorical(y_train)

y_test = to_categorical(y_test_)



count_classes = y_test.shape[1]

print(count_classes)


# build the model

model = Sequential()

model.add(Dense(250, activation='relu', input_dim=(X_train.shape[1])))

model.add(Dropout(.2))

model.add(Dense(200, activation='relu'))

model.add(Dropout(.2))

model.add(Dense(200, activation='tanh'))

model.add(Dropout(.2))

model.add(Dense(100, activation='relu'))

model.add(Dropout(.3))

model.add(Dense(50, activation='relu'))

model.add(Dense(2, activation='softmax'))



# Compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#Fitting The Mdoel

model.fit(X_train, y_train, epochs=300, verbose = True)
pred_train = model.predict(X_train)

scores = model.evaluate(X_train, y_train, verbose=0)

print('Accuracy on training data: {:.2f}% \n Error on training data: {:.2f}'.format(scores[1], 1 - scores[1]))   

 

pred_test = model.predict(X_test)

scores2 = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy on test data: {:.2f}% \n Error on test data: {:.2f}'.format(scores2[1], 1 - scores2[1]))    
y_pred =  model.predict(X_test)
y_pred[:,1]

y_test_
_ =plot_roc(y_test_, y_pred[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test_, y_pred[:,1])

_ =plot_precision_recall(precisions, recalls, thresholds)
threshold = 0.57



y_pred_ = y_pred[:,1] > threshold



_ =plot_confusion_matrix(y_test_, y_pred_)

confusion_matrix(y_test_, y_pred_)
## Pessoas sem covid

df['grau_doenca'] = 0





## Pessoas sem covid mas em leitos hospitalares

ind = (df[df['SARS-Cov-2_result'] == 0][['Patient_addmited_regular_ward_care_bool',

       'Patient_addmited_semi-intensive_care_bool',

       'Patient_addmited_intensive_care_bool']].sum(axis=1) == 1)

ind = [ind for ind, boolean in ind.items() if boolean == 1]

df.loc[ind, 'grau_doenca'] = 1







# ## Pessoas com covid mas fora de leito

ind = (df[df['SARS-Cov-2_result'] == 1][['Patient_addmited_regular_ward_care_bool',

       'Patient_addmited_semi-intensive_care_bool',

       'Patient_addmited_intensive_care_bool']].sum(axis=1) == 0)



ind = [ind for ind, boolean in ind.items() if boolean == 1]

df.loc[ind, 'grau_doenca'] = 2









# ## Pessoas com covid em leito

ind = (df[df['SARS-Cov-2_result'] == 1][['Patient_addmited_regular_ward_care_bool',

       'Patient_addmited_semi-intensive_care_bool',

       'Patient_addmited_intensive_care_bool']].sum(axis=1) != 0)



ind = [ind for ind, boolean in ind.items() if boolean == True]

df.loc[ind, 'grau_doenca'] = 3



dic={0: 'sem doença',

2: 'com covid leve',

1: 'internado sem covid',

3: 'covid internado'}



{dic[i]:j for i,j in  df['grau_doenca'].value_counts().items()}
X = df.drop(['Patient ID','SARS-Cov-2_result', 'Patient_addmited_regular_ward_care_bool',

       'Patient_addmited_semi-intensive_care_bool',

       'Patient_addmited_intensive_care_bool', 'grau_doenca'], axis=1)

y = df['grau_doenca']

X_scaled = StandardScaler().fit_transform(X)
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_scaled, y, test_size=0.3, random_state=30)

columns = X.columns



os_data_X, os_data_y = os.fit_sample(X_train_all, y_train_all)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns)

os_data_y = pd.DataFrame(data=os_data_y)



X_train_all = os_data_X

y_train_all = os_data_y['grau_doenca']





print('items no dataset de teste')

{dic[i]:j for i,j in  y_test_all.value_counts().items()}
X_train = X_train_all.copy()

y_train_ = y_train_all.copy()

X_test= X_test_all.copy()

y_test_= y_test_all.copy()



# one hot encode outputs

y_train = to_categorical(y_train_)

y_test = to_categorical(y_test_)

y_test_=y_test_.values

count_classes = y_test.shape[1]

print(count_classes)
# build the model

model = Sequential()

model.add(Dense(250, activation='tanh', input_dim=(X_train.shape[1])))

model.add(Dropout(.2))

model.add(Dense(250, activation='relu'))

model.add(Dropout(.2))

model.add(Dense(200, activation='relu'))

model.add(Dropout(.2))

model.add(Dense(100, activation='tanh'))

model.add(Dropout(.3))

model.add(Dense(50, activation='relu'))

model.add(Dense(count_classes, activation='softmax'))



# Compile the model

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])



#Fitting The Mdoel

history = model.fit(X_train, y_train, epochs=150, verbose = True)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.title('model metrics')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['accuracy', 'loss'], loc='upper left')

plt.show()
pred_train = model.predict(X_train)

scores = sklearn.metrics.accuracy_score(y_train_, pred_train.argmax(axis=1))

print('Accuracy on training data: {:.2f}%'.format(scores))   

 

pred_test = model.predict(X_test)

scores2 = sklearn.metrics.accuracy_score(y_test_, pred_test.argmax(axis=1))

print('Accuracy on test data: {:.2f}%'.format(scores2))
names=['sem doença',

'covid leve',

'internado sem covid',

'covid grave']

plot_confusion_matrix2(y_test_, pred_test.argmax(axis=1), figsize= (12,12),names=names)
explainer = shap.KernelExplainer(model.predict, X_train[:150])
shap_values = explainer.shap_values(X_test[:150], nsamples=150)
shap.summary_plot(shap_values[0], X_test[:150])

shap.summary_plot(shap_values[1], X_test[:150])

shap.summary_plot(shap_values[2], X_test[:150])

shap.summary_plot(shap_values[3], X_test[:150])