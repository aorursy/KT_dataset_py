# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from glob import glob

base_skin_dir = os.path.join('..', 'input')



%matplotlib inline

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
# Datenbeschreibung visualisieren
# Zusammenfügen von HAM10000_images_part1.zip und HAM10000_images_part2.zip in einem Ordner



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
# Alle Daten von HAM10000_metadata.csv in die Variable skin_cancer_data einlesen



skin_cancer_data = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
# Oberen Teil der Tabelle mit Daten sehen



skin_cancer_data.head()
# Datentypen ausgeben lassen



print(skin_cancer_data_clean.dtypes)
# Datenaufbereitung
# Daten bereinigen von NAN bei Alter und unknown bei Geschlecht und Körperstelle



skin_cancer_data_without_NAN = skin_cancer_data.dropna(subset=['age'])

skin_cancer_data_without_NAN_and_unknown = skin_cancer_data_without_NAN[skin_cancer_data_without_NAN.sex != 'unknown']

skin_cancer_data_clean = skin_cancer_data_without_NAN_and_unknown[skin_cancer_data_without_NAN_and_unknown.localization != 'unknown']
# Schauen, dass in den Daten keine Null-Werte mehr enthalten sind



skin_cancer_data_clean.isnull().sum()
# Datenanalyse
# Hautkrebs-Typen im Graphen anzeigen lassen (alle Zeilen berücksichtigt)



fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))

skin_cancer_data['dx'].value_counts().plot(kind='bar', ax=ax1)
# Hautkrebs-Ermittlungsmethoden im Graphen anzeigen lassen (alle Zeilen berücksichtigt)



fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))

skin_cancer_data['dx_type'].value_counts().plot(kind='bar', ax=ax1)
# Alter im Graphen anzeigen lassen (Zeilen mit NAN-Werten sind nicht berücksichtigt)



fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))

skin_cancer_data_clean['age'].value_counts().plot(kind='bar', ax=ax1)
# Geschlecht im Graphen anzeigen lassen (Zeilen mit unknown-Werten sind nicht berücksichtigt)



fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))

skin_cancer_data_clean['sex'].value_counts().plot(kind='bar', ax=ax1)
# Anzahl der Befragten berechnen



print (skin_cancer_data_clean['sex'].value_counts())
# Körperstellen im Graphen anzeigen lassen (Zeilen mit unknown-Werten sind nicht berücksichtigt)



fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))

skin_cancer_data_clean['localization'].value_counts().plot(kind='bar', ax=ax1)