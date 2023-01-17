# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline

sns.set()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df = pd.read_json("../input/houses.json")



# Any results you write to the current directory are saved as output.
#Vorbereitung

#Einstellungen

# Festlegung der Anzahl der Kreuzvalidierunegen im Modellierungsteil 

nr_cv = 5



# Ziel für die Korrelation  

target = 'ExactPreis'



#Schwellwert der Korrelation

min_val_corr = 0.01   
#Vorbereitung

#Benötigte Funktionen

def plot_corr_matrix(df, nr_c, targ) :

    

    corr = df.corr()

    corr_abs = corr.abs()

    cols = corr_abs.nlargest(nr_c, targ)[targ].index

    cm = np.corrcoef(df[cols].values.T)



    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))

    sns.set(font_scale=1.25)

    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values

               )

    plt.show()
#Vorbereitung

#Benötigte Funktionenen



def get_best_score(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score
#DatenAnalyse -- 1.1. Überblick über die Daten und fehlende Werte

#Wie viele Zeilen und Spalten gibt es?

print(df.shape)

print("*"*80)



#Welche Spalten gibt es? Wie Felder haben einen Wert enthalten? 

print(df.info())
#DatenAnalyse

#Anzeige der ersten Zeilen im Datensatz

df.head()
#DatenAnalyse

#Überblick über die Statistiken (z.B kleinster, größter Wert, Durchschnittswert(mean) etc.) (nur für numerische Spalten)

#Kann man vielleicht in Arbeit nennen und ein bis zwei Beispielwerte 

df.describe()
#DatenAnalyse

#Wie viele Merkmale sind numerisch, welche kategorisch?



numerical_feats = df.dtypes[df.dtypes != "object"].index

print("Anzahl numerischer Merkmale: ", len(numerical_feats))



categorical_feats = df.dtypes[df.dtypes == "object"].index

print("Anzahl kategorischer Merkmale: ", len(categorical_feats))

print("*"*80)



#Welche Merkmale sind numerisch, welche kategorisch?

print(df[numerical_feats].columns)

print("*"*80)

print(df[categorical_feats].columns)



#Die ersten Datensätze mit nur numerischen Merkmalen

df[numerical_feats].head()
#DatenAnalyse

#Die ersten Datensätze mit nur kategorischen Merkmalen

df[categorical_feats].head()
#DatenAnalyse

#Liste/Tabelle der Merkmale mit fehlenden Werten und die Anzahl der fehlenden Werte und die Prozentanzahl

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Anzahl(NaN)', 'Prozent'])

missing_data.head(50)



#Für einige Spalten gibt es viele NaN-Einträge. Es ist dabei gemeint, dass keine Daten fehlen,

#sondern wie z.B. beim Attribut Barrierefrei, die Immoblie nicht Barrierefrei ist
#DatenAnalyse 

#-->Ziel: keine leeren Einträge in den Feldern

# Spalten, wo NaN eine Bedeutung hat:

cols_fillna = ['Aktuell_vermietet','Badewanne','Balkon','Barrierefrei','Dachboden','Denkmalobjekt',

               'Dusche','Einbaukueche','Einliegerwohnung','Gaeste_WC', 'Garage/Stellplatz',

               'Garten/_mitnutzung', 'Haustiere_erlaubt', 'Heizungsart', 'Keller', 'Moebliert/Teilmoebliert',

               'Terrasse','WG_geeignet']



# In diesen Spalten wird NaN durch "false"

for col in cols_fillna:

    df[col].fillna('false',inplace=True)

    df[col].fillna('false',inplace=True)



#Merkmale die weiterhin noch NaN Eintäge besitzen Anzeigen

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Anzahl(NaN)', 'Prozent'])

missing_data.head(20)
#DatenAnalyse

#Saplten, die auf Grund von zu vielen fehlenden Daten unbrauchbar sind:

cols_unusable = ['Heizkosten__in_€_','Warmmiete__in_€_', 'Nebenkosten__in_€_', 'Kaution__in_€_', 

                 'Verfuegbar_ab_Monat', 'Verfuegbar_ab_Jahr', 'Provision', 'Preis']



#Spalten, mit unbrauchbaren Daten löschen 

df.drop(cols_unusable, inplace= True, axis = 1)



#Merkmale die weiterhin noch NaN Eintäge besitzen Anzeigen

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Anzahl(NaN)', 'Prozent'])

missing_data.head(10)
#DatenAnalyse

#Zeilen in denen das Merkmal ExactPreis NaN ist, löschen --> weil nicht brauchbar

df.dropna(subset=['ExactPreis'], inplace= True, axis=0)



#Merkmale die weiterhin noch NaN Eintäge besitzen Anzeigen

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Anzahl(NaN)', 'Prozent'])

missing_data.head(5)
#DatenAnalyse

#Den Mittelwert für die restlichen Spalten berechnenen mit NaN: Baujahr und Grundstuecksflaeche__m²_

df.fillna(df.mean(), inplace=True)



#Merkmale die weiterhin noch NaN Eintäge besitzen Anzeigen

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Anzahl(NaN)', 'Prozent'])

missing_data.head(5)
#DatenAnalyse

#Gibt es trotzdem noch fehlende Werte im Datensatz? Gibt die Anzahl möglicher leerer Felder aus

df.isnull().sum().sum()

#DatenAnalyse



#Es ist au0erdem aufgefallen, dass es Immobilien mit nur vierstelligen Postleitzahlen gibt, was heutzutage nicht mehr möglich ist, 

#weshalb diese Datensätze gelöscht werden. Des weiteren ist aufgefallen, dass es Immobilien für ein paar Hundert Euro gibt,

#die auch gelöscht werden. Es wurde sich für einen Minimalpreis von 50.000 Euro entschieden

df.drop(df[df.ExactPreis < 50000].index, inplace=True)

df.drop(df[df.plz < 10000].index, inplace=True)
#DatenAnalyse -- 1.2 Relationen der Merkmale zum Zielmerkmal (ExactPreis)



#Diagramme der Relation zum Zielmerkmal für alle numerischen Merkmale

nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



numerical_feats = df.dtypes[df.dtypes != "object"].index

li_num_feats = list(numerical_feats)

li_not_plot = ['ExactPreis']

li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]





for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(df[li_plot_num_feats[i]], df[target], ax = axs[r][c])

            stp = stats.pearsonr(df[li_plot_num_feats[i]], df[target])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()  
#Schlussfolgerung aus Korrelation der nummerischen Merkmale:

#für einige Merkmale wie z.B. 'Grundstuecksflaeche_m^2_' ist eine leichte lineare Korrelation zum Zielmerkmal erkennbar

#für andere Merkmale wie 'adid' ist die Korrelation sehr schwach.



#daher werden nur die Merkmale für die Vorhersage des Preises verwendet, die eine Korrelation größer als 0.01 haben.



#Daraus lässt sich ableiten, dass später (Daten Preparation) die Spalten gelöscht werden, die eine Korrelation kleiner 0.01 

#haben; welche Spalten das genau sind, siehe nächster Schritt



#auch ersichtlich, dass Einträge für einige nummerische Saplten tatsächlich kategorische Werte sind:

#z.B. repräsentiert das Merkmale 'Wohnflaeche_m^2_'bestimmte Gruppen für dieses Merkmal
#DatenAnalyse



#Es wird eine Liste mit allen Merkmalen und dessen Korrelationen ausgegeben

corr = df.corr()

corr_abs = corr.abs()



nr_num_cols = len(numerical_feats)

ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]



cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)

cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)



#Es werden zwei Listen mit den Merkmalen einmal über dem bestimmten Schwellwert und einmal

#unter dem Schwellwert (0.01) ausgegeben



print(ser_corr)

print("*"*80)

print("Liste der nummerischen Merkmale mit dem Korrelationskoeffizienten über min_val_corr (0.01): ")

print(cols_abv_corr_limit)

print("*"*80)

print("Liste der nummerischen Merkmale mit dem Korrelationskoeffizienten unter min_val_corr (0.01): ")

print(cols_bel_corr_limit)

#DatenAnalyse

#Nach Betrachtung der Daten wird entschlossen, zum späteren Zeitpunkt die Spalten neben dem Merkmal adid (wegen Schwellwert)

#auch das Mermal 'posterid' herauszunehmen, da ihm die Sinnhaftigkeit fehlt
#DatenAnalyse

#Es wird eine Liste ausgegeben, die die kategorischen Merkmale und ihrer eindeutigen Werte und dessen Anzahl angibt

categorical_feats = df.dtypes[df.dtypes == "object"].index

for catg in list(categorical_feats) :

    print(df[catg].value_counts())

    print('#'*50)
#DatenAnalyse

#Das Merkmal 'kw' wird an dieser Stelle rausgeschmissen. Nach Sichtung der vorherigen Ausgabe wird deutlich, dass 'kw' als

#eine Art Kommentarfeld verwendet wurde. Daher ist es unbrauchbar für den späteren Lernalgorithmus und eine Korrelationsanalyse



#Löschen von 'kw'

df.drop('kw', inplace= True, axis = 1)
#DatenAnalyse

#Nach Betrachtung der Daten, wird an dieser Stelle aus logischen Gründen ebenenfalls alle Zeilen rausgeschmissen,

#wo als Angebotstyp Gesuche drinstehen, danach wird ebenfalls der Angebotstyp rausgeschmissen 



df[~df.Angebotstyp.str.contains("Gesuche")]

df.drop('Angebotstyp', inplace= True, axis = 1)
#DatenAnalyse

#Diagramme zeigen die Relation von den kategorischen Merkmalen zum Zielattribut

categorical_feats = df.dtypes[df.dtypes == "object"].index

li_cat_feats = list(categorical_feats)

nr_rows = 9

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=target, data=df, ax = axs[r][c])

    

plt.tight_layout()    

plt.show() 
#DatenAnalyse

#Schlussfolgerung der Korellationen: 



#Merkmale mit einer starken Korellation sind: Badewanne,Balkon, Barrierefrei, Dachboden, Denkmalobjekt, Dusche, Einbauküche,

#Einliegerwohnung, Gaeste_WC, Garage/Stellplatz, Garten/_mitnutzung, Haustiere_erlaubt, Haustyp,.....



#für weitere Arbeit werden auch nur die Merkmale verwendet, die eine Mindestkorellation mit dem Zielmerkmal haben. 

#Die übrigen Mermale werden ebenfalls im nächsten Schritt gelöscht. Das sind: Aktuell_vermietet, Angebotstyp(Besonderheit),

#Verkaeufer,abtest, elatic Search, yo_m, yo_s



#Es wird außerdem deutlich, dass die Angaben der Zimmer überprüft und bereinigt werden müssen, da unlogische Werte enthalten 

#sind
#DatenAnalyse

#Zurodnung der kategorischen Merkmale in Liste mit starken und schwachen Korrelationen



catg_strong_corr = ['Badewanne', 'Balkon', 'Barrierefrei', 'Dachboden', 'Denkmalobjekt', 'Dusche', 'Einbaukueche',

                    'Einliegerwohnung', 'Gaeste_WC', 'Garage/Stellplatz', 'Garten/_mitnutzung', 'Haustiere_erlaubt',

                    'Haustyp', 'Heizungsart', 'Keller', 'Moebliert/Teilmoebliert', 'Terrasse','WG_geeignet', 'Zimmer']

catg_weak_corr = ['Aktuell_vermietet', 'Verkaeufer', 'abtest', 'elasticSearch', 'yo_m', 'yo_s']





print("Liste der kategorischen Merkmale mit starken Korrelationen:")

print(catg_strong_corr)

print("*"*80)

print("Liste der kategorischen Merkmale mit schwachen Korrelationen:")

print(catg_weak_corr)



#DatenPreparation

#Zuvor wurde festgelegt, dass die Spalten, numerische, als auch kategroische, gelöscht werden, die eine geringe Korrelation

#zum Zielmerkmal aufweise. Diese werden in diesem Schritt gelöscht, sowie das Merkmal 'posterid'.



to_drop_num  = cols_bel_corr_limit

to_drop_catg = catg_weak_corr



cols_to_drop = to_drop_num + to_drop_catg +['posterid']



df.drop(cols_to_drop, inplace= True, axis = 1)



#Es werden die übrig gebliebenen Mermale ausgegeben

print(df.columns)
#DataPreparation

#Zuvor ist aufgefallen, dass die Daten im Mermal 'Zimmer' unrein sind. Zunächst wird das Merkmal Zimmer in den Datentyp

#float umgewandelt. An dieser Stelle werden alle Zeilen gelöscht, in denen

#das Attribut Zimmer einen Wert kleiner 1 oder größer 20 ist. Eine Immobilie mit mehreren Zimmer scheint umwahrschienlich bzw.

#handelt es sich dabei auch um Tippfehler



df['Zimmer'] = df.Zimmer.astype(float)

df.drop(df[df.Zimmer > 20].index, inplace=True)

df.drop(df[df.Zimmer < 1].index, inplace=True)



#Nur zu Überprüfen

df.Zimmer.dtypes
#DatenPreparation

#An dieser Stelle beginnt die Umwandlung der kategorischen Merkmale in numerische. 

#Die Liste die im Folgenden Ausgegeben wird zeigt den Mittelwert der Zielmerkmals in bezug auf den jeweiligen Wert des Merkmals

#z.B. ist der Mittelwert des Preises, wenn keine Badewanne vorhanden ist 256968.53....



catg_list = catg_strong_corr.copy()

catg_list.remove('Zimmer')

for catg in catg_list :

    g = df.groupby(catg)[target].mean()

    print(g)

    print("*"*80)
#DatenPreparation

#weitere Schritte für die Umwandlung. Hier wurden die bisherigen Werte teilweise kategoriesiert

#bei ja/nein Merkmalen, fand keine wirkliche kategorisiert statt.

#Kategorisierung hauptsächlich bei Haustyp und Heinzungsart; 

#dort wurden anhand der vorher ausgegeben Mittelwerte, Gruppen gebildet



#Badewanne

bdw_true = ['true']

bdw_false = ['false'] 



#Balkon

balk_true = ['true']

balk_false = ['false'] 



#Barrierefrei

bar_true = ['true']

bar_false = ['false'] 



#Dachboden

dbo_true = ['true']

dbo_false = ['false'] 



#Denkmal

denk_true = ['true']

denk_false = ['false'] 



#Dusche

du_true = ['true']

du_false = ['false'] 



#Einbauküche

ebk_true = ['true']

ebk_false = ['false'] 



#Einliegerwohnung

elw_true = ['true']

elw_false = ['false'] 



#Gaeste_WC

gwc_true = ['true']

gwc_false = ['false'] 



#Garage/Stellplatz

stl_true = ['true']

stl_false = ['false'] 



#Garten/_mitnutzung

gar_true = ['true']

gar_false = ['false'] 



#Haustiere_erlaubt

hti_true = ['true']

hti_false = ['false'] 



#Keller

klr_true = ['true']

klr_false = ['false'] 



#Moebliert/Teilmoebliert

moe_true = ['true']

moe_false = ['false'] 



#Terrasse

ter_true = ['true']

ter_false = ['false']



#WG_geeignet

wgg_true = ['true']

wgg_false = ['false'] 



#Haustyp

htp_kat1 = ['bauernhaus']

htp_kat2 = ['reihenhaus', 'bungalow', 'andere']

htp_kat3 = ['doppelhaushaelfte', 'einfamilienhaus']

htp_kat4 = ['mehrfamilienhaus']

htp_kat5 = ['villa']



#Heizungsart

har_kat1 = ['ofenheizung', 'elektroheizung']

har_kat2 = ['andere', 'gasheizung', 'oelheizung', 'etagenheizung']

har_kat3 = ['false']

har_kat4 = ['zentralheizung', 'fussbodenheizung']

har_kat5 = ['fernwaerme']

#DatenPreparation

#weitere Schritte für die Umwandlung. Hier werden die neuen numerischen Saplten eingefügt und

#mit den Werten 1 für true und 0 für false bzw. den Kategorien Werte 1-5 gefüllt.

for df in [df]: 

    

    #Badewanne

    df['BDW_num'] = 1  

    df.loc[(df['Badewanne'].isin(bdw_true) ), 'BDW_num'] = 1    

    df.loc[(df['Badewanne'].isin(bdw_false) ), 'BDW_num'] = 0 

    

    #Balkon

    df['BALK_num'] = 1  

    df.loc[(df['Balkon'].isin(balk_true) ), 'BALK_num'] = 1    

    df.loc[(df['Balkon'].isin(balk_false) ), 'BALK_num'] = 0  

        

    #Barrierefrei

    df['BAR_num'] = 1  

    df.loc[(df['Barrierefrei'].isin(bar_true) ), 'BAR_num'] = 1    

    df.loc[(df['Barrierefrei'].isin(bar_false) ), 'BAR_num'] = 0  

    

    #Dachboden

    df['DBO_num'] = 1  

    df.loc[(df['Dachboden'].isin(dbo_true) ), 'DBO_num'] = 1    

    df.loc[(df['Dachboden'].isin(dbo_false) ), 'DBO_num'] = 0  

    

    #Denkmal

    df['DENK_num'] = 1  

    df.loc[(df['Denkmalobjekt'].isin(denk_true) ), 'DENK_num'] = 1    

    df.loc[(df['Denkmalobjekt'].isin(denk_false) ), 'DENK_num'] = 0 

    

    #Dusche

    df['DU_num'] = 1  

    df.loc[(df['Dusche'].isin(du_true) ), 'DU_num'] = 1    

    df.loc[(df['Dusche'].isin(du_false) ), 'DU_num'] = 0 

    

    #Einbaukueche

    df['EBK_num'] = 1  

    df.loc[(df['Einbaukueche'].isin(ebk_true) ), 'EBK_num'] = 1    

    df.loc[(df['Einbaukueche'].isin(ebk_false) ), 'EBK_num'] = 0

    

    #Einliegerwohnung

    df['ELW_num'] = 1  

    df.loc[(df['Einliegerwohnung'].isin(elw_true) ), 'ELW_num'] = 1    

    df.loc[(df['Einliegerwohnung'].isin(elw_false) ), 'ELW_num'] = 0  

    

    #Gaeste_WC

    df['GWC_num'] = 1  

    df.loc[(df['Gaeste_WC'].isin(gwc_true) ), 'GWC_num'] = 1    

    df.loc[(df['Gaeste_WC'].isin(gwc_false) ), 'GWC_num'] = 0 

    

    #Garage/Stellplatz

    df['STL_num'] = 1  

    df.loc[(df['Garage/Stellplatz'].isin(stl_true) ), 'STL_num'] = 1    

    df.loc[(df['Garage/Stellplatz'].isin(stl_false) ), 'STL_num'] = 0  

    

    #Garten/_mitnutzung

    df['GAR_num'] = 1  

    df.loc[(df['Garten/_mitnutzung'].isin(gar_true) ), 'GAR_num'] = 1    

    df.loc[(df['Garten/_mitnutzung'].isin(gar_false) ), 'GAR_num'] = 0 

    

    #Haustiere_erlaubt

    df['HTI_num'] = 1  

    df.loc[(df['Haustiere_erlaubt'].isin(hti_true) ), 'HTI_num'] = 1    

    df.loc[(df['Haustiere_erlaubt'].isin(hti_false) ), 'HTI_num'] = 0 

    

    #Keller

    df['KLR_num'] = 1  

    df.loc[(df['Keller'].isin(klr_true) ), 'KLR_num'] = 1    

    df.loc[(df['Keller'].isin(klr_false) ), 'KLR_num'] = 0 

    

    #Moebliert/Teilmoebliert

    df['MOE_num'] = 1  

    df.loc[(df['Moebliert/Teilmoebliert'].isin(moe_true) ), 'MOE_num'] = 1    

    df.loc[(df['Moebliert/Teilmoebliert'].isin(moe_false) ), 'MOE_num'] = 0 

        

    #Terrasse

    df['TER_num'] = 1  

    df.loc[(df['Terrasse'].isin(ter_true) ), 'TER_num'] = 1    

    df.loc[(df['Terrasse'].isin(ter_false) ), 'TER_num'] = 0 

        

    #WG_geeignet

    df['WGG_num'] = 1  

    df.loc[(df['WG_geeignet'].isin(wgg_true) ), 'WGG_num'] = 1    

    df.loc[(df['WG_geeignet'].isin(wgg_false) ), 'WGG_num'] = 0 

    

    #Haustyp

    df['HTP_num'] = 1       

    df.loc[(df['Haustyp'].isin(htp_kat1) ), 'HTP_num'] = 1    

    df.loc[(df['Haustyp'].isin(htp_kat2) ), 'HTP_num'] = 2 

    df.loc[(df['Haustyp'].isin(htp_kat3) ), 'HTP_num'] = 3 

    df.loc[(df['Haustyp'].isin(htp_kat4) ), 'HTP_num'] = 4 

    df.loc[(df['Haustyp'].isin(htp_kat5) ), 'HTP_num'] = 5

    

    #Heizungsart

    df['HAR_num'] = 1       

    df.loc[(df['Heizungsart'].isin(har_kat1) ), 'HAR_num'] = 1 

    df.loc[(df['Heizungsart'].isin(har_kat2) ), 'HAR_num'] = 2 

    df.loc[(df['Heizungsart'].isin(har_kat3) ), 'HAR_num'] = 3 

    df.loc[(df['Heizungsart'].isin(har_kat4) ), 'HAR_num'] = 4 

    df.loc[(df['Heizungsart'].isin(har_kat5) ), 'HAR_num'] = 5 

    
#DatenPreparation

#An dieser Stelle wird die Korrelation der neuen numerischen Werte mit dem Zielmerkmal analysiert. Erst danach werden 

#überflüssige Spalten herausgestrichen, auch die alten kategorischen Merkmale



new_col_num = ['BDW_num', 'BALK_num', 'BAR_num', 'DBO_num', 'DENK_num', 'DU_num', 'EBK_num', 'ELW_num', 'GWC_num', 

               'STL_num', 'GAR_num', 'HTI_num', 'KLR_num', 'MOE_num', 'TER_num', 'WGG_num', 'HTP_num', 'HAR_num', 'Zimmer']



nr_rows = 7

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(new_col_num):

            sns.regplot(df[new_col_num[i]], df[target], ax = axs[r][c])

            stp = stats.pearsonr(df[new_col_num[i]], df[target])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()   
#DataPreparation

#Schlussfolgerung:

#es ist ersichtlich, dass keins der Merkmale eine starke Korrelation zum Zielmerkmal hat 

#daher werden Spalten mit extrem geringer Korrelation gelöscht. Auch hier werden wieder alle Spalten

#mit einer Korrelation kleiner 0.01 gelöscht
#DataPreparation

#Hier werden alle alten kategorischen Merkmale gelöscht und alle noch existierenden Mermale in eine Liste mit ihren 

#dazugehörigen Korrelationen aufgelistet. 

#Dann werden noch alle neuen numerischen Merkmale gelöscht, die eine korrelation kleiner 0.01 haben.

#Danach werden alle noch vorhanden Merkmale in einer Liste ausgegeben

#Abschließend wird noch eine List der Merkmale mit dessen Datentypen zur überprüfung ausgegeben.



catg_cols_to_drop = ['Badewanne', 'Balkon', 'Barrierefrei','Dachboden', 'Denkmalobjekt', 'Dusche', 'Einbaukueche',

                    'Einliegerwohnung', 'Gaeste_WC', 'Garage/Stellplatz', 'Garten/_mitnutzung', 'Haustiere_erlaubt',

                    'Haustyp', 'Heizungsart', 'Keller', 'Moebliert/Teilmoebliert', 'Terrasse', 'WG_geeignet']



corr1 = df.corr()

corr_abs_1 = corr1.abs()



nr_all_cols = len(df)

ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, target)[target]



print(ser_corr_1)

print("*"*80)

cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)





for df in [df] :

    df.drop(catg_cols_to_drop, inplace= True, axis = 1)

    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1)  



print (df.columns)

print("*"*80)

print (df.dtypes)
#DataPreparation

#Vorheriger Schritt hat gezeigt, dass einige float Datentypen in einen int umgewandelt werden sollten. Dies passiert an 

#dieser Stelle. Merkmale Baujahr und Preis werden umgewandelt.



df.Baujahr = df.Baujahr.astype(int)

df.ExactPreis = df.ExactPreis.astype(int)

#DataPreparation

#Nach vollständiger Bereinigung des Datensatztes werden hier nocheinmal alle übrigen Merkmale und dessen Korrelationen zum

#Zielmerkmal in einer Liste dargestellt.



corr2 = df.corr()

corr_abs_2 = corr2.abs()



nr_all_cols = len(df)

ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]



print(ser_corr_2)
#DataPreparation

#Abschließend werden noch alle Merkmale und dessen Korrelationen mit

#dem Zielmerkmal in einer Matrix dargestellt



nr_feats=len(df.columns)

plot_corr_matrix(df, nr_feats, target)
#DataPreparation

#Endstand der Anzahl der Zeilen und Spalten

print(df.shape)

print("*"*80)

print(df.columns)
#DataPreparation

#Zeigt die ersten Zeilen des "neuen" Datensatzes

df.head()
#Modellierung

#Trennen des Datensatzes

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)
#Modellierung

#Vorbereitung des Daten -- StandardScaler

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

df_train_sc = sc.fit_transform(df_train)

df_test_sc = sc.transform(df_test)



df_train_sc = pd.DataFrame(df_train_sc)

df_train_sc.head()
#Modellierung

#Vorbeiteung der Daten 

X = df_train.copy()

y = df_train[target]



X_test = df_test.copy()

y_test = df_test[target]



X_sc = df_train_sc.copy()

y_sc = df_train[target]

X_test_sc = df_test_sc.copy()



X.info()

X_test.info()
#Modellierung - Lernalgorithmen

#Vorbereitung

from sklearn.model_selection import GridSearchCV

score_calc = 'neg_mean_squared_error'
#Modellierung - Lernalgorithmen

#Lineare Regression

from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)

grid_linear.fit(X, y)



sc_linear = get_best_score(grid_linear)



linregr_all = LinearRegression()

linregr_all.fit(X, y)

pred_linreg_all = linregr_all.predict(X_test)

pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()
#Modellierung - Lernalgorithmen

#Lasso -- Least absolute shrinkage and selection operator

#für alpha=0,001; 0,01; 0,1; 0,5; 0,8 und 1



from sklearn.linear_model import Lasso



lasso = Lasso()

parameters = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}

grid_lasso = GridSearchCV(lasso, parameters, cv=nr_cv, verbose=1, scoring = score_calc)

grid_lasso.fit(X, y)



sc_lasso = get_best_score(grid_lasso)



pred_lasso = grid_lasso.predict(X_test)
#Modellierung - Lernalgorithmen

#Elastic Net - elastisches Netz

#für alpha=0,1; 1 und 10





from sklearn.linear_model import ElasticNet



enet = ElasticNet()

parameters = {'alpha' :[0.1,1.0,10], 'max_iter' :[1000000], 'l1_ratio':[0.04,0.05], 

              'fit_intercept' : [False,True], 'normalize':[True,False], 'tol':[1e-02,1e-03,1e-04]}

grid_enet = GridSearchCV(enet, parameters, cv=nr_cv, verbose=1, scoring = score_calc)

grid_enet.fit(X_sc, y_sc)



sc_enet = get_best_score(grid_enet)



pred_enet = grid_enet.predict(X_test_sc)
#Modellierung - Lernalgorithmen

#Stochastic Gradient Descent (stochastischer Gradientenabstieg)



from sklearn.linear_model import SGDRegressor



sgd = SGDRegressor()

parameters = {'max_iter' :[10000], 'alpha':[1e-05], 'epsilon':[1e-02], 'fit_intercept' : [True]  }

grid_sgd = GridSearchCV(sgd, parameters, cv=nr_cv, verbose=1, scoring = score_calc)

grid_sgd.fit(X_sc, y_sc)



sc_sgd = get_best_score(grid_sgd)



pred_sgd = grid_sgd.predict(X_test_sc)
#Modellierung -- Lernalgorithmen

#KNeighborsRegressor



from sklearn.neighbors import KNeighborsRegressor



param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,    

              'weights' : ['uniform','distance'] ,

              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}



grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_knn.fit(X_sc, y_sc)



sc_knn = get_best_score(grid_knn)



pred_knn = grid_knn.predict(X_test_sc)



sub_knn = pd.DataFrame()

sub_knn['ExactPreis'] = pred_knn
#Modellierung - Lernalgorithmus

#GBR - Gradient Boosting Regression



from sklearn.ensemble import GradientBoostingRegressor



models = [GradientBoostingRegressor(n_estimators=200,max_depth=12)]

learning_mods = pd.DataFrame()

temp = {}



for model in models:

    print(models)

    m = str(models)

    temp['Model'] = m[:m.index('(')]

    model.fit(X, y)

    print('score on training',model.score(X, y))

    learning_mods = learning_mods.append([temp])

learning_mods.set_index('Model', inplace=True)



sc_boost = model.score(X,y)

pred_boost = model.predict(X_test)
#Evaluation

#Abbildung aller Verfahren

list_scores_alle = [sc_linear, sc_sgd, sc_lasso, sc_enet, sc_knn, sc_boost]

list_regressors_alle= ['Linear','SGD','Lasso','ElaNet', 'KNN', 'GBoost']



fig, ax = plt.subplots()

fig.set_size_inches(10,7)

sns.barplot(x=list_regressors_alle, y=list_scores_alle, ax=ax)

plt.ylabel('Wurzel der Summe der Abweichungsquadrate')

plt.show()
#Evaluation

#Abbildung von Linearer Regression, Lasso, Elastisches Netz, k-nächste Nachbarn und Gradient Boosting

#Da für das SGD-Verfahren ein so hoher Wert erhalten wurde, konnte nur dafür ein Balken erkannt werden, also werden die anderen noch einmal ausgegeben

list_scores_llr = [sc_linear, sc_lasso, sc_enet, sc_knn, sc_boost]

list_regressors_llr = ['Linear','Lasso', 'Ela Netz', 'KNN', 'GBoost']



fig, ax = plt.subplots()

fig.set_size_inches(8,7)

sns.barplot(x=list_regressors_llr, y=list_scores_llr, ax=ax)

plt.ylabel('Wurzel der Summe der Abweichungsquadrate')

plt.show()
#Evaluation

#Abbildung von lineare Regression, Lasso und Gradient Boosting

#Da für die Verfahren elastisches Netz und k-nächste Nachbarn ein so hoher Wert erhalten wurde, konnte nur dafür ein Balken erkannt werden, 

#also werden die anderen noch einmal ausgegeben

list_scores_llr = [sc_linear, sc_lasso, sc_boost]

list_regressors_llr = ['Linear','Lasso', 'GBoost']



fig, ax = plt.subplots()

fig.set_size_inches(6,7)

sns.barplot(x=list_regressors_llr, y=list_scores_llr, ax=ax)

plt.ylabel('Wurzel der Summe der Abweichungsquadrate')

plt.show()
#Evaluation

#Abbildung von linearer Regression und Lasso

#Da für das GBoost-Verfahren ein so hoher Wert erhalten wurde, konnte nur dafür ein Balken erkannt werden, also werden die anderen noch einmal ausgegeben

#Obwohl das lineare Regression-Verfahren immer noch nicht sichtbar ist, wird es trotzdem nicht noch einmal ausgegeben,

#da es nicht sinnvoll ist, nur ein Verfahren in einem Diagramm darzustellen, da die Vergleichbbarkeit fehlt.

list_scores_llr = [sc_linear, sc_lasso]

list_regressors_llr = ['Linear','Lasso']



fig, ax = plt.subplots()

fig.set_size_inches(4,7)

sns.barplot(x=list_regressors_llr, y=list_scores_llr, ax=ax)

plt.ylabel('Wurzel der Summe der Abweichungsquadrate')

plt.show()
#Evaluation 

#Korrelationen der Ergebnisse der Lernalgorithmen 



predictions = {'Linear': pred_linreg_all,  'Lasso': pred_lasso,

               'SGD': pred_sgd,'ElaNet': pred_enet,  'KNN': pred_knn, 'GBoost': pred_boost}

df_predictions = pd.DataFrame(data=predictions) 

df_predictions.corr()
#Evaluation 

#Korrelationen der Ergebnisse der Lernalgorithmen als Matrix



plt.figure(figsize=(7, 7))

sns.set(font_scale=1.25)

sns.heatmap(df_predictions.corr(), linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=df_predictions.columns , xticklabels=df_predictions.columns

            )

plt.show()
#Evaluation

#Vergleich Vorhergesagte Preise und Orignal Preise



predictions_01 = pred_boost

predictions_02 = pred_enet

predictions_03 = pred_sgd

predictions_04 = pred_linreg_all

predictions_05 = pred_knn

predictions_06 = pred_lasso



submission_01 = pd.DataFrame({

        "Org Hauspreis": y_test,

        "Prog Linear": predictions_04,

        "Prog SGD": predictions_03,

        "Prog Lasso": predictions_06,

        "Prog ElaNet": predictions_02,

        "Prog kNN": predictions_05,

        "Prog GBoost": predictions_01,        

    })



submission_01.to_csv('mean01.csv',index=False)

submission_01.head()
predictions_durchschnitt = np.round( (pred_lasso + pred_enet + pred_linreg_all + pred_sgd + pred_knn + pred_boost) / 6.0 )

predictions_beste3 = np.round((pred_lasso + pred_linreg_all + pred_boost) / 3.0)

predictions_beste2 = np.round((pred_lasso + pred_linreg_all) / 2.0)

submission_02 = pd.DataFrame({

        "Org Hauspreis": y_test,

        "Durchschnittsprognose": predictions_durchschnitt,

        "Prognose beste 3": predictions_beste3,

        "Prognose beste 2": predictions_beste2

})





submission_02.to_csv('mean02.csv',index=False)

submission_02.head()