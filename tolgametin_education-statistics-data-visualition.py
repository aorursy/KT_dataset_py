
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

import datetime as dt


import os
print(os.listdir("../input/education-statistics/edstats-csv-zip-32-mb-"))

import os
print(os.listdir("../input/education-statistics/edstats-excel-zip-72-mb-"))
        
import os
print(os.listdir("../input"))


data = pd.read_csv("../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsData.csv")
dataSeries = pd.read_csv("../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsSeries.csv")
dataCountry= pd.read_csv("../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry-Series.csv")
dataFoot = pd.read_csv("../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsFootNote.csv")
dataStats = pd.read_csv("../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry.csv")
data.info()
data.tail(5)
data.head(5)


data.columns
data.corr()
#korelasyon haritası oluşturma
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data.corr(), annot=False, linewidths=.3, fmt= '.1f',ax=ax)
plt.show()
dataFoot.describe()
data["Country Name"].unique()        #tüm ülke verilerinin tek şekilde gösterilmesi.
dataTurkey = data[data["Country Name"] == "Turkey"]     #türkiye verilerinin 1970 ten itibaren ilk 5 satırının gösterilmesi.
dataTurkey.head(5)
yearDic = {}

startYear =0
now = dt.datetime.now()

cols = dataTurkey.columns
for columnName in cols:
    try:
        year = int(columnName)
        if startYear == 0:
            startYear = year
        if year - startYear == 10 or now.year-1 == year:
            yearDic[str(startYear)]=str(year)
            startYear = year
    except:
        print("")    

for key,value in yearDic.items():
    print(key," : ",value)
i=0
for key,value in yearDic.items():
    dataTurkey.plot(kind='line', x=key, y=value,alpha = 1,color = 'C'+str(i),label=key+" - "+value,grid=True)
    i=i+1
plt.show()
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
data.shape
data.loc[:,'Country Name':'Indicator Code'].describe()
countries = data.loc[:,['Country Name','Country Code']].drop_duplicates()
indicators = data.loc[:,['Indicator Name','Indicator Code']].drop_duplicates()
print (countries.shape, indicators.shape)
countries.head()
indicators.head()
present = data.loc[:,'1977':'2016'].notnull().sum()/len(data)*100
future = data.loc[:,'2020':].notnull().sum()/len(data)*100
plt.figure(figsize=(10,7))
plt.subplot(121)
present.plot(kind='barh', color='green')
plt.title('Kaybolan Veriler (% of Veri sutunlari)')
plt.ylabel('Column')
plt.subplot(122)
future.plot(kind='barh', color='limegreen')
plt.title('Kaybolan Veriler (% of Veri sutunlari)')
plt.show()
countries[countries['Country Name'].str.contains('Turkey')]
turkeyVeri = data.loc[data['Country Code']=='TUR']
turkeyVeri.head()
turkeyVeri = turkeyVeri.dropna('index', thresh = 5) # İlk dort sutunda en az 1 deger bulunması gerekir.
turkeyVeri.shape
turkeyVeri.head()
indicators.to_csv('indicators.csv', index=False)
nRowsRead = 1000
dataStats = pd.read_csv('../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = dataStats.shape
print(f'Satır sayisi {nRow} sutun ve {nCol} sutun')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
def plotHistogram(data, nHistogramShown, nHistogramPerRow):
    nunique = data.nunique()
    data = data[[col for col in data if nunique[col] > 1 and nunique[col] < 50]] #Görüntüleme amacıyla 1ve 50 arasında benzersiz değer içeren sütunlar
    nRow, nCol = data.shape
    columnNames = list(data)
    nHistRow = (nCol + nHistogramPerRow - 1) / nHistogramPerRow
    plt.figure(num=None, figsize=(6*nHistogramPerRow, 8*nHistRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nHistogramShown)):
        plt.subplot(nHistRow, nHistogramPerRow, i+1)
        data.iloc[:,i].hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()
def plotCorrelationMatrix(data, graphWidth):
    filename = data.dataframeName
    data = data.dropna('columns') # nan degerli sutunlar cikarildi.
    data = data[[col for col in data if data[col].nunique() > 1]] 
    if data.shape[1] < 2:
        print(f'Korelasyon grafiginde Nan degerler icin sütun ve satırlar goruntulenmedi ({data.shape[1]}) is less than 2')
        return
    corr = data.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Korelasyon matrisi for {filename}', fontsize=15)
    plt.show()
def plotScatterMatrix(data, plotSize, textSize):
    data = data.select_dtypes(include =[np.number]) # Secilen verilerin data tipi sadece numerik olması icin gerekli kod.
    data = data.dropna('columns')
    data = data[[col for col in data if data[col].nunique() > 1]] # Ciktinin tek (unique) olması için .
    columnNames = list(data)
    if len(columnNames) > 10: # Matris transpozu icin sutun sayısı azaltıldı.
        columnNames = columnNames[:10]
    data = data[columnNames]
    ax = pd.plotting.scatter_matrix(data, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = data.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Sacılma ve Yayılma grafigi')
    plt.show()
plotHistogram(dataStats, 10, 5)
nRowsRead = 1000 
data2 = pd.read_csv('../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry.csv', delimiter=',', nrows = nRowsRead)
data2.dataframeName = 'EdStatsCountry-Series.csv'
nRow, nCol = data2.shape
print(f'Gosterilen satırlar {nRow} satır ve gosterilen sutunlar {nCol} sutun')
data2.head(5)
plotHistogram(data2, 10, 5)