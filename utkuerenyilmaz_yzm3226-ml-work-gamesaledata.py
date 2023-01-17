import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score



#Datamızın ilk 5 satırını inceliyoruz.
df = pd.read_csv("../input/vgsales.csv")
df.head() #İlk 5 satırı inceliyoruz
#Son 5 satırını inceliyoruz
df.tail()
#İstatiksel değerlerine bakıyoruz
df.describe()
#Sütunların veritipleri inceleniyor
df.info()
#Dataset içerisindeki toplam satır ve sütun sayıları gösteriliyor
df.shape
#Sütun isimleri Türkçeleştiriliyor.
df = df.rename(columns={'Rank': 'BasariSiralamasi', 
                        'Name': 'OyunAdi', 
                        'Year' : 'CikisTarihi', 
                        'Genre':'OyunTarzi',
                        'Publisher': 'Yayimci',
                        'NA_Sales': 'KuzeyAmerikaSatis',
                       'EU_Sales' : 'AvrupaSatis',
                       'JP_Sales' : 'JaponyaSatis',
                       'Other_Sales' : 'DigerSatis',
                        'Global_Sales' : 'GlobalSatis'})
#Satış değerleri içerisindeki veriler ondalıklı halde bulunduklarından bu değerleri normalleştiriyoruz

df['KuzeyAmerikaSatis'] = df['KuzeyAmerikaSatis'] * 100
df['AvrupaSatis'] = df['AvrupaSatis'] * 100
df['JaponyaSatis'] = df['JaponyaSatis'] * 100
df['DigerSatis'] = df['DigerSatis'] * 100
df['GlobalSatis'] = df['GlobalSatis'] * 100

df.head()
#Gereksiz sütunların çıkarılması işlemi gerçekleştiriliyor ve 3 rastgele örnek görüntüleniyor.
df = df[['BasariSiralamasi', 'OyunAdi','Platform','CikisTarihi','OyunTarzi','Yayimci','GlobalSatis']]
df.sample(3)
#Platformu PC olan veriler seçilerek tablo güncelleniyor
df = df[(df['Platform']=='PC')]
#Oluşan yeni satır-sütun sayılarının görülmesi
df.shape
#Tüm boş değerleri bulup toplamlarının yazdırılması
df.isnull().sum()
#Büyük ölçekte dağılımların görülmesi için ayrı ayrı yazılıyor.
df.hist('BasariSiralamasi')
df.hist('CikisTarihi')
df.hist('GlobalSatis')
#CikisTarihi için filtreleme yapılıyor.
df = df[(df['CikisTarihi']>2000) & (df['CikisTarihi']<2019)] 

df.hist('CikisTarihi')
#Uç değerler inceleniyor
sns.boxplot(x=df['CikisTarihi'])
df.hist('GlobalSatis')
sns.boxplot(x=df['GlobalSatis'])
#İlk filtreleme
df = df[(df['GlobalSatis']>10) & (df['GlobalSatis']<250)] 
sns.boxplot(x=df['GlobalSatis'])
df.hist('GlobalSatis')
df.shape
#İkinci filtreleme
df = df[(df['GlobalSatis']<100)] 
sns.boxplot(x=df['GlobalSatis'])
df.hist('GlobalSatis')
df.shape
df.isnull().sum()
#Tabloda yalnızca kullandığımız sütunlar bırakılıyor.
df = df[['BasariSiralamasi', 'OyunAdi','Platform','CikisTarihi','GlobalSatis']]
#Sıralama işlemi gerçekleştiriliyor.
df = df.sort_values('CikisTarihi', axis=0, ascending=True)
df.head(10)
#İlişkilerin sayısal incelemesi
df.corr()
#Isı haritası
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Bulunan ilişkinin çizim işlemi gerçekleştiriliyor.
df.plot(x='GlobalSatis', y='BasariSiralamasi', style='.')  
plt.title('Global Satış Derecesi - Başarı Sıralaması')  
plt.xlabel('Global Satışlar')  
plt.ylabel('Başarı Sıralamaları')  
plt.show() 
x = df.iloc[:,4].values.reshape(-1,1) #GlobalSatis sütunundaki değerler x 
y = df.iloc[:,0].values.reshape(-1,1) #BaşarıSiralamasi sütunundaki değerler y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Lineer Regresyon Modeli Eğitimi
lr.fit(x_train, y_train.ravel())

#Modelin yaptığı tahminler y_pred olarak tanımlanıyor.
y_pred = lr.predict(x_test)
print("Kesim Noktası: ",lr.intercept_)
print("Eğim: ",lr.coef_)
#Test verilerinin ve Tahminlerin tablosal olarak incelenmesi
LRegressionTablo = pd.DataFrame({'Gerçek':y_test.ravel(),'Tahmin':y_pred.ravel()})
LRegressionTablo
#Tahminler ve gerçek değerler çizdiriliyor.
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred ,color="blue")
plt.title('Lineer Regresyon Doğrusu')
plt.xlabel('Global Satislar')
plt.ylabel('Basari Siralamasi')
plt.show()
#Regresyon için başarı ölçütleri
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Başarı ölçütü
print('Lineer Regresyon Başarısı: ',r2_score(y_test, y_pred))
#CM için..
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed) #KFold model tercih ediliyor.
cv_results = model_selection.cross_val_score(nb, X_train, Y_train, cv=kfold, scoring=scoring)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=0)
# Naive Bayes - Gaussian Naive Bayes Modeli Eğitimi
nb.fit(x_train2, y_train2)
#Modelin yaptığı tahminler y_pred olarak tanımlanıyor.
y_pred2 = nb.predict(x_test2)
NBayesTablo = pd.DataFrame({'Gerçek':y_test2.ravel(),'Tahmin':y_pred2.ravel()})
NBayesTablo
#NB için tahminler ve gerçek değerler çizdiriliyor.
plt.scatter(x_test2, y_test2, color='red')
plt.plot(x_test2, y_pred2 ,'b+')
plt.title('Naive Bayes Algoritması Tahminleri')
plt.xlabel('Global Satislar')
plt.ylabel('Basari Siralamasi')
plt.show()
# NB öltütleri inceleniyor
print(classification_report(y_test2, y_pred2))
print(confusion_matrix(y_test2, y_pred2))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred2,y_test2))
print('Naive Bayes Başarısı: ',r2_score(y_test2, y_pred2))
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 5)

dtr.fit(x_train3, y_train3)
y_pred3 = dtr.predict(x_test3)

plt.scatter(x_test3, y_test3, color='red')
plt.plot(x_test3, y_pred3 ,'b+')
plt.title('Decision Tree Algoritması Tahminleri')
plt.xlabel('Global Satislar')
plt.ylabel('Basari Siralamasi')
plt.show()

print('Başarı Puanı: ',r2_score(y_test3, y_pred3))
x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2)

x_poly = pr.fit_transform(x_train4)

# Lineer Regresyon Modeli Eğitimi
pr.fit(x_poly, y_train4)

lr2 = LinearRegression()

# Lineer Regresyon Modeli Eğitimi
lr2.fit(x_poly, y_train4.ravel())

#Modelin yaptığı tahminler y_pred olarak tanımlanıyor.
y_pred4 = lr2.predict(pr.fit_transform(x_test4))


#Test verilerinin ve Tahminlerin tablosal olarak incelenmesi
PRegressionTablo = pd.DataFrame({'Gerçek':y_test4.ravel(),'Tahmin':y_pred4.ravel()})
PRegressionTablo.head()
plt.scatter(x_test4, y_test4, color='red')
plt.scatter(x_test4, y_pred4.ravel() ,color="blue")
plt.title('Polinomsal Regresyon Doğrusu')
plt.xlabel('Global Satislar')
plt.ylabel('Basari Siralamasi')
plt.show()
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test4,y_pred4))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test4,y_pred4))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test4,y_pred4)))

print('Polinomsal Regresyon Başarısı: ',r2_score(y_test4, y_pred4))