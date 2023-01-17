# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#REFERENCES DanB
#https://www.kaggle.com/dansbecker
melbourne_file_path = '../input/melb_data.csv' #path yolunu dağişken içine aldık kullanmak için
melbourne_Data = pd.read_csv(melbourne_file_path) #pandas'ın read metodunda direk path yolunu vermek ye-
#rine o pathi içine aldığımız değişkeni parametre olarak verdik.
#print(melbourne_Data.describe()) #Oluşturduğumuz frame'in belli değerlerini gördük
#print(melbourne_Data.columns) #Oluşturduğumuz frame'in sütunlarını görüntüledik
melbourne_data_price = melbourne_Data.Price #DataFrame'mimizde sadece 'Price' sütununu aldık.
print(melbourne_data_price.tail()) #Aldığımız 'Price' sütununa ait değerlerin sondan 5 tanesi getirildi.
print(melbourne_data_price.head())#Aldığımız 'Price' sütununa ait değerlerin baştan 5 tanesi getirildi.
columns_of_insterest = ['Landsize','BuildingArea'] #Oluşturduğumuz string listeye parametre olarak
#verimizin 2 sütunu olan 'Landsize','BuildingArea' sütunlarını verdik. (SERİ)
two_columns_of_data = melbourne_Data[columns_of_insterest]#DataFrame'mimize parametre olarak bu seriyi
#verdik ve bu yapıyı bir değişkene atadık.
two_columns_of_data.describe() # Atamış olduğumuz değişkenin belirli değerlerini göstermek için describe()
#fonksiyonunu kullandık.

#columns_of_insterest=LİST
#melbourne_Data=DataFrame
#two_columns_of_data=variable





##Choosing Predictors Target (Tahmin Hedefi Seçme)
melbourne_prediktor = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude'] #melbourne_perdiktor
#adında bir liste oluşturduk ve sütun isimlerimizi string parametre olarak bu seriye verdik daha sonra,
X = melbourne_Data[melbourne_prediktor] # bu seriyi parametre olarak DataFrame'mimize verdik ve onu da
# X değişkenine eşitledik.
print(X.head()) #X değişkeni içerisindeki değerleri baştan 5 tane olacak şekilde ekrana yazdırdık.
y = melbourne_Data.Price #Tahmin Hedefi olarak seçtiğimiz sütun
melbourne_model = DecisionTreeRegressor() #Tahmin yapmak istediğimiz üstünde çalışacağımız diğer 
#verilerimiz.
melbourne_model.fit(X,y)
print("Making predictions for the following 5 houses") #Listenin başından aldığımız 5 ev
print(X.head())
print("The predictions are") #Bu 5 ev için tahmini fiyatlarımız
print(melbourne_model.predict(X.head()))
#İlk olarak tahminlemek istediğimiz 'hedef değişkeni(Price)' belirledik. Bu değişkeni y'ye atadık
#Price'ı tahmin etmek için SERİ TİPİNDE tahmin edicileri tanımladık. Yani bunlar üzerinden tahminde bulunduk.
#Bunlar (Rooms,Bathroom,Landsize,Lattitude,Longtitude)
#Bu listeyi DataFrame'mimizde çağırdık ve çıkan gelen değerleri X değişken adıyla kaydettik.
#Modelimizi oluşturduk. Modelimize Parametre olarak X ve y değerlerini verdik.
#Son bölümde ise 'predict' fonksiyonu ile modelimizdeki tahmini gerçekleştirdik parametre olarak ise
#X'in ilk 5 değerini aldı.

melbourne_Data.corr()
#Corelasyon matrisi birbiriyle alakalı olan parametreleri görmemizde bize yardımcı olur. Yani kısaca an-
#latmak gerekirse 1 parametredeki artış diğer parametrede de artışa neden olur azalış aynı şekilde o para-
#metrede de azalmaya neden olur. Buna pozitif(+) korelasyon denir. Negatif korelasyon ise bir parametre-
#deki artış diğer parametrenin değerini azaltıyorsa yani aralarında ters orantı varsa buna negatif (-) 
#korelasyon denir. Korelasyon değerinin çok çok küçük olması (0.037) bu iki değer arasında herhangi bir
#ilişki olmadığı anlamına gelir. Kısaca anlatmak gerekirse bir parametredeki artış yada azalış diğer 
#parametrenin değerini herhangi bir şekilde etkilememektedir. 
melbourne_Data.info()
#melbourne_Data.info()
#print(melbourne_Data.Rooms.head(10))
#print(melbourne_Data.Bedroom2.head(10))
#print(melbourne_Data.Bathroom.head(10))
#print(melbourne_Data.Price.head(10))
four_columns = ['Rooms','Bedroom2','Bathroom','Price','Distance']
a=melbourne_Data[four_columns]
print(a.head(20))

#-------------------------------------------------#
#Evdeki oda sayısının; yatak odası(Bedroom2),uzaklık(Distance),banyo(Bathroom) ve evin fiyatına(Price) göre değerleri aşağı-
#daki gibidir. Şimdi biz ise verdiğimiz parametreler ile bir evin kaç odalı olduğunu tahmin etmeye 
#çalışacağız.

b=melbourne_Data.Rooms #Hedef değişkenimizi Frameden çekiyor ve 'b' değişkenine aktarıyoruz.
melbourne_prediktor2 = ['Price','Bathroom','Bedroom2','Distance'] #Tahminimizi hangi parametrelere göre
#yapmak istiyorsak o parametreleri bir "LİSTE" içine alıyoruz.
A = melbourne_Data[melbourne_prediktor2] #Listemizi DataFramimizin içine alıyoruz ve sonucu A değişkenine
#eşitliyoruz.
melbourne_model2 = DecisionTreeRegressor() #DecisionTreeRegressor() fonksiyonu ile bir model oluşturup,
#o modele parametre olarak hesapladığımız A ve b değerlerini gönderiyoruz.
melbourne_model2.fit(A,b)


print("İlk 5 evin kaç odalı olduğunu ekrana yazdıralım:")
print(melbourne_Data.Rooms.head())
print("İlk 5 evin tahmini kaç odalı olduğu:")
print(melbourne_model2.predict(A.head()))
#Programımızın Bütün Veriler Üzerindeki Tahminleri
print("Mevcut evlerin kaç odalı oldukları:")
print(melbourne_Data.Rooms)
print("Mevcut evlerin tahmini kaç odalı oldukları:")
print(melbourne_model2.predict(A))
#Bir model oluşturduk peki bu modelimiz ne kadar iyi çalışıyor?
#Yaptığımız hemen her model için bu soruyu cevaplamamız gerekecektir.Çoğu uygulamalarda, model kalitesinin
#ilgili ölçütü tahmindeki doğruluktur. Başka bir deyişle modelin tahminin gerçekliğe olan yakınlığıdır.
#Bazı insanlar bu problemi, eğitim verilerinin tahminlerini yaparak yanıtlamaya çalışırlar. Bu tahminleri,
#eğitim verilerindeki gerçek değerlerle karşılaştırırlar. Bu yaklaşım bir eksikliğe sahiptir. Bu basit yak-
#laşım bile model kalitesini birinin anlayabileceği bir biçimde özetlememiz gerekir. 10000 konut için her 
#birini teker teker denemek anlamsız olur. Model kalitesini özetlemek için bir çok metrik bulunmaktadır.
#Bunlardan MAE olarak bilineniyle başlayacağız.(Mean Absolute Error) Bu tekniğin son kelimesi Error ile,
#başlayalım.
#Her ev için hata = Gerçek-Tahmin
#Yani evin gerçek fiyatı 150k$ ve senin tahminin 100k$ ise hatan 50k$'dır.
#MAE tekniği ile her hatanın mutlak değerini alırız bu da her hatayı pozitif bir sayıya dönüştürür.Daha son-
#ra mutlak hataların ortalamasını alırız. Bu bizim model kalitesinin ölçüsüdür.
#Melbourne verilerindeki ortalama mutlak hatanın hesaplanması:
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
#Az önce hesapladığımız ölçüm, "örnek" puan olarak adlandırılabilir. Hem modeli oluşturmak hem de 
#MAE puanını hesaplamak için tek bir dizi ev (veri örneği olarak adlandırılır) kullandık. Bu kötü.
#Büyük emlak piyasasında kapı renginin ev fiyatına bağlı olmadığını düşünün. Bununla birlikte, 
#modeli oluşturmak için kullandığınız veri örneğinde, yeşil kapıları olan tüm evlerin çok pahalı olması olabilir. 
#Modelin görevi, ev fiyatlarını öngören kalıplar bulmaktır, bu yüzden bu deseni görecek ve her zaman yeşil kapıları olan evler için yüksek fiyatları öngörecektir.
#Bu model başlangıçta eğitim verilerinden elde edildiğinden, model eğitim verilerinde doğru olarak görünecektir.

#Ancak model, yeni verileri gördüğü zaman bu model muhtemelen geçerli olmayacaktır ve model,
# gayrimenkul işimize uyguladığımız zaman çok yanlış olur (ve bize çok fazla paraya mal olur).

#Verilerdeki sadece olay ilişkilerini yakalayan bir model bile olsa, yeni veriler olduğunda 
#tekrarlanmayacak ilişkiler, örnek doğruluk ölçümlerinde çok doğru görünebilir.

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X,train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))

