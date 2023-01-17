import numpy as np

import pandas as pd

# burada numpy da dahil ediyoruz cunku birbiriyle baglantili kullandigimiz yerler olacak





labels_list = ["mustafa" , "kemal" , "murat" , "tugce" , "tugba"]



data_list = [10,20,30,40,50]



# PANDAS SERILERINI BIR TANE INDEKS VE DEGERLERDEN OLUSAN ARRAY DIYEBILIIRIZ



print(pd.Series(data = data_list, index=labels_list))

"""

mustafa    10

kemal      20

murat      30 ----> ciktisini verir

tugce      40 

tugba      50

dtype: int64 --> burada datalarimizin hangi veritipinden oldugunu gorebiliyoruz



"""







print(pd.Series(data_list))



"""

0    10

1    20

2    30

3    40

4    50

dtype: int64



BURADA HERHANGIBIR INDEX VERMEDIGIMIZ ICIN 

OTOMATIK DEFAULT OLARAK 0,1,2,3,4. INDEKSLER OLARAK CIKAR



"""



npArray = np.array([10,20,30,40,50])



print(pd.Series(npArray))

#burada pandas icine bir array serisi verebiliriz

#burada print ettigimizde yine default olarak index karsiligini verir

"""

0    10

1    20

2    30

3    40

4    50

dtype: int64

"""



print(pd.Series(npArray,labels_list))

#burada herbir deger karsisine indeksi vermis olduk

"""

mustafa    10

kemal      20

murat      30

tugce      40

tugba      50

dtype: int64

"""



print(pd.Series(data=npArray, index=["A","B","C","D","E"]))

#burada da indexleri kendi istedgimiz gibi yazabiliriz



"""

A    10

B    20

C    30

D    40

E    50

dtype: int64



"""



dataDict = {"kadir":30, "kemal":80,"tugba":60}

#burada bir sozluk olusturup degerlerin karsiliklarini yazdik

print(pd.Series(dataDict))

#bu sozlugu pandas serisi olarak yazdirabiliriz

#asagidaki gibi vikti verir





"""

sozlukler sirali bir veritipi olmadigi icin karisik olarak cikmis oldu:





kadir    30

kemal    80

tugba    60

dtype: int64



"""



ser2017=pd.Series([5,10,14,20],["bugday","misir","kiraz","erik"])



print(ser2017)



ser2018=pd.Series([2,12,12,6],["bugday","misir","cilek","erik"])



print(ser2018)



#burada seriler arasinda " cilek " ve " kiraz "  farkli digerleri ayni



print(ser2018+ser2017)



"""

bugday     7.0

cilek      NaN

erik      26.0

kiraz      NaN

misir     22.0

dtype: float64



ORTAK OLMAYAN SERILERI TOPLAYAMAYACAGI ICIN "NOT A NUMBER" (NAN) HATASI ALMIS OLDUK





"""



total  = ser2018+ser2017



print(total["erik"])#burada degerlerden sadece erik  e karsilik geleni almak istedigimizi soyluyoruz



"""26.0"""  #cikti olarak sadece erik e karsilik gelen deger



print(total["cilek"])

#burada "nan" hatasinin ciktisini alacaktir
import pandas as pd

import numpy as np



from numpy.random import randn



print(randn(3,3))#bruada negatif degerlerle beraber 3e3 bir matris olusturacak



"""

ornek:

[[ 3.05125875e-01  2.30795044e+00  7.93996435e-01]

 [-1.53363802e-01 -1.87714762e-01  7.93797135e-01]

 [-4.39728563e-01  1.91089607e-01 -2.02253836e-03]]

 """



#simdi burada 3e3 matristen birtane frame olusturacagiz





df = pd.DataFrame(data=randn(3,3),index=["A","B","C"],columns=["column1","column2","column3"])



print(df)



"""



   column1   column2   column3

A -1.501969 -1.272858 -0.265564

B  1.536454 -0.168669  0.083941

C -0.618879 -0.205629  0.867064



bu sekilde bir dataframe olusmus olur



aslinda serilerin birlesmis hali gibi dusunulebilir



yani A,B,C satirlari birer seri

column1,column2,column3 sutunlari ayri birer seri olarak kullanilabilir



"""

#ornek olarak column1 i almak istersek



print(df["column1"])



"""

A    1.492723

B    0.407823

C    0.592123

Name: column1, dtype: float64



burda sadece colum1 e denk gelen degerleri almis olduk

"""



print(type(df["column2"]))

#<class 'pandas.core.series.Series'>  burada turunun  seri oldugunu gorebiliriz



#yani sonuc olarak dataframe ler serilerin birlesimi olarak dusunulebilir





#BU SEFER A SATIRINI KULLANMAK ISTERSEK



print(df.loc["A"]) #burada loc location in kisaltilmasi





"""

column1   -0.953905

column2   -2.534302

column3    2.123128

Name: A, dtype: float64



column lar aslinda burda index gibi davrandi



type a bakarsak yine bunun seri oldugunu gorecegiz





"""





print(df[["column1","column2"]])



"""

burda da kucuk bir dataframe olusmus oldu 





    column1   column2

A -1.587557  1.081876

B  1.074711 -0.701565

C -1.114674  0.257387





"""



# DATAFRAME BIR TANE DAHA COLUMN EKLEMEK ISTERSEK



df["column4"] = pd.Series(randn(3),["A","B","C"])



print(df)



"""

burada yeni bir column un eklendigini goruruz



 

    column1   column2   column3   column4

A  0.241777  1.298464 -0.499235  1.585928

B  0.654612  2.820911 -1.472475 -0.416358

C -0.289617 -1.108525  0.453456 -0.062038



"""





df["column5"] = df["column1"] + df["column2"] + df["column3"]



print(df)



"""

column5 in yani 1-2-3 columnlarinin toplamini bu sekilde seriye ekleyebiliriz



 column1   column2   column3   column4   column5

A -0.983716  0.764547 -0.387030 -0.672593 -0.606199

B  0.431860 -0.382803  1.956575 -0.084792  2.005632

C -0.136474 -0.208077 -0.021431  0.413457 -0.365983



"""



# BIR TANE SUTUNU YA DA COLUMN I SILME



print(df.drop("column5" , axis=1))



"""

column5 i silindigini gormus oluruzu burda



 column1   column2   column3   column4

A  0.573660  0.493511 -0.099867 -0.138561

B -0.959622 -0.502691 -0.498539  1.156420

C -0.030925 -1.556798 -0.525098 -0.413382



"""



print(df)

#burada df yi tekrar yazdirirsak drop islemini yani silme islemini yaptiktan sonra

#drop ettigimiz column5 in tekrar geri geldgini goruruz



#icine bir parametre daha verirsek eger ancak kalici olarak silmis oluruz



print(df.drop("column5",axis=1,inplace=True))

#burada icine "inplace=True" parametresini verirsek bu sefer sil ve guncelle demis oluyoruz



print(df)



"""

burada silindigini gorebiliriz kalici olarakta guncellenmis olur



    column1   column2   column3   column4

A -0.174924 -0.159935 -0.106543  0.696225

B -0.022623  0.834390  0.449900 -1.901480

C  0.426884 -2.239874  0.856686  0.268647





"""



# INDEX E ERISMEK ICIN



print(df.loc["A"])

#sadece "A" satirindakileri almak icin





"""

column1   -0.817550

column2    1.117828

column3   -1.097090

column4   -1.565289

Name: A, dtype: float64



"""



print(df.iloc[0])

#burada yine ayni sonucu verir

"""

yine a satirini aldigimiz gibi ayni sonucu aldik burada





column1   -0.091171

column2    0.332072

column3    0.012015

column4    1.016859

Name: A, dtype: float64



"""





print(df.loc["A","column1"])

#burda da "A" endexine karsilik gelen column sutunundaki degeri almak istersek



"""

-1.8590818436924574



yine:

ornek:



df.loc["B","column2"]  desek bu seferde b nin column2 ye denk gelen degerini verir





"""
import numpy as np

import pandas as pd



from numpy.random import randn





outerIndex =  ["Group1","Group1","Group1","Group2","Group2","Group","Group3","Group3","Group3"]





innerIndex = ["Index1","Index1","Index1","Index2","Index2","Index2","Index3","Index3","Index3"]





zip(outerIndex,innerIndex)

#burada zip fonksiyonunu kullaniyoruz

#ve zip fonksiyonunu daha sonra listeye cevirebiliriz





print(list(zip(outerIndex,innerIndex)))



"""



[('Group1', 'Index1'), ('Group1', 'Index1'), ('Group1', 'Index1'), ('Group2', 'Index2'), ('Group2', 'Index2'), ('Group', 'Index2'), ('Group3', 'Index3'), ('Group3', 'Index3'), ('Group3', 'Index3')]



"""



#dataframe i gruplamak istersek



hierarchy = list(zip(outerIndex,innerIndex))





hierarchy = pd.MultiIndex.from_tuples(hierarchy)



print(hierarchy)



"""

MultiIndex([('Group1', 'Index1'),

            ('Group1', 'Index1'),

            ('Group1', 'Index1'),

            ('Group2', 'Index2'),

            ('Group2', 'Index2'),

            ( 'Group', 'Index2'),

            ('Group3', 'Index3'),

            ('Group3', 'Index3'),

            ('Group3', 'Index3')],

           )

           

"""



df = pd.DataFrame(randn(9,3),hierarchy,columns=["column1","column2","column3"])



print(df)



"""

                column1   column2   column3

Group1 Index1  1.269333 -0.398785 -0.571607

       Index1 -0.400218 -0.538992 -0.906395

       Index1 -0.756547  0.201318  1.018360

Group2 Index2 -1.403767 -0.156331 -0.660266

       Index2  1.923512  0.648003  0.276761

Group  Index2  1.528844 -0.262915 -0.025348

Group3 Index3 -1.796768  2.367349  0.537825

       Index3 -0.595885 -1.379446  0.389105

       Index3  0.586776  1.258694 -1.312381

       

"""





print(df["column1"])



"""

SADECE COLUMN1 E AIT VERILERI ALIR





Group1  Index1    0.888592

        Index1    0.149276

        Index1   -0.062622

Group2  Index2   -0.130522

        Index2   -0.248451

Group   Index2    0.239760

Group3  Index3   -0.529793

        Index3    1.130778

        Index3    0.712903

Name: column1, dtype: float64

"""



print(df.loc["Group1"])



"""

BURADA DA SADECE GROUP1 ALINMIS OLDU

        column1   column2   column3

Index1 -0.675209  0.567387 -0.400552

Index1 -0.714443 -1.631250 -0.860060

Index1 -0.893227  0.274972  0.319853



"""



#YANI DATAFRAME ILE IHTIYACA GORE PARCALAYIP KULLANABILIRIZ VERILERI



print(df.loc[["Group1","Group2"]])



"""

SADECE GROUP1 VE GROUP2 ALINIR



               column1   column2   column3

Group1 Index1 -0.166519 -1.064305  1.522499

       Index1  2.532445  0.612128 -0.257651

       Index1  0.465853 -0.553955  0.142491

Group2 Index2  0.268160 -0.170358 -0.957564

       Index2 -0.221887 -0.610284  1.888540





"""



#GROUP1 IN ICINDEKI INDEX 1 IN DEGERLERINI BULMAK ISTERSEK EGER



print(df.loc["Group1"].loc["Index1"])



"""

yani sadece index1 i almis olduk



       column1   column2   column3

Index1  0.893492  1.563686  0.123854

Index1  0.480956  1.861047  1.081648

Index1  1.185695  0.354632  0.392441



"""



print(df.loc["Group1"].loc["Index1"]["column1"])



df.index.names= ["Groups","Index"]



print(df)





"""

burda grouplara Groups ismini indexlere de Index ismini vermis olduk





                column1   column2   column3

Groups Index                               

Group1 Index1 -0.973220  1.061227 -0.980360

       Index1 -0.297062 -1.447446  0.965287

       Index1 -0.483209 -0.057918  0.715275

Group2 Index2 -1.799557 -1.043056  0.248148

       Index2  1.357227  0.918505 -0.190696

Group  Index2  0.861682 -0.865498  0.177148

Group3 Index3 -1.622727  0.545117 -2.034281

       Index3 -1.084391 -0.130436 -1.968499

       Index3 -0.200170  0.328910 -1.009514





"""



print(df.xs("Group1"))



"""

     column1   column2   column3

Index                               

Index1 -0.401660 -0.004233 -1.454002

Index1  0.400219 -0.339027  1.815558

Index1 -0.028161 -2.415524 -0.344498





"""



print(df.xs("Group2").xs("Index2"))





print(df.xs("Index1",level= "Index"))

#burada sadece gruplarin index1 lerini almak istedigimizi soyluyoruz
import numpy as np

import pandas as pd



arr = np.array([[10,20,np.nan],[5,np.nan,np.nan],[21,np.nan,10]])



#burada ornek olmasi acisindan icerisinde not a number (nan) degerler verdik



print(arr)



"""



[[10. 20. nan]

 [ 5. nan nan]

 [21. nan 10.]]



"""



#buradaki arrayden bir dataframe olusturuyoruz



df=pd.DataFrame(arr,index=["Index1","Index2","Index3"],columns=["Column1","Column2","Column3"])

#dataframe icin serilerin birlesmis hali diyebiliriz





print(df)



"""

        Column1  Column2  Column3

Index1     10.0     20.0      NaN

Index2      5.0      NaN      NaN

Index3     21.0      NaN     10.0



"""



#NaN olan verileri silmek icin "df.dropna" metodunu kullanabiliriz



df.dropna()

#burada axis=0 oldugu zaman index yani satira gore axis=1 oldugu zaman column yani sutuna gore siler

#yani burda index1,index2,index3 e bakicak NaN varmi yokmu?-varsa satirlari silecek



print(df.dropna())



"""

Columns: [Column1, Column2, Column3]

Index: []



silinmis hali bu sekilde olacaktir



burda index e gore silmis olduk NaN olan satirlari





"""



#column a gore de silebiliriz



df.dropna(axis=1)

#icine parametre olarak "axis=1" dersek bu sefer sutun yani column a gore siler

#icine birsey vermezsek otomatik "axis=0" olarak aldigi icin yani satirlari siler (indexleri yani)



print(df.dropna(axis=1))



"""

       Column1

Index1     10.0

Index2      5.0

Index3     21.0



-sadece column1 de NaN olmadigi icin digerleri silinmis oldu yani



"""



"""

ozet olarak

axis=0 oldugu zaman satirlari (indexleri) siler

axis=1 oldugu zaman sutunlari(columnlari) siler

"""



#eger bir satirda ki butun verileri silmek istemiyorsak "thresh" parametresini kullanabiliriz

#yani satirda 2 tane normal veri var bir tane NaN var diyelim o zaman silme tut komutu verebilirizi



df.dropna(thresh=2)

#yani burada satirda 2 tane sayi varsa tut silme demis oluyoruz



print(df.dropna(thresh=2))



"""

         Column1  Column2  Column3

Index1     10.0     20.0      NaN

Index3     21.0      NaN     10.0



goruldugu gibi burada  satirdaki 2 tane sayi olan yeryerler silinmemis duruyor icinde NaN olmasina ragmen

"""





# "NaN" degerleri yerine bir deger eklemek icin "fillna" yi kullanabiliriz(icinede deger vermemiz gerekiyor)

# fillna(value= buraya ne deger vermek istersek onu yaziyoruz)



print(df.fillna(value=1))

# 1 degerini verirsen NaN olan heryere 1 degeri gelir



"""

        Column1  Column2  Column3

Index1     10.0     20.0      1.0

Index2      5.0      1.0      1.0

Index3     21.0      1.0     10.0



burada NaN degeerleri yerine 1 gelmis oldu







"""



# NaN larin yerin dataframe icindeki tum sayilarini ortalamasini vermek istersek eger



#bunun icin once tum degerleri toplamamiz lazim ortalama degerini bulmak icin. yani first step degerleri toplamak



print(df.sum())



"""

Column1    36.0

Column2    20.0

Column3    10.0

dtype: float64



#sayilari toplayip birer seri haline donusturmus oldu

"""



# daha sonraki adim olarak gostermek istersek bu serileride toplayip tek bir sayi haline getirmek gerekiyor onun icinde:



print(df.sum().sum())

#yani burada yukaridaki seri haline getiriyor sonra o serileride toplayip tek bir seri haline getir komutunu vermis oluyoruz



"""

66.0



cikan sonuc tum sayilarin toplami 



"""



#next step olarakta NaN lari saymazsak 5 tane normal verimiz var toplam 9 veri var NaN lar iler beraber

# toplam veri sayisini bulmak icin "size" fonksiyonunu kullaniyoruz





print(df.size)



"""

9

#toplamda 9 veri var yani



"""



# kac tane NaN veri oldugunu bulmak icin ise "isnull" metodunu kullaniyoruz



print(df.isnull().sum())

#kac tane NaN oldugunu "isnull" ile buluyoruz kac tane oldugunu bulmak icin "sum" ile toplamina bakiyoruz



"""

Column1    0

Column2    2

Column3    2

dtype: int64



hangi columnda kac tane NaN var toplami



"""



print(df.isnull().sum().sum())#yine birtane daha "sum()" eklersek genel toplamini bulmus oluruzu

"""

4

yani genel toplam NaN sayisi



step by step bakarsak asamalara daha basit ve anlasilabilir olur...



"""



# 9 dan 4 u cikarirsek kac tane normal degerimiz oldgunu bulabiliriz

#bunun icin gerekli fonksiyon:



def calculateMean(df):

    totalSum = df.sum().sum()

    totalNum = df.size - df.isnull().sum().sum()



    return totalSum/totalNum



#yani burda total summary yani genel sayilarin toplamini bulduk

#daha sonra size dan yani kac tane veri varsa ordan(9 adet veri) toplam NaN veri sayisini cikarttik

#daha sonra toplami sayilarin toplam veri sayisina bolduk

#yani once sayilarin toplamini bulduk daha sonra genel ortalamayi aradigimiz icin buldugumuz toplami kac tane sayi varsa one bolduk





# simdi NaN larin yerine buldugumuz ortalamalari koymak icin "value=calculateMena" fonksiyonunu yazmamiz lazim



print(df.fillna(value=calculateMean(df)))



"""

        Column1  Column2  Column3

Index1     10.0     20.0     13.2

Index2      5.0     13.2     13.2

Index3     21.0     13.2     10.0



#burada gordugumuz gibi NaN larin yerine buldugumuz ortalamalari koymus olduk





"""
# BURADA DATAFRAME DE KI GROUPBY SORGULARI CALISILMISTIR

# SQL TABLOLARINDA KI GROUPBY ILE BIREBIR AYNI!



import numpy as np

import pandas as pd



dataset = {

        "Departman":["Bilişim","İnsan Kaynakları","Üretim","Üretim","Bilişim","İnsan Kaynakları"],

        "Çalışan": ["Mustafa","Jale","Kadir","Zeynep","Murat","Ahmet"],

        "Maaş":[3000,3500,2500,4500,4000,2000]

        }



#dataset i dataframe'e ceviriyoruz burada

df = pd.DataFrame(dataset)



print(df)



"""

          Departman  Çalışan  Maaş

0           Bilişim  Mustafa  3000

1  İnsan Kaynakları     Jale  3500

2            Üretim    Kadir  2500

3            Üretim   Zeynep  4500

4           Bilişim    Murat  4000

5  İnsan Kaynakları    Ahmet  2000



#burada gordugumuz gibi dataset icinde verilenleri dataframe e cevirmis olduk

#tablo yapisi olusturduk yani

"""



# olusturdugumuz dataframe uzerinde "groupby" islemlerini yapmaya baslayabiliriz

# ornek olarak "groupby" ile departman uzerinde islemler yapalim



DepGroup = df.groupby("Departman")# "DepGroup" adli degiskene atadik

# "Departman" a gore islemk yapmak istedigimiz icin icine onu atiyoruz



print(DepGroup)



"""

<pandas.core.groupby.generic.DataFrameGroupBy object at 0x10cd1c208>



#calistirdigimizda bize boyle bir obje doner



#artik biz bu objenin uzerinde

-toplama

-min deger bulma

-ortalama deger bulma gibi islemleri yapabiliriz





"""



print(DepGroup.sum())



"""

                  Maaş

Departman             

Bilişim           7000

Üretim            7000

İnsan Kaynakları  5500



#burada departmanlara gore toplam maaslari bir araya getirmis olduk

"""



#islemi daha kisa yapabiliriz



print(df.groupby("Departman").sum())

#yine ayni sonucu verir



#sadece tek bir sektorun toplamini almak istersek





print(df.groupby("Departman").sum().loc["Bilişim"])

"""

Maaş    7000

Name: Bilişim, dtype: int64



#sadece bilisim departmaninin maaslarinin toplamini almis olduk



"""

# toplam cikan maas sonucunu integer e cevirip tek bir sayi olarak ciktisini alabiliriz

#yapmamiz gereken tek sey basina "int" yazmak



print(int(df.groupby("Departman").sum().loc["Bilişim"]))

"""

7000

#sadece 7000 olarak cikti verecektir



"""



# "sum" yerine "count" fonksiyonunu kullanirsak



print(df.groupby("Departman").count())

"""

                  Çalışan  Maaş

Departman                      

Bilişim                 2     2

Üretim                  2     2

İnsan Kaynakları        2     2



#bu sefer departmanlarda calisan sayisini bulmus oluyoruz



"""



# "max" degeri kullanimi



print(df.groupby("Departman").max())



"""

                  Çalışan  Maaş

Departman                      

Bilişim           Mustafa  4000

Üretim             Zeynep  4500

İnsan Kaynakları     Jale  3500



#departmanlarda calisan en fazla maas alanlarin ciktisini verir bu sekilde

#isim siralamsida sozlukteki buyukluge gore gerceklesiyor



"""



# "min" deger kullanimi



print(df.groupby("Departman").min())



"""

                Çalışan  Maaş

Departman                     

Bilişim            Murat  3000

Üretim             Kadir  2500

İnsan Kaynakları   Ahmet  2000



"""

# sadece maas degerlerini almak istersek yani isimleri almadan



print(df.groupby("Departman").min()["Maaş"])



"""

Departman

Bilişim             3000

Üretim              2500

İnsan Kaynakları    2000

Name: Maaş, dtype: int64



#sadece maaslari aldik isimler cikartilmis oldu



"""



#dataframe uzerinde adim adim giderek islemlerimizi gerceklestirebiliiriz yani

#ornegin sadece bilisim departmanini almak istersek





print(df.groupby("Departman").min()["Maaş"]["Bilişim"])



"""

3000



#sadece bilisim dep. ciktisini verir



"""



#toplam maaslarin ortalmasini bulmak istersek eger



print(df.groupby("Departman").mean())#buradaki "mean" ortalama bulmak icin kullandigimiz fonksiyon

"""



                  Maaş

Departman             

Bilişim           3500

Üretim            3500

İnsan Kaynakları  2750



#ortalama maaslarin ciktisi

"""

#yine adim adim istedgimiz gibi analiz edebiliriz



print(df.groupby("Departman").mean()["Maaş"]["İnsan Kaynakları"])



"""

2750



#burda sadece insan kaynaklari departmanini ortalama maasini almis olduk yani





"""
import numpy as np

import pandas as pd



#concatenate: eklemek anlamina geliyor. 2 tane dataframe i istersek index e gore istersekte column a gore birbirine ekleyebiliyoruz



dataset1 = {

    "A": ["A1","A2","A3","A4"],

    "B":["B1","B2","B3","B4"],

    "C":["C1","C2","C3","C4"],

}



dataset2 = {

    "A": ["A5","A6","A7","A8"],

    "B":["B5","B6","B7","B8"],

    "C":["C5","C6","C7","C8"],

}



#simdi bu datasetlerden dataframe olusturuyoruyz



df1 = pd.DataFrame(dataset1,index = [1,2,3,4])

df2 = pd.DataFrame(dataset2,index = [5,6,7,8] )



print(df1)



"""

 A   B   C

1  A1  B1  C1

2  A2  B2  C2

3  A3  B3  C3

4  A4  B4  C4

"""



print(df2)



"""

  A   B   C

5  A5  B5  C5

6  A6  B6  C6

7  A7  B7  C7

8  A8  B8  C8



"""



#dataframe lerimizi olusturduk



#bunlari birbirine eklemek icin "concat" metodunu kullaniyoruz



print(pd.concat([df1,df2]))



"""

    A   B   C

1  A1  B1  C1

2  A2  B2  C2

3  A3  B3  C3

4  A4  B4  C4

5  A5  B5  C5

6  A6  B6  C6

7  A7  B7  C7

8  A8  B8  C8



#index lere gore topladigimiz zaman bu sekilde oluyor

# 2 tane dataframe i toplamak icin birbirlerine benzemesi lazim

#axis=0 a gore boyle toplama

"""



# column lari toplamak icin yine axis=1 yapmamiz lazim



print(pd.concat([df1,df2],axis=1))



"""

     A    B    C    A    B    C

1   A1   B1   C1  NaN  NaN  NaN

2   A2   B2   C2  NaN  NaN  NaN

3   A3   B3   C3  NaN  NaN  NaN

4   A4   B4   C4  NaN  NaN  NaN

5  NaN  NaN  NaN   A5   B5   C5

6  NaN  NaN  NaN   A6   B6   C6

7  NaN  NaN  NaN   A7   B7   C7

8  NaN  NaN  NaN   A8   B8   C8



#columnlari toplanmasi sonucunda toplanamayan degerler NaN olarak cikiyor





"""



# join metodu icin ornekler:



dataset3 = {

    "X" : ["X1","X2","X3","X4"],

    "Y" : ["Y1","Y2","Y3","Y4"],

    "anahtar" : ["K1","K2","K5","K4"]

}



#yine burda datasetimizi dataframe e ceviriyoruz



df3 = pd.DataFrame(dataset3,index= [1,2,3,4])



print(df3)



print(df3.join(df1))



"""

    X   Y anahtar   A   B   C

1  X1  Y1      K1  A1  B1  C1

2  X2  Y2      K2  A2  B2  C2

3  X3  Y3      K5  A3  B3  C3

4  X4  Y4      K4  A4  B4  C4





"""
import numpy as np

import pandas as pd



# MERGE ISLEMINDE DATAFRAMELERIN ORTAK OLAN DEGERLERINI ALIP YENI BIR DATAFRAME OLUSTURMAK OLARAK TANIMLAYABILIRIZ

# YANI 2 KUME DUSUNUN ORTAK ELEMANLARINDAN OLUSTURDUGUMUZ KUME MERGE ISLEMIYLE ALIDIGIMIZ DEGERLERE ESIT



dataset1 = {

    "A" : ["A1","A2","A3"],

    "B" : ["B1","B2","B3"],

    "anahtar" : ["K1","K2","K3"]

}



dataset2 = {

    "X" : ["X1","X2","X3","X4"],

    "Y" : ["Y1","Y2","Y3","Y4"],

    "anahtar" : ["K1","K2","K5","K4"]

}



# datasetlerimizi yazdik sonrasinda dataframlerimizi olusturuyoruz



df1 = pd.DataFrame(dataset1,index = [1,2,3])



df2 = pd.DataFrame(dataset2,index = [1,2,3,4])







print(df1)



"""

    A   B anahtar

1  A1  B1      K1

2  A2  B2      K2

3  A3  B3      K3



"""



print(df2)



"""

   X   Y anahtar

1  X1  Y1      K1

2  X2  Y2      K2

3  X3  Y3      K5

4  X4  Y4      K4



"""



#BURADA birtane "anahtar" column una gore dataframeleri birlestirecegiz



# burada "anahtar" column una gore islem gerceklestirdigimiz zaMAN "K1" ve "K2" satirlarinin ortak oldugunu goruyoruz



"""

        FARK



JOIN : indexler uzerinden 

MERGE : columnlar uzerinden yapilir



"""



print(pd.merge(df1,df2,on = "anahtar"))

#burda onemli olana "on" parametresi "on" parametresine " anahtar" i yazdigimizda o zamana " anahtar kelimesine gore islem yapar



"""

   A   B anahtar   X   Y

0  A1  B1      K1  X1  Y1

1  A2  B2      K2  X2  Y2



Bu sekilde "anahtar kelimesine gore columnlari ortak alarak islem yapilir



"""
import pandas as pd

import numpy as np



df = pd.DataFrame({

    "Column1":[1,2,3,4,5,6],

    "Column2":[100,100,200,300,300,100],

    "Column3":["Mustafa","Kamil","Emre","Ayşe","Murat","Zeynep"]

})



#modullerimizi dahil ettikten sonra dataframemiz olusturuyoruz



print(df)



"""

   Column1  Column2  Column3

0        1      100  Mustafa

1        2      100    Kamil

2        3      200     Emre

3        4      300     Ayşe

4        5      300    Murat

5        6      100   Zeynep



"""



# "head" kullanimi --> "head(n = 3)" burda " n=3 " dersek sadece ilk 3 satiri alir!

# anlasilacagi uzere "n" e verdigimiz degere gore alinacak satir sayisi belirlenir



print(df.head(n=3))



"""

 Column1  Column2  Column3

0        1      100  Mustafa

1        2      100    Kamil

2        3      200     Emre



goruldugu gibi burada sadece ilk 3 satir alinmis olur



"""



# dataframe icinde birbirini tekrar eden degerler var. Kac tane farkli deger var onu gormek icin:

# Kullanacagimiz fonksiyon: "unique"



print(df["Column2"].unique())



"""



[100 200 300]



"""



# "unique" degerlerin kac adet oldugunu bulmak icin ise: "nunique" fonksiyonu kullaniyoruz



print(df["Column2"].nunique())



"""

3

toplamda birbirinden farkli toplamda 3 adet unique yani essiz deger var



"""



# "value_count" fonksiyonu islevi



print(df["Column2"].value_counts())



"""

00    3

300    2

200    1

Name: Column2, dtype: int64



burada Column2 icinde hangi degerden kacar adet oldugunu verir bu fonksiyon



"""





# Ornekleri cesitlendirerek ogrenmek acisindan; "Column1" deki 4 ten buyuk degerleri ve 300 olan degerleri nasil buluruz

# burada dataframe filtreleme islemleri yapacagiz



print(df[df["Column1"] >=4])



"""

 Column1  Column2 Column3

3        4      300    Ayşe

4        5      300   Murat

5        6      100  Zeynep



burdq 4 ten buyuk degerler gelmis oldu



"""



#ayni zamanda son ornege ek olarak bir sart koyabiliriz ornegin "column2" de ki degerler de 300 olacak sekilde al diyebiliriz



print(df[(df["Column1"] > 2) & (df["Column2"] == 300)])



"""

 Column1  Column2 Column3

3        4      300    Ayşe

4        5      300   Murat



"""



# Column2 de ki butun degerleri bir sayi ile carpmak istersek eger:

# Column2 de ki her degerin uzerinde bir fonksiyon uygulamamiz lazim

# Bunun icinde pandas da ki "apply" fonksiyonunu kullanacagiz



def times3(x):

    return x * 3

#fonksiyonumuzu yazdik



print(times3(3))

"""

9



fonksiyon bu sekilde caliscak icine aldigi degeri 3 le carpar 



"""

#simdi bu fonksiyonu "Column2" deki herbir degere uygulamak icin yapmamiz gereken:



print(df["Column2"])



"""

0    100

1    100

2    200

3    300

4    300

5    100

Name: Column2, dtype: int64



normalde "column2" bu sekilde

"""

print(df["Column2"].apply(times3))



"""

0    300

1    300

2    600

3    900

4    900

5    300

Name: Column2, dtype: int64



"Column2" de ki butun degerleri goruldugu gibi 3 ile carpmis olduk



"""

# Eger ki "Column2" yi son halinde aldigi degerler ile guncellemek istersem:



df["Column2"] = df["Column2"].apply(times3)



print(df["Column2"])



"""

0    300

1    300

2    600

3    900

4    900

5    300

Name: Column2, dtype: int64



"Column2" yi yeni degerlerine guncellemis olduk yani artik 3 ile carpilmis haline guncellenmis oldu

"""



# "lambda" fonksiyonunu kullanarak islem yaoma( lambda : fonksiyon olusturmak icin kullaniyoruz "def" yerine tek satirda yazilabilir



print(df["Column2"].apply(lambda x : x*2))



"""

0     600

1     600

2    1200

3    1800

4    1800

5     600

Name: Column2, dtype: int64



Burada goruldugu gibi ayri bir fonksiyon olusturup eklemek yerine direkt "lambda" ile halledilebilir



"""



# "len" fonksiyonu ornek:



print(df["Column3"].apply(len))



"""

0    7

1    5

2    4

3    4

4    5

5    6

Name: Column3, dtype: int64



"column3" teki stringlerin uzunluklarini "len" ile tek tek gormus olduk

    

"""



# Columnlardan birini silmek istersek: "drop" fonksiyonunu kullaniyoruz



# columnlarin oldugu axis=1 olmasi lazim satirlarin oldugu axis=0



print(df.drop("Column3",axis=1))



"""

   Column1  Column2

0        1      300

1        2      300

2        3      600

3        4      900

4        5      900

5        6      300



Column3 burda silinmis oldu





"""



# eger burda silme islemini gerceklestirdikten sonra son haline guncellemek istersek "inplace = True" parametresini yazmamis lazim



print(df.drop("Column3",axis=1 , inplace=True))



#son haliyle "Column3" silinmis haline yani guncellenmis oldu dataframe





# dataframe uzerinde kac tane sutun yani calumn oldugunu bulmak icin:



print(df.columns)



"""



Index(['Column1', 'Column2'], dtype='object')



bu sekilde cikti verir

bunu ozellikle buyuk veri setlerinde kullaniyoruz



"""

# dataframe icinde kac tane satir yani index oldugunu bulmak icin



print(df.index)



"""



RangeIndex(start=0, stop=6, step=1)



"""

print(len(df.index))



"""

6 ciktisini verir



kac tane satir oldugunu "len" fonksiyonu ile bulabiliriz



"""



# indexlerini isimlerine bakmak icin:



print(df.index.names)



"""

burada: "[None]" ciktisini verecektir cunku dataframe e baktigimizda indexklerin isimleri olmadigini gorecegiz

"""



# bu arada dataframe in son haline bakalim



print(df)



"""

Column1  Column2

0        1      300

1        2      300

2        3      600

3        4      900

4        5      900

5        6      300



"""



# dataframe i column a gore "kucukten buyuge" dogru siralamak icin:



print(df.sort_values("Column2"))



"""

   Column1  Column2

0        1      300

1        2      300

5        6      300

2        3      600

3        4      900

4        5      900



"""



# "buyukten kucuge" dogru siralamak istersek eger: normalde "sort_value" fonks kullandigimizda icinde "ascending = True" default olarak verir

# "ascending = False" olarak degistirirsek eger o zaman buyukten kucuge dogru siralar



print(df.sort_values("Column2",ascending=False))



"""

   Column1  Column2

3        4      900

4        5      900

2        3      600

0        1      300

1        2      300

5        6      300



"""



#buraya kadar kullanilan operasyonlar "data mining" isinde cok kullanilan operasyonlar



# "PIVOT TABLE" mantigi

df2 = pd.DataFrame({

    "Ay" : ["Mart","Nisan","Mayıs","Mart","Nisan","Mayıs","Mart","Nisan","Mayıs"],

    "Şehir":["Ankara","Ankara","Ankara","İstanbul","İstanbul","İstanbul","İzmir","İzmir","İzmir"],

    "Nem":[10,25,50,21,67,80,30,70,75]

})



print(df2)



"""

    Ay     Şehir  Nem

0   Mart    Ankara   10

1  Nisan    Ankara   25

2  Mayıs    Ankara   50

3   Mart  İstanbul   21

4  Nisan  İstanbul   67

5  Mayıs  İstanbul   80

6   Mart     İzmir   30

7  Nisan     İzmir   70

8  Mayıs     İzmir   75



"""



# burada yapmak istedigimiz sey bu dataframe i daha guzel,toplu gostermek(yani programlama mantigida budur zaten kolaylastirmak)



#bu dataframe uzerinden bir pivot table olusturacagiz once



print(df2.pivot_table(index = "Şehir",columns ="Ay",values = "Nem"))



"""

Ay        Mart  Mayıs  Nisan

Şehir                       

Ankara      10     50     25

İstanbul    21     80     67

İzmir       30     75     70



yani buradaki amac dataframe i daha toplu duzgun bir hale getirmek

"""



# baska sekilde column ve indexlerin yerini degistirerekte yazmamiz mumkun



print(df2.pivot_table(index = "Ay",columns ="Şehir",values = "Nem"))



"""Şehir  Ankara  İstanbul  İzmir

Ay                            

Mart       10        21     30

Mayıs      50        80     75

Nisan      25        67     70



"""
import pandas as pd

import numpy as np



# you can use datas from kaggle as "csv"

# https://www.kaggle.com/datasnaek/youtube-new  this is the link that i am usind data from kaggle

# and when you download data which is "USvideos.csv" it must be in same folder

# "USvideos.csv" is in my folder already



dataset = pd.read_csv("USvideos.csv")



print(dataset)



# when you print out dataset its gonna look like this:



"""

          video_id  ...                                        description

0      2kyS6SvSYSE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1      1ZAPwfrtAFY  ...  One year after the presidential election, John...

2      5qpjK5DgCt4  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3      puqaWrEC7tY  ...  Today we find out if Link is a Nickelback amat...

4      d380meD0W0M  ...  I know it's been a while since we did this sho...

...            ...  ...                                                ...

23357  pH7VfJDq7f4  ...  ...and other musings on thermal movement of la...

23358  hV-yHbbrKRA  ...  Visit Our Website! ▶ http://www.townsends.us/ ...

23359  CwKp6Xhy3_4  ...  Chris Young's Hangin' On from his #1 album Los...

23360  vQiiNGllGQo  ...  very wholesome stuff.\n\nThis video was taken ...

23361  2afSbqlp5HU  ...  The new marvel film Black Panther is set in th...



[23362 rows x 16 columns]



there are 23362 index and 16 columns in this dataset

"""



# if you wanna delete some things from this dataset you need to yous "drop" function and if you wanna delete from column you have to write "axis=1" alos

# if you wanna delete some things from index(row) you do not have to write anything because its axis=0 a=s default already



newdataset1 = dataset.drop(["video_id","trending_date"],axis = 1)



# when you look at "the newdataset1" you will see there is no "video_id","trending_date" anymore



print(newdataset1)



"""

                                                  title  ...                                        description

0                     WE WANT TO TALK ABOUT OUR MARRIAGE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1      The Trump Presidency: Last Week Tonight with J...  ...  One year after the presidential election, John...

2      Racist Superman | Rudy Mancuso, King Bach & Le...  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3                       Nickelback Lyrics: Real or Fake?  ...  Today we find out if Link is a Nickelback amat...

4                               I Dare You: GOING BALD!?  ...  I know it's been a while since we did this sho...

...                                                  ...  ...                                                ...

23357                                Why Bridges Move...  ...  ...and other musings on thermal movement of la...

23358                      Macaroni - A Recipe From 1784  ...  Visit Our Website! ▶ http://www.townsends.us/ ...

23359                           Chris Young - Hangin' On  ...  Chris Young's Hangin' On from his #1 album Los...

23360      Elderly man making sure his dog won't get wet  ...  very wholesome stuff.\n\nThis video was taken ...

23361         How to speak like Black Panther - BBC News  ...  The new marvel film Black Panther is set in th...



[23362 rows x 14 columns]



as you can see "video_id","trending_date" columns dropped (deleted) from dataset

and there is 14 column after that



"""



# we created new dataset "as newdataset". if you wanna write this dataset as csv :



newdataset1.to_csv("UsVideosNew.csv")



# we created new csv in folder now wherever is your location



# if you do not wanna take indexes: the purpose is here making this code,app,programme(whatever) better and easier



newdataset1.to_csv("UsVideosNew.csv" , index=False)



print(newdataset1)



# that will be look better than last one



"""

                                                  title  ...                                        description

0                     WE WANT TO TALK ABOUT OUR MARRIAGE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1      The Trump Presidency: Last Week Tonight with J...  ...  One year after the presidential election, John...

2      Racist Superman | Rudy Mancuso, King Bach & Le...  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3                       Nickelback Lyrics: Real or Fake?  ...  Today we find out if Link is a Nickelback amat...

4                               I Dare You: GOING BALD!?  ...  I know it's been a while since we did this sho...

...                                                  ...  ...                                                ...

23357                                Why Bridges Move...  ...  ...and other musings on thermal movement of la...

23358                      Macaroni - A Recipe From 1784  ...  Visit Our Website! ▶ http://www.townsends.us/ ...

23359                           Chris Young - Hangin' On  ...  Chris Young's Hangin' On from his #1 album Los...

23360      Elderly man making sure his dog won't get wet  ...  very wholesome stuff.\n\nThis video was taken ...

23361         How to speak like Black Panther - BBC News  ...  The new marvel film Black Panther is set in th...



[23362 rows x 14 columns]



"""

# reading writing on excel files with pandas: using "read_excel"

# excel file mustbe in same folder



excelset = pd.read_excel("excelfile.xlsx")



print(excelset)



"""

 Unnamed: 0  Column1  Column2  Column3  Column4

0     Index1       10       50       90      130

1     Index2       20       60      100      140

2     Index3       30       70      110      150

3     Index4       40       80      120      160



"""



# if you wanna add column on this dataset which is excelset



excelset["Column5"] = [170,180,190,200]



print(excelset)



"""

 Unnamed: 0  Column1  Column2  Column3  Column4  Column5

0     Index1       10       50       90      130      170

1     Index2       20       60      100      140      180

2     Index3       30       70      110      150      190

3     Index4       40       80      120      160      200



column will be added already 



"""



# if you wanna save this excel file to different(another excel file) excel file: using "to_excel"



excelset.to_excel("excelfilenew.xlsx")



"""

Unnamed: 0  Column1  Column2  Column3  Column4  Column5

0     Index1       10       50       90      130      170

1     Index2       20       60      100      140      180

2     Index3       30       70      110      150      190

3     Index4       40       80      120      160      200



it will be saved as new excel file right in the location folder



"""



# If you wanna take dataset from website: using "read_html"



new = pd.read_html("http://www.contextures.com/xlSampleData01.html",header = 0)



print(new)
import pandas as pd

USvideos = pd.read_csv("../input/USvideos.csv")