import numpy as np



arr1 = np.array([10,20,30,40,50,60])

arr2 = np.array([2,3,4,5,6,7])



print(arr2+arr1)

#burada her bir birbirine es elamanlari toplar



"""[12 23 34 45 56 67]

"""



#ayni sekilde diger islemleride yapabiliriz



print(arr1*arr2)



print(arr1+10)#burada da arr1 in herbir elemanina 10 ekler

print(arr2-2)#ayni sekilde arr2  nin herbir elemanindan 2 cikartir



print(arr1/3)



# KAREKOK ALMA



print(np.sqrt(arr1))

"""[3.16227766 4.47213595 5.47722558 6.32455532 7.07106781 7.74596669]

"""

#burada butun elemanlarin karekokunu alir



"""

numpy icinde kullanilan cok fazla dokumantasyonn var

ihtiyaca gore bakilabilir 





"""
data_list = [1,2,3]



print(data_list)



import numpy as np



arr = np.array(data_list)



print(arr)



data_list2= [[10,20,30],[40,50,60],[70,80,90]]



arr2 = np.array(data_list2)



print(arr2)



#ya da daha onceden liste olusturmak zorunda degiliz direk listeyi icine de atabiliriz



arr3 = np.array([1,2,3,4,5,6,7,8,9,0])



print(arr3)



#genel olarak numpy listelerinin olusturulmasi bu sekilde



#liste icindeki elemanlara erismek icinse listelerde yaptigimiz gibi



print(arr3[0])#burada arr3 listesi icinde 0. index e erismik olduk



#arr2 matrisindeki 2 ye 2. indexi bulmak istedigimizde yani 2. indexin 2. indexi(o da 90 oluyor)

#gosterim bicimi



print(arr2[2,2,])#sonuc 90 olacaktir



#numpy uzerinde yine range metodunu kullanabiliriz



print(np.arange(10,20))#burada 10-20 arasindaki degerleri verir (20 dahil degil)



print(np.arange(0,99,3))#burada da 0-99 arasindaki degerleri 3 er atlayarak veriri



#10 tane ornek olarak 0 in depolandigi bir array yazmak istersek



print(np.zeros(10))



print(np.ones(10))#ya da 1 depolamak istersek "ones"



#bunlar tek boyutlu arrayler cok boyutlu matrisleride 0 ile doldurabiliriz



print(np.zeros((2,2)))#burada 2ye2 bir matris olustur icine de 0 degerlerini ver demis oluyoruz



print(np.zeros((3,6)))



"""BURADAKI ORNEKLERIN HER BIRINI AYRI BIR PENCEREDE KARISMAMASI ACISINDAN DENENEBILIR"""





# "linspace" kullanimi



print(np.linspace(0,100,5)) #burada 0-100 arasindaki degerleri 5 esit parcaya bolerek 5 tane deger olarak sakla demek

# 0  ile 100 de dahil





print(np.eye(6)) #burada 6 ya 6 bir matris olusturacak ve kosegenlerde hepsi 1 degerine sahip olacak

#yani bir tane birim matrisi olusturmus oluyoruz





print(np.random.randint(0,10))# 0 dahil 10 dahil degil

#burada 0 ile 10 arasinda rastgele degerler donecek

print(np.random.randint(10))#burada da 0 yazmasakta 0 yazmis gibi hareket edecek



print(np.random.randint(1,10,5))#burada 1-10 arasinda deger donsun ancak 5 tane deger donerek bunu bir tane array icinde sakla





print(np.random.randn(5))#burada sadece 0-5 arasinda degerler ortaya cikmayacak

#burada negatif degerlerde ortaya cikacak

#yani burada Gaussian distribution yapmis olduk



# [ 1.23090143 -0.28569486 -0.42070302 -0.73690302  0.98744533] bu sekilde ciktilar verecektir





import numpy as np



arr=np.arange(25)



print(arr)#burada 25 dahil degil 0 ile 25 arasindaki degerleri vermis olacak



# RESHAPE KULLANIMI



print(arr.reshape(5,5))



#burada arr yi 5-5 bir matrise donusturmus olduk



#ama matrisin 5x5 lik olmasi lazim yani 5x4 yaparsak hata verecektir



"""

sonuc bu sekilde olacaktir=



[[ 0  1  2  3  4]

 [ 5  6  7  8  9]

 [10 11 12 13 14]

 [15 16 17 18 19]

 [20 21 22 23 24]]

"""



newArray = np.random.randint(1,100,10)# 1-100 arasinda 10 tane deger uret komutu



print(newArray)



print(newArray.max())# icindeki en buyuk degeri verir rastgele(yani her defasinda farkli en buyuk degeri doner)

print(newArray.min())# icindeki en kucuk degeri verir rastgele(yani her defasinda farkli en buyuk degeri doner)



print(newArray.sum())# sum ile icindeki butun degerlerin toplamini almis oluruz





print(newArray.mean()) # "mean" ile tum degerlerin ortalamisini bulabiliriz



""" YINE BURADA DA TUM KODLARI AYRI AYRI BASKA BIR PENCEREDE KARISMAMASI ACISINDAN DENENEBILIR. CIKTI DEGERLERIN DAHA IYI GORULMESI ACISINDAN"""





print(newArray.argmax())# burada da max. sayinin oldugu indexi verir

print(newArray.argmin())# burada da min. sayinin oldugu indexi verir





# DETERMINANT HESAPLAMA



detArray = np.random.randint(1,100,25)#burada 1-100 arasinda (100 dahil degil) 25 tane random deger olusturduk





print(detArray)#ciktisina baktik



detArray = detArray.reshape(5,5)#daha  sonra bunu 5 e 5 lik matrise donusturduk





print(detArray)#ciktisina baktik



print(np.linalg.det(detArray))#burada determinantini hesapladik


