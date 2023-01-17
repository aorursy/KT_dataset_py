ihtimal=pow(5,6)/pow(6,10)
ihtimal
kombinasyon = (10*9*8*7)/(4*3*2*1)
kombinasyon
ihtimal * kombinasyon
import numpy as np

liste=[]
for i in range (0,100000):
    liste.append(np.random.randint(1,7,10))
liste[0:3] # 100,000 kere 10'ar zar atışı yaptık ve bu atışları listeye kaydettik
SayacListesi=[]
for j in liste:
    sayac=0
    for i in j:
        if(i==6):
            sayac+=1
    SayacListesi.append(sayac)
SayacListesi[0:3]#100,000 adet listede geçen "6" miktarlarını hesaplayıp kaydettik. 
SayacSon=0
for i in SayacListesi:
    if(i==4):
        SayacSon+=1
SayacSon#SayacListesindeki 4 leri aradık ve miktarlarını saydık. Bu çıkan miktarı toplam liste sayısına böldüğümüzde olasılığımız hesaplanmış olur
SayacSon/len(SayacListesi)