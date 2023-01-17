list=[]
for i in range(1,101):
    if i%3==0 & i%5==0 :
        list.append(i)
print(list)
metin=input("Metin giriniz:")
for i in metin:
    print(i)
list=[]
s1=int(input("Sayıyı giriniz:"))
s2=int(input("Sayıyı giriniz:"))
for i in range(s1+1,s2):
    list.append(i)
sonuc=sum(list)
print(list,sonuc)
def Cevre(s1):
    s1=int(input("Yarıçapı giriniz:"))
    cevre=2*3*(s1*s1)
    print(cevre)
Cevre(s1)
def Dikdortgen(s1,s2):
    s1=int(input("Kısa kenarı giriniz:"))
    s2=int(input("Uzun kenarı giriniz:"))
    sonuc=s1*s2
    print(sonuc)
Dikdortgen(s1,s2)