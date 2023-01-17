print('Deprem Raporu')
print('sakin olunuz')






def deprem(oncuartcıdeprem,anadeprem,*mesaj):

    if oncuartcıdeprem<anadeprem:

        print('Deprem buyukluk:',mesaj[2] )

    else:

            print('deprem buyukluk:',mesaj[1])





deprem(3,5,'deprem oldu','deprem tehlikeli değil','deprem tehlikeli')
#depremsonrasigbt

def depremsonrasigbt(isim,soyisim,yas,ilce,il = 'istanbul'):

    print('İsminiz:',isim,'soyisminiz:',soyisim,'yasiniz:',yas,'ilceniz:',ilce,'iliniz:',il)
depremsonrasigbt('tugce','aydin',15,'sariyer')
#aileyemesaj



def ailememesaj(isim, *mesaj):

    print('ben',isim,mesaj[0],mesaj[2])
ailememesaj('tuğçe','iyiyim', 'kötüyüm','siz nasılsınız')
#ihtiyacduyulansulitresi



def sulitre(kisi1,k2,k3,k4,k5):

    toplam=kisi1+k2+k3+k4+k5

    print('toplamda ',toplam,'litre suya ihtiyacımız var')
sulitre(3,5,3,5,2)
#ihtiyacduyulanailebasinacadirsayisi



aile= lambda x : x*2

print("toplamda  " , aile(25),'çadıra ihtiyacımız var')