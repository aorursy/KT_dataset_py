import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data.corr()
data.info()
data.corr()
#data visuilation using corralation
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.1f', ax = ax)#annot -> sayılar gözüksünmü?, linewidth -> kutucuklar arası boşluklar
#fmt -> virgülden sonra gösterilecek basamak sayısı, ax -> boyut ayarlama
plt.show()
data.head(10)
data.columns
data.Speed.plot(kind = 'line', color = 'g',label = 'speed(hız)', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-')
data.Defense.plot (color = 'r', label = 'defense(defans)', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')
plt.legend(loc = 'upper right')
plt.xlabel('attack')
plt.ylabel('defance')
plt.title('speed defence line plot')
plt.show()
data.columns
data.head()
# x = attack   y = defense ikisi arasındaki ilişki
data.plot(kind = 'scatter', x = 'Attack' ,y = 'Defense', alpha = 0.5, grid = True, color ='black')
#plt.scatter(data.Attack, data.Defense, color = 'red', alpha = 0.5) // yukardaki ile aynı işi yapar (grid kabul etmiyor)
plt.xlabel('attack(atak)')
plt.ylabel('Defence(defans)')
plt.title('attack defence scatter plot ')
plt.show()


data.head()
data.plot(kind = 'scatter', x = 'Defense', y = 'Sp. Def', alpha = 0.5, grid = True, color ='green')
plt.xlabel('defanse')
plt.ylabel('sp def')
plt.title('ben ne istersem o')
plt.show()

data.Speed.plot(kind = 'hist', bins = 50, figsize  = (10,10)) #bins -> çubuk sayısı.
plt.show()
data.head()
#data.Defense.plot(kind = 'hist', bins = 400, figsize =(10,10), color = 'black')
data.plot(kind = 'hist', x = 'Name',y = ['HP','Attack'], bins = 500, figsize = (20,10))
plt.show()
data.Speed.plot(kind = 'hist', bins = 50, figsize  = (10,10)) #bins -> çubuk sayısı.
plt.clf() # figür temizleyicisi
dic = {'türkiye':'ankara', 'kayseri':'talas'}
print (dic.keys())
print (dic.values())
ss = {'hello' : 'selam', 'yes' : 'evet', 'no' : 'hayır'}
print (ss.keys())
print (ss.values())
print(ss)
ss['hello'] = 'merhaba'
ss['yes'] = 'evet'
print(ss)
ss['see'] = 'gör'
print(ss)
del ss['see']
print(ss)
print('hello' in ss)
ss.clear()
print(ss)
dic['türkiye'] = "istanbul"
dic['kayseri'] = "melikgazi"
print(dic)
# buraya kadar var olanı değiştirme
dic['france'] = "paris"# yeni key ve value ekleme
print(dic)
del dic['kayseri']#key silme (value de gider)
print(dic)
print ('france' in dic)
dic.clear()
print(dic)
#del dic
print (dic)
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data.info()
series = data['Defense']
print(type(series))
data_frame = data[['Defense']]
print(type(data_frame))

#compression (karşılaştırma) operatörleri:
print(2<3)
print(2>5)
print(5!=3)
print(True and False)
print(True or False)


#fitering
x = data['Defense']>200 
data[x]
a = np.logical_and(data['Legendary'] == 1, data['Speed']>130) 
data[a]
x = data[np.logical_and(data['Defense']>200, data['Attack']>100)] 

data[(data['Defense']>150) & (data['Attack']>100)]
i = 0
while i != 5:
    print("İ",i+1,":", i)
    i+=1
print(i,': 5')
i = 0
while i != 12:
    print(i)
    i+=2
sss = [1,2,3,4,5,6]
toplam = 0
carp = 1
for i in range (0,len(sss)):
    toplam+=sss[i]
    carp*=sss[i]
print (toplam)
print (carp)
list = [1,2,3,4,5]
for i in list:
    print('i : ',i)
print('\n')

#enumerate: hem değerlere hemde değerlerin indexlerine erişim sağlamak için kullanıyoruz.
for index, value in enumerate(list):
    print(index, ":", value)
print('\n')


#dictionary ler için:
#dictionary'lerin içindeki key ve value değerlerini de for döngüsü kullanarak gerçekleştirebiliriz.
dic = {'türkiye' : 'ankara', 'france' : 'paris'}
for key, value in dic.items():
    print (key, " : ", value)
print('\n')


for index, value in data [['Attack']][0:1].iterrows():
    print(index, " , ", value)
    
data.head()
for index, value in data [['Attack']][0:5].iterrows():
    print(index, " , ", value)
data.head(10)
def tuble_ex():
    """return t """
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print (a)
print (b)
print (c)
x = 5 #global scope 
def f():
    x = 7 # local scope 
    return x
print (x)
print (f())
# local ve global scopelar tanımlandı ikisinin ismi de x fakat ayrı ayrı kullanılabiliyor
import builtins
dir(builtins)
def karesi():
    """değerin karesini bulacak (return edecek)"""
    def toplam():
        x = 2
        y = 3
        z = x + y
        return z
    return toplam()**2
print (karesi())
def f(a,b = 1):
    """bu fonksiyon a ve b değerleri alıyor fakat b değeri alınmaz ise b default olarak 1 kabul ediliyor"""
def s(*args):
    """args lar birden fazla olabilir flexible argümants"""
def z(a,b = 3, c = 6):
    """default fonksiyona örnek"""
    d = a + b + c
    return d
print(z(1))
print(z(1,2))
print(z(1,2,3))

    
def f(*args):
    """alınması gereken parametre sayısı belirli olmayan durumlarda kullanılan bir method dur esnek argüman olarak isimlendirilir(flexible)"""
    for i in args:
        print (i)
f(1)
print("")
f(1,2,3)
'''def f(*kwargs):
    for key , value in kwargs.items():
        print(key, " ", value)
f(country = 'Turkey', capital = "Ankara", population = 123456)'''
def f(**kwargs):# çift yıldıza dikkat et!!!
    for key, value in kwargs.items():
        print(key, " ", value)
f(a = "turan", b = "555", c = "adana")
def karesi(x):
    return x**2
karesi(5)
karesi = lambda x: x**2
print(karesi(3))
multiver = lambda x,y,z : x+y+z
print (multiver(3,3,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)
print (*y)
name = "Turan"
it = iter(name)
print(next(it))
print(next(it))
print(next(it))
print(next(it))
print(next(it))

a = [1,2,3]
b = [4,5,6]
z = zip(a,b)
li = list(z)
print(li)



