# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/woman-murdering-in-turkey-20082020/women_who_have_been_murdered_in_turkey.csv')

data.head() 
data = data.replace({"^\s*|\s*$":""}, regex=True) #verilerimizdeki boslukları yok edebilmek için

year = data['year'] = data['year'].astype('int') #yearımızın degeri float olarak görünüyor o yüzden int e çevirdik

data['city'] = data['city'].astype('str')

data.columns
data.info()
data.tail()
data

yillaragoreolum = data.year.value_counts().head(13).sort_index()

yillaragoreolum
kimtarafindan = data.killer1.value_counts()

kimtarafindan.head(15)
data.statusofkiller.value_counts().head(7)
data[

    ((data.killer1 == 'Kocasi') | (data.killer2 == 'Kocasi')) &

    ((data.killingway1 == 'Atesli Silah') | (data.killingway2 == 'Atesli Silah')|(data.killingway3 == 'Atesli Silah'))

]
yillaragoreolenkadinlar =data.year.plot(kind = 'hist' , bins = 30 , figsize = (10,6) , range = (2008 , 2020) , label = 'Yil' )

yillaragoreolenkadinlar.set_title("Ocak 2008 ve Ağustos 2020 Tarihleri Arasında Ölen Kadınlar" , fontsize = 12)

yillaragoreolenkadinlar.set_xlabel("Yıl", fontsize = 12)

yillaragoreolenkadinlar.set_ylabel("Ölen Kadınlar", fontsize = 12)

plt.show()
degerler = data.killingway1.value_counts().head(7)

nasiloldurulduler = degerler.plot(kind='pie'  , figsize = (9, 8) , startangle = 60 , shadow = False , autopct = "%1.1f%%")

nasiloldurulduler.set_title("Öldürülme Şekli" , fontsize = 15)

nasiloldurulduler.set_ylabel("" , fontsize = 15)

plt.show()
degerler2 = data.protectionorder.value_counts().head(3)

korumakarari = degerler2.plot(kind='pie'  , figsize = (9, 8) , startangle = 60 , shadow = False , autopct = "%1.1f%%")

korumakarari.set_title("Koruma Kararı Durumu" , fontsize = 15)

korumakarari.set_ylabel("" , fontsize = 15)

plt.show()
degerler3 = data.statusofkiller.value_counts().head(6)

katildurumu = degerler3.plot(kind='pie'  , figsize = (9, 8) , startangle = 60 , shadow = False , autopct = "%1.1f%%")

katildurumu.set_title("Katillerin Durumu" , fontsize = 15)

katildurumu.set_ylabel("" , fontsize = 15)

plt.show()
# Kızı ve Oğlu Tarafından

ailesi1 = data[(data['killer1'] == 'Oglu') | (data['killer1'] == 'Kizi')]

ailesi1filtre = ailesi1[['year','killer1']]

ailesi1filtre.year.value_counts().head(13).sort_index()

listeailesi1 = list(ailesi1filtre.year.value_counts().head(13).sort_index())

listeailesi1.insert(0,0) #Bazı yıllarda veri yok bu yüzden o yıllara 0 ekledik

#Abisi , Kardesi , Erkek Kardesi Tarafından

ailesi2 = data[(data['killer1'] == 'Abisi') | (data['killer1'] == 'Kardesi') | (data['killer1'] == 'Erkek Kardesi') ]

ailesi2filtre = ailesi2[['year','killer1']]

ailesi2filtre.year.value_counts().head(13).sort_index()

listeailesi2 = list(ailesi2filtre.year.value_counts().head(13).sort_index())

listeailesi2.append(0)

# Annesi Babası Tarafından

ailesi3 = data[(data['killer1'] == 'Babasi') | (data['killer1'] == 'Annesi') ]

ailesi3filtre = ailesi3[['year','killer1']]

ailesi3filtre.year.value_counts().head(13).sort_index()

listeailesi3 = list(ailesi3filtre.year.value_counts().head(13).sort_index())

listeailesi3.insert(4,0)

#Torunu Tarafından

ailesi4 = data[(data['killer1'] == 'Torunu' ) | (data['killer1'] == 'Yegeni') | (data['killer1'] == 'Amcasi') | (data['killer1'] == 'Dayisi')|(data['killer1'] == 'Kuzeni') | (data['killer1'] == 'Akrabasi')]

ailesi4filtre = ailesi4[['year','killer1']]

ailesi4filtre.year.value_counts().head(13).sort_index()

listeailesi4 = list(ailesi4filtre.year.value_counts().head(13).sort_index())

#Verimizi Listeye Alalım

datamiz1 = []

datamiz1.append(listeailesi1)

datamiz1.append(listeailesi2)

datamiz1.append(listeailesi3)

datamiz1.append(listeailesi4) 

# X axisindeki değerleri atayalım

barWidth = 0.22

r1 = np.arange(len(datamiz1[0]))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

#plotu oluşturalım ve görüntüleyelim

plt.figure(figsize=(18,9))

plt.grid(zorder = 0)

plt.bar(r1, datamiz1[0], color='#1089bc', width=barWidth, edgecolor='white', label='Çocukları Tarafından')

plt.bar(r2, datamiz1[1] , color='#ff3300', width=barWidth, edgecolor='white', label='Kardeşi Tarafından')

plt.bar(r3, datamiz1[2], color='#ffa500', width=barWidth, edgecolor='white', label='Annesi veya Babası Tarafından')

plt.bar(r4, datamiz1[3], color='purple', width=barWidth, edgecolor='white', label='Akrabaları Tarafından')

plt.xlabel('1. Dereceden Kan Bağı Olanlar Tarafından Öldürülen Kadınlar', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(datamiz1[0]))], ['2008', '2009', '2010', '2011', '2012','2013', '2014', '2015', '2016', '2017','2018', '2019', '2020'])

plt.legend()

plt.show()

# Kocası Tarafından Öldürülenler

kocasitarafindan = data[data['killer1'] == 'Kocasi' ]

kocasitarafindanfiltre = kocasitarafindan[['year','killer1']]

kocasitarafindanfiltre.year.value_counts().head(13).sort_index()

listekocasi = list(kocasitarafindanfiltre.year.value_counts().head(13).sort_index())

#Eski Kocası Tarafından Öldürülenler

eskikocasitarafindan = data[data['killer1'] == 'Eski Kocasi' ]

eskikocasitarafindanfiltre = eskikocasitarafindan[['year','killer1']]

eskikocasitarafindanfiltre.year.value_counts().head(13).sort_index()

listeeskikocasi = list(eskikocasitarafindanfiltre.year.value_counts().head(13).sort_index())

#Eski Sevgilisi Tarafından Öldürülenler

eskisevgilisitarafindan = data[(data['killer1'] == 'Eski Sevgilisi') ]

eskisevgilisitarafindanfiltre = eskisevgilisitarafindan[['year','killer1']]

eskisevgilisitarafindanfiltre.year.value_counts().head(13).sort_index()        

listeeskisevgilisi = list(eskisevgilisitarafindanfiltre.year.value_counts().head(13).sort_index())

#Sevgilisi Tarafından Öldürülenler

sevgilisitarafindan = data[data['killer1'] == 'Sevgilisi' ]

sevgilisitarafindanfiltre = sevgilisitarafindan[['year','killer1']]

sevgilisitarafindanfiltre.year.value_counts().head(13).sort_index()

listesevgilisi = list(sevgilisitarafindanfiltre.year.value_counts().head(13).sort_index())

#Verimizi Listeye Alalım

datamiz = []

datamiz.append(listeeskisevgilisi)

datamiz.append(listeeskikocasi)

datamiz.append(listesevgilisi)

datamiz.append(listekocasi)

# X axisindeki değerleri atayalım

barWidth = 0.22

r1 = np.arange(len(datamiz[0]))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

#plotu oluşturalım ve görüntüleyelim

plt.figure(figsize=(18,9))

plt.grid(zorder = 0)

plt.bar(r1, datamiz[0], color='#1089bc', width=barWidth, edgecolor='white', label='Eski Sevgilisi Tarafından')

plt.bar(r2, datamiz[1] , color='#ff3300', width=barWidth, edgecolor='white', label='Eski Kocası Tarafından')

plt.bar(r3, datamiz[2], color='#ffa500', width=barWidth, edgecolor='white', label='Sevgilisi Tarafından')

plt.bar(r4, datamiz[3], color='purple', width=barWidth, edgecolor='white', label='Kocası Tarafından')

plt.xlabel('Duygusal İlişki Kurulan İnsanlar Tarafından Öldürülen Kadınlar', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(datamiz[0]))], ['2008', '2009', '2010', '2011', '2012','2013', '2014', '2015', '2016', '2017','2018', '2019', '2020'])

plt.legend()

plt.show()

Bolgeler= pd.DataFrame({'AkdenizBolgesi': ['Adana', 'Antalya', 'Mersin' , 'Burdur' , 'Hatay' , 'Isparta' , 'Kahramanmaras' , 'Osmaniye', '','','','','','','','','',''], 

              'EgeBolgesi': ['Izmir', 'Mugla' , 'Aydin' , 'Afyonkarahisar' , 'Denizli', 'Kutahya' , 'Manisa', 'Usak','','','','','','','','','',''], 

              'MarmaraBolgesi' : ['Balikesir', 'Bilecik' , 'Bursa' , 'Canakkale', 'Edirne', 'Istanbul', 'Kirklareli', 'Kocaeli' , 'Izmit', 'Sakarya', 'Tekirdağ' ,'Yalova','','','','','',''],

              'DoguAnadoluBolgesi' : ['Agri', 'Ardahan', 'Bingol', 'Bitlis', 'Elazig', 'Erzincan' , 'Erzurum', 'Hakkari', 'Igdir', 'Kars', 'Malatya','Mus', 'Tunceli', 'Van','','','',''] ,

              'IcAnadoluBolgesi': ['Aksaray', 'Ankara', 'Cankiri', 'Eskisehir', 'Karaman', 'Kayseri', 'Kirikkale', 'Kirsehir', 'Konya', 'Nevsehir', 'Nigde', 'Sivas','Yozgat','','','','',''],

              'GuneydoguAnadoluBolgesi' : ['Adiyaman', 'Batman', 'Diyarbakir', 'Gaziantep', 'Mardin', 'Siirt', 'Sanliurfa', 'Urfa', 'Sirnak', 'Kilis','','','','','','','',''],

              'KaradenizBolgesi' : ['Trabzon' , 'Amasya', 'Artvin', 'Bartın' , 'Bayburt', 'Bolu', 'Corum', 'Duzce', 'Giresun', 'Gumushane', 'Karabuk', 'Kastamonu', 'Ordu', 'Rize', 'Samsun', 'Sinop', 'Tokat', 'Zonguldak'] })

akdenizcount = 0 

egecount = 0

marmaracount = 0

doguanadolucount=0 

icanadolucount = 0

guneydogucount=0

karadenizcount =0

nancount= 0

for i in range (0,2972):

    for j in range (0, 18):

    #if data.city == deneme.AkdenizBolgesi[i]:

        #count = count + 

        x = np.where((data.city[i] == Bolgeler.AkdenizBolgesi[j]), 1 ,0)

        if x :

            akdenizcount = akdenizcount +1

            break

        else :

            y = np.where((data.city[i] == Bolgeler.EgeBolgesi[j]), 1 , 0)

            if y : 

                egecount= egecount +1

                break

            else : 

                z = np.where((data.city[i] == Bolgeler.MarmaraBolgesi[j]), 1 ,0)

                if z : 

                    marmaracount= marmaracount +1

                    break

                else:

                    t = np.where((data.city[i] == Bolgeler.DoguAnadoluBolgesi[j]), 1 , 0)

                    if t :

                        doguanadolucount = doguanadolucount +1

                        break

                    else :    

                        w = np.where((data.city[i] == Bolgeler.IcAnadoluBolgesi[j]), 1 ,0)

                        if w : 

                            icanadolucount = icanadolucount + 1 

                            break

                        else:

                            r = np.where((data.city[i] == Bolgeler.GuneydoguAnadoluBolgesi[j]), 1 , 0)

                            if r :

                                guneydogucount = guneydogucount + 1 

                                break

                            else: 

                                v = np.where((data.city[i] == Bolgeler.KaradenizBolgesi[j]), 1 , 0)

                                if v: 

                                    karadenizcount = karadenizcount +1

                                    break

                                else:

                                    if( j == 17):

                                        nancount = nancount +1

bolgeyegoreliste = [akdenizcount,egecount,marmaracount,icanadolucount,doguanadolucount,guneydogucount,karadenizcount,nancount]

bolgeyegoreliste.sort()

bolgeyegoreliste = bolgeyegoreliste[::-1]

bolgeyegoreliste
data.city.value_counts().head(10)
counts = pd.Series([1294, 464, 273, 258, 243, 170, 164, 106],index = ['','','','','','','',''] )



explode = (0.075, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,0.025)

colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8', '#00AAAA',

          '#00C69C', '#00E28E', '#00FF80','#00FF80' ,]

index1 = ['Tespit Edilemeyen','Marmara','Akdeniz','Ege','İç Anadolu','Güneydoğu Anadolu','Karadeniz','Doğu Anadolu']

counts.plot(kind='pie', fontsize=14, colors=colors, explode=explode,figsize = (8,8), autopct = "%1.1f%%" ,startangle = 15 ,textprops = dict(color = 'w'))

plt.axis('equal')

plt.ylabel('')

plt.legend(labels=index1, loc="upper right")

plt.show()
degerler3 = data.city.value_counts().head(10)

colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8', '#00AAAA',

          '#00C69C', '#00E28E', '#00FF80','#00FF80' ,]

explode = (0.075, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,0.025, 0.025,0.025)

sehirler = degerler3.plot(kind='pie'  , textprops = dict(color = 'w'),figsize = (8, 8),colors = colors ,explode = explode, startangle = 20 , shadow = False , autopct = "%1.1f%%", fontsize=14)

sehirler.set_title("İlk 10 ile göre ölüm oranları")

sehirler.set_ylabel("" )

indexsehirler = ['Bilinemeyen','İstanbul','İzmir','Ankara','Bursa','Antalya','Adana','Konya','Gaziantep' , 'Mersin' ]

plt.axis('equal')

plt.legend(labels = indexsehirler , loc = 'upper right' , fontsize = 10)

plt.ylabel('')

plt.show()


counts = data.why1.value_counts().head(10)

explode = (0.075, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,0.025,0.025,0.025)

colors = ['#7a0100', '#d30000', '#ee2c2c', '#ff3030', '#ff4040', '#ff5656', '#ff6b6b',

          '#ff8484', '#ffaaaa', '#fcbfbf','#00FF80' ,]

index = ['Tespit Edilemeyen','Tartışma','Kıskançlık','Boşanma Talebi','Reddedilme','Para','Boşanma','Erkeğin istediği bir şeyin gerçekleşmemesi','Ayrılma Talebi','Namus']

counts.plot(kind='pie', fontsize=14, colors=colors, explode=explode,figsize = (10,8), autopct = "%1.1f%%" ,startangle = 20,textprops = dict(color = 'w') )

plt.axis('equal')

plt.ylabel('')

plt.legend(labels = index,loc = 'upper right')

plt.show()