# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()

#5 tane değeri inceleyelim.
percentage_people_below_poverty_level.info()

#datayı inceleme için kullanalım.
percentage_people_below_poverty_level.poverty_rate.value_counts()

#anlamsız birşeyleri kontroletmek için bunu kullanılım.
percentage_people_below_poverty_level['Geographic Area'].unique()

#farklı eyaletleri bul.
#Toplam eyalet saysını bulalım.

len(percentage_people_below_poverty_level['Geographic Area'].unique())
#Her bir eyaletin fakirlik oranı

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0, inplace=True)

#"-" değerini "0" değerine eşitledik.

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)

#object değerini sayısal bir değere çevirdik.

area_list=list(percentage_people_below_poverty_level['Geographic Area'].unique())

#farklı eyaletlerin listesini oluşturuyoruz.

area_poverty_ratio=[]

#sıralama yapmak için bu listeyi açtık.

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})

new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data["area_poverty_ratio"])

plt.xticks(rotation=90)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title(" Poverty Rate Given States")
percentage_people_below_poverty_level.poverty_rate.value_counts()

#değeri olmayanları 0 eşitledik.Değeri olan kalmışmı diye tekrardan kontrol ettik.
kill.head()
kill.name.value_counts()

#burda 49 tane ismi yazılmamış tk tk var
#name_count

#hangi ismin ne kadar kullanılmış onu inceledik.

#a

#b

#a+b

c=Counter(kill.name).most_common(5)

c

#deneme en çok kullanılan 5 isim ve soy isim
#Öldürülen isimlerin ve soy isimlerin en çok kullanılanları 15 tanesini bulacağız.

separate=kill.name[kill.name !='TK TK'].str.split()

#isimleri ayırıyoruz, separate listesini oluşturuyoruz.

a,b=zip(*separate)

name_list=a+b

#alt alta liste şeklinde ekliyoruz.

name_count=Counter(name_list)

#hangi isim ne kadar kullanılmış ona bakıyoruz.

most_common_names=name_count.most_common(15)

#most_common ile en çok kullanılanları buluyoruz.(parantez içi değer kadarını)

x,y=zip(*most_common_names)

x,y=list(x),list(y)

plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

#burda yeni olan palette verdiğim değer kadar birbirine yakın yoğunluğa göre renk veriyor.

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most Common 15 Name or Surname of Killed people')

percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()

#tanımsız - 197 tane değer bulduk.
percent_over_25_completed_highSchool.info()
#Eyaletlerdeki 25 yaşından büyüklerin lise mezunu olma oranları

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list=list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool=[]



for i in area_list:

    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    area_highschool_rate=sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)



data=pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})

new_index=(data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation=90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
#Eyaletlerdeki ırkları grafikleştirme

share_race_city.head()
share_race_city.info()
#hepsi object float çevirmem lazım kullanabilmek için

share_race_city.replace(['-'],0.0,inplace=True)

share_race_city.replace(['(X)'],0.0, inplace=True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']]=share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list=list(share_race_city['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]



for i in area_list:

    x=share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

    

    

f,ax=plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha=0.6,label='White')

sns.barplot(x=share_black,y=area_list,color='blue',alpha=0.6,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='orange',alpha=0.6,label='Native American')

sns.barplot(x=share_native_american,y=area_list,color='yellow',alpha=0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha=0.6,label='Hispanic')



ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Percentage of Races',ylabel='States',title="Percentage of State's Population According to Races")

sorted_data.head()
sorted_data2.head()
#Eyaletlerdeki liseden mezun olma oranı ile fakirlik oranını karşılaştırma

sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)



#Visualize



f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize=17, style='italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize=17, style='italic')

plt.xlabel('States',fontsize=16,color='blue')

plt.ylabel('Values',fontsize=16,color='blue')

plt.title('High School Graduete vs Poverty Rate',fontsize=20,color='Brown')

plt.grid()

data.head()
#EYALETLER ARASINDAKİ MEZUN OLMA VE FAKİRLİK ORANINI KARŞILAŞTIRMA YİNE YUKARDAKİ POINT PLOTLA AYNISI

g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind="kde",size=7)

plt.savefig('graph.png')

plt.show()

g=sns.jointplot(x='area_poverty_ratio',y='area_highschool_ratio',data=data,size=5,ratio=3,color='red')
#kill.head()

#kill.race.head()

kill.race.value_counts()
#Öldürülen insanların ırkların oranı

kill.race.dropna(inplace=True)

#tanımsız olanları çıkar

labels=kill.race.value_counts().index

colors=['grey','blue','red','yellow','lime','black']

explode=[0,0,0,0,0,0]

sizes=kill.race.value_counts().values



#visual

plt.figure(figsize=(8,8))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.2f%%')

#autopct virgülden sonra kaçtane göstermek için kullanılıyor.

plt.title('Killed People According to Races',color='blue',fontsize=15)
#FAKİRLİK ORANI VE LİSEDEN MEZUN ORANINI LM İLE KARŞILAŞTIRACAĞIZ

sns.lmplot(x='area_poverty_ratio',y='area_highschool_ratio',data=data)

plt.show()
data.head()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut=5)

plt.show()
pal=sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data,palette=pal, inner="points")

plt.show()
data.corr()

#1 çıksaydı doğru orantı derdik.

#correlation görseleştirme aracı
f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidth=.5,linecolor='r',fmt='.2f',ax=ax)

#annot içindeki sayılar çıksın diye 



plt.show()
kill.head()
kill.gender.unique()
#ÖLÜM ŞEKLİNE GÖRE SININFLANDIRMA YAPACAĞIZ (Ateş edilerek,Ateş edilerek ve Şok tabancası ile)

#Cinsiyetinide görmek istiyoruz.

#Yaşınıda görmek istiyorum

sns.boxplot(x='gender',y='age',hue='manner_of_death',data=kill,palette="PRGn")

plt.show()

#hue klaslarına ayır demek

#barplotla aynı konuyu ele alacağız.

sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=kill)

plt.show()
sns.pairplot(data)

plt.show()
#kill.head()

kill.manner_of_death.value_counts()
sns.countplot(kill.gender)

plt.title('Manner ıf death',color='r',fontsize=15)
armed=kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color='blue',fontsize=15)
above25=['above25'if i >=25 else 'below25' for i in kill.age]

df=pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.xlabel('Age of Killed People',color='Blue',Fontsize=15)
sns.countplot(data=kill,x='race')

plt.title('Race of killed people',color='blue',fontsize=15)

plt.show()
city
city=kill.city.value_counts()

plt.figure(figsize=(15,15))

sns.barplot(x=city[:15].index,y=city[:15].values)

plt.xticks(rotation=90)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)

plt.show()
sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)

plt.show()
kill.head()
sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'red',fontsize = 15)

plt.show()