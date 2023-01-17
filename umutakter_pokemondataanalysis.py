import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info() #Veri setinin içeriği hakkında biraz ön bilgi kazanalım.
data.count() # Veri setinde kaç adet gözlem ve değişken olduğunu yazdıralım.
data.isna().sum() #Eksik gözlemlerin kaç tane olduğuna bakalım.
data.head() #Veri setinin ilk 5 gözlemini görüntüleyelim.
data.tail() #Veri setinin son 5 gözlemini görüntüleyelim.
data.sample(10) #Veri setinden restgele 10 gözlem görüntüleyelim.
data["Type 1"].sort_values().unique() #Type 1 kolonunun benzersiz değerlerini görüntüleyelim ve sıralayalım.

data["Type 2"].sort_values().unique() #Type 2 kolonunun benzersiz değerlerini görüntüleyelim ve sıralayalım.
data.describe().T #Sadece sayısal verileri açıklar.
data = data.rename(columns = {"Type 1":"Type1","Type 2":"Type2","Sp. Atk":"SpAtk","Sp. Def":"SpDef"})
data
corr = data.corr()
corr
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(corr, annot=True, ax=ax)
plt.show()
data.plot(kind="scatter", x="Defense", y="SpDef", color="green")
plt.xlabel("Defense")
plt.xlabel("SpDef")
plt.title("Defense - SpDef Scatter Plot")
plt.show()
data.plot(kind="scatter", x="SpAtk", y="SpDef",color="red")
plt.xlabel("SpAtk")
plt.xlabel("SpDef")
plt.title("SpAtk - SpDef Scatter Plot")
plt.show()
data.groupby(["Type1"]).describe()["Attack"].sort_values('mean',ascending=False) #Pokemonların tiplerine göre ortalama attack değerlerini sıralayalım.
data.groupby(["Type1"]).describe()["Defense"].sort_values('mean',ascending=False)
data.groupby(["Type1"]).describe()["HP"].sort_values('mean',ascending=False)
data[(data["Type1"] == "Dragon")] #Dragon tipindeki pokemonları gözlemleyelim.
data[(data["Type1"] == "Fairy")] #Fairy tipindeki pokemonları gözlemleyelim.
#Veri setiyle daha kolay uğraşabilmek için biraz küçültelim ve 6 tip pokemon üzerinden ilerleyelim.
dragon = data[(data["Type1"] == "Dragon")]
rock = data[(data["Type1"] == "Rock")]
fire = data[(data["Type1"] == "Fire")]
water = data[(data["Type1"] == "Water")]
grass = data[(data["Type1"] == "Grass")]            
fairy = data[(data["Type1"] == "Fairy")]
smallData = pd.concat([dragon,rock,fire,water,grass,fairy])
sns.violinplot(x = "Type1", y = "Attack", data = smallData);
plt.scatter(dragon.Name, dragon.Attack, color="#ff6600", label="Dragon" )
plt.scatter(rock.Name, rock.Attack, color="#993300", label="Rock" )
plt.scatter(fire.Name, fire.Attack, color="#ff1a1a", label="Fire" )
plt.scatter(water.Name, water.Attack, color="#0066ff", label="Water" )
plt.scatter(grass.Name, grass.Attack, color="#33cc33", label="Grass" )
plt.scatter(fairy.Name, fairy.Attack, color="#ff0066", label="Fairy")

dicts={}
smallData2 = smallData.loc[:,"HP":"Speed"]
liste=list()
for each in smallData2.columns:
    smallData.groupby(["Type1"])[each].mean()
    for i in range(6):
        liste.append(smallData.groupby(["Type1"])[each].describe().iloc[i,1]-smallData2.describe().iloc[1,i])
    dicts[each] = liste.copy()
    liste.clear()
    
typeMean=pd.DataFrame(dicts)
typeMean.index=smallData.Type1.sort_values().unique().copy()
typeMean
typeMean = typeMean.T.describe()
typeMean
plt.plot(typeMean.columns, typeMean.iloc[1,:], color="#ff6600", label="Dragon" )
