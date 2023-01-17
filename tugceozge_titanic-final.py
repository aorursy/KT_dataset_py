# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#pandas kütüphanesini csv dosyasını okumak için kullandım.

#numpy kütüphanesini sayısal hesaplamalar için kullandım.

import pandas as pd

import numpy as np



#pandas dataframe'i oluşturdum ve dataTitanic değişkenine atadım:

dataTitanic = pd.read_csv("../input/train.csv")

#dataframe'in ilk 5 satırını yazdırdım:

dataTitanic.head()
dataTitanic.shape

#shape komutu, veri setindeki örneklerin satır/sütun saysısını verir.
dataTitanic_test = pd.read_csv("../input/test.csv")

#Transpozunu aldım:

dataTitanic_test.head().T

#Burada hedef değişkenimiz "Survived"
dataTitanic.info()

#info metodu veri seti ile ilgili bilgi verir.

#Her sütundaki toplam değerler, null/ null değil, veri türü, kullanılan bellek gibi bilgileri verir.

#Buradan eksik değerler olduğunu görebiliriz. 

#Mesela tüm sütunlar için 891 değer olması gerekirken; "Age" için 714, "Cabin" için 204 değer var.
dataTitanic.describe()

#describe metodu veri setindeki sayısal sütunlar ile ilgili isttiksel bilgi verir

#Sütunlarda eksik değerlerin olup olmadığını tablodan kontrol edebiliriz.

#MEsela burada "Age" özniteliğinin null olan değerleri bulunmaktadır.
#Eksik değerleri olan başka sütun olup olmadığını kontrol edelim:

dataTitanic.isnull().sum()
#Test veri setimize de bakalım:

dataTitanic_test.isnull().sum()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



pd.options.display.mpl_style = 'default'

dataTitanic.hist(bins=10,figsize=(9,7),grid=False)
g = sns.FacetGrid(dataTitanic, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple")
g = sns.FacetGrid(dataTitanic, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g = sns.FacetGrid(dataTitanic, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare')
#"Embarked" ile "Survived" özniteliklerinin birbirlerine göre durumlarına baktığımızda ise;

sns.factorplot(x = 'Embarked',y="Survived", data = dataTitanic,color="r")
#Yolcu sınıflarına göre kaç erkek yolcu kurtuldu, kaç kadın yolcu kurtuldu bakalım:

sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=dataTitanic, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Male", "Female"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Male and Female Survived by Passenger Class')
#Yaş dağılımına göre kurtulanların, ölenlerin bilgisine bakalım:

ax = sns.boxplot(x="Survived", y="Age", 

                data=dataTitanic)

ax = sns.stripplot(x="Survived", y="Age",

                   data=dataTitanic, jitter=True,

                   edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12)
#Öznitelikler arasındaki korelasyona bakalım:

corr=dataTitanic.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between features')
#Özniteliklerin hedef değişkenle, yani sonuçla olan korelasyonu ise şöyledir:

dataTitanic.corr()["Survived"]
# "Pclass", "Sex", "Age" ve "Embarked" özniteliklerinin birbirlerine göre durumu:

g = sns.factorplot(x="Age", y="Embarked",

                    hue="Sex", row="Pclass",

                    data=dataTitanic[dataTitanic.Embarked.notnull()],

                    orient="h", size=2, aspect=3.5, 

                   palette={'male':"purple", 'female':"blue"},

                    kind="violin", split=True, cut=0, bw=.2)
#"Embarked" sütununda hangi satırların boş olduğuna bakalım:

dataTitanic[dataTitanic['Embarked'].isnull()]
#"Fare" ve "Embarked" özniteliklerine göre "Pclass" özniteliğinin durumu:

sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=dataTitanic)
#Yukarıdaki grafikten yola çıkarak en fazla bindirme limanı Cherbourg;

#Bu nedenle de sonucu etkilememesi için boş olan "Embarked" satırlarını 'C' değeri ile doldurdum.

dataTitanic["Embarked"] = dataTitanic["Embarked"].fillna('C')
#test veri setimizdeki boş değerlere tekrar göz atalım:

dataTitanic_test.describe()
#"Fare" özniteliğinde boş olan değer var mı bakalım:

dataTitanic_test[dataTitanic_test['Fare'].isnull()]
#Üçüncü yolcu sınıfını paylaşan ve "S" sınıfından çıkmış olan yolcuların tüm ücretlerini ortancayı bularak "Fare" özniteliğindeki eksik değeri değiştirelim:

def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

#'S'

       #print(median_fare)

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



dataTitanic_test=fill_missing_fare(dataTitanic_test)
#"Cabin" bilgisi boş olanlara bakalım:

dataTitanic[dataTitanic['Cabin'].isnull()]
#"Cabin" değerleri

dataTitanic["Deck"]=dataTitanic.Cabin.str[0]

dataTitanic_test["Deck"]=dataTitanic_test.Cabin.str[0]

dataTitanic["Deck"].unique() # null değerler için 0
#"Cabin" özniteliğinde değerlere göre "Survived" bilgisine bakalım:

g = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=dataTitanic[dataTitanic.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8)
#"Age", "Deck" ve "Pclass" özniteliklerinin birbirlerine göre durumlarına bir bakalım:

dataTitanic = dataTitanic.assign(Deck=dataTitanic.Deck.astype(object)).sort("Deck")

g = sns.FacetGrid(dataTitanic, col="Pclass", sharex=False,

                  gridspec_kws={"width_ratios": [5, 3, 3]})

g.map(sns.boxplot, "Deck", "Age");
 # null değerler başarıyı düşürdüğü için Z ile doldurdum:

dataTitanic.Deck.fillna('Z', inplace=True)

dataTitanic_test.Deck.fillna('Z', inplace=True)

dataTitanic["Deck"].unique()
#Bir aile değişkeni oluşturalım:

#Bu değişken yolcunun kendisini, kardeş/eş sayısını("SibSp") ve ebeveyn/çocuk sayısını("Parch") verecek.

dataTitanic["FamilySize"] = dataTitanic["SibSp"] + dataTitanic["Parch"] + 1

dataTitanic_test["FamilySize"] = dataTitanic_test["SibSp"] + dataTitanic_test["Parch"] + 1

print(dataTitanic["FamilySize"].value_counts())
#Ailenin boyutunu ayrıklaştıralım 

#(Yani aile boyutunu tek kişi olanlar için singleton, küçük aileler için small ve büyük aileler için de large olarak belirtelim):

dataTitanic.loc[dataTitanic["FamilySize"] == 1, "FsizeD"] = 'singleton'

dataTitanic.loc[(dataTitanic["FamilySize"] > 1)  &  (dataTitanic["FamilySize"] < 5) , "FsizeD"] = 'small'

dataTitanic.loc[dataTitanic["FamilySize"] > 4, "FsizeD"] = 'large'



dataTitanic_test.loc[dataTitanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'

dataTitanic_test.loc[(dataTitanic_test["FamilySize"] > 1)  &  (dataTitanic_test["FamilySize"] < 5) , "FsizeD"] = 'small'

dataTitanic_test.loc[dataTitanic_test["FamilySize"] > 4, "FsizeD"] = 'large'



print(dataTitanic["FsizeD"].unique())

print(dataTitanic["FsizeD"].value_counts())
#Eklediğimiz feature'a göre "Survived"é" feature'ının durumuna bakalım:

sns.factorplot(x="FsizeD", y="Survived", data=dataTitanic)
#"Name" uzunluğu için özellik oluşturma

# .apply yöntemi ile yeni bir dizi oluşturdum:

dataTitanic["NameLength"] = dataTitanic["Name"].apply(lambda x: len(x))



dataTitanic_test["NameLength"] = dataTitanic_test["Name"].apply(lambda x: len(x))



#print(dataTitanic["NameLength"].value_counts())

'''

dataTitanic.loc[dataTitanic["NameLength"]>37 , "NlengthD"] = 'long'

dataTitanic.loc[dataTitanic["NameLength"]<38 , "NlengthD"] = 'short'



dataTitanic_test.loc[dataTitanic_test["NameLength"]>37 , "NlengthD"] = 'long'

dataTitanic_test.loc[dataTitanic_test["NameLength"]<38 , "NlengthD"] = 'short'

'''



bins = [0, 20, 40, 57, 85]

group_names = ['short', 'okay', 'good', 'long'] #Uzunluğa göre grulandırma türleri

dataTitanic['NlengthD'] = pd.cut(dataTitanic['NameLength'], bins, labels=group_names)

dataTitanic_test['NlengthD'] = pd.cut(dataTitanic_test['NameLength'], bins, labels=group_names)



sns.factorplot(x="NlengthD", y="Survived", data=dataTitanic)

print(dataTitanic["NlengthD"].unique())
import re



#Bir isimden ünvanın alınması:

def get_title(name):

    #Ünvan aramak için düzenli bir ifade kullandım. Ünvanlar/isimler her zaman büyük harfler ile küçük harflerden oluşur ve nokta ile biter.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #Eğer ünan varsa çıkar ve return et.

    if title_search:

        return title_search.group(1)

    return ""



#Tüm ünvanları aldım ve her birinin hangi sıklıkta tekrarlandığını yazdırdım.

titles = dataTitanic["Name"].apply(get_title)

print(pd.value_counts(titles))



'''

#Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare Title":5}

#,Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



#Verify that we converted everything.

print(pd.value_counts(titles))

'''

#Ünvan("Title") sütununa ekledim

dataTitanic["Title"] = titles



# Çok düşük hücre sayısına sahip ünvanları "rare"'da birleştirdim:

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Ek olarak 'Mlle', 'Ms' ve 'Mme' değerlerini buna göre yeniden atadım:

dataTitanic.loc[dataTitanic["Title"] == "Mlle", "Title"] = 'Miss'

dataTitanic.loc[dataTitanic["Title"] == "Ms", "Title"] = 'Miss'

dataTitanic.loc[dataTitanic["Title"] == "Mme", "Title"] = 'Mrs'

dataTitanic.loc[dataTitanic["Title"] == "Dona", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Lady", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Countess", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Capt", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Col", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Don", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Major", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Rev", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Sir", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Jonkheer", "Title"] = 'Rare Title'

dataTitanic.loc[dataTitanic["Title"] == "Dr", "Title"] = 'Rare Title'



#dataTitanic.loc[dataTitanic["Title"].isin(['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

#                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']), "Title"] = 'Rare Title'



#dataTitanic[dataTitanic['Title'].isin(['Dona', 'Lady', 'Countess'])]

#dataTitanic.query("Title in ('Dona', 'Lady', 'Countess')")



dataTitanic["Title"].value_counts()





titles = dataTitanic_test["Name"].apply(get_title)

print(pd.value_counts(titles))



#Ünvan("Title") sütununa ekledim

dataTitanic_test["Title"] = titles



# Çok düşük hücre sayısına sahip ünvanları "rare"'da birleştirdim:

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Ek olarak 'Mlle', 'Ms' ve 'Mme' değerlerini buna göre yeniden atadım:

dataTitanic_test.loc[dataTitanic_test["Title"] == "Mlle", "Title"] = 'Miss'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Ms", "Title"] = 'Miss'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Mme", "Title"] = 'Mrs'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Dona", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Lady", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Countess", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Capt", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Col", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Don", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Major", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Rev", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Sir", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

dataTitanic_test.loc[dataTitanic_test["Title"] == "Dr", "Title"] = 'Rare Title'



dataTitanic_test["Title"].value_counts()
#Eklediğimiz feature'larla birlikte veri setimize tekrardan göz atalım:



from sklearn import preprocessing

encoder=preprocessing.LabelEncoder()

cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']

for items in cat_vars:

    mask = ~dataTitanic[items].isnull()

    encoder.fit(dataTitanic[items][mask])

    dataTitanic[items]=encoder.transform(dataTitanic[items][mask])



for items in cat_vars:

    mask = ~dataTitanic_test[items].isnull()

    encoder.fit(dataTitanic_test[items][mask])

    dataTitanic_test[items]=encoder.transform(dataTitanic_test[items][mask])



    

dataTitanic.head()
#"Age" özniteliğimizi inceleyelim:

with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(dataTitanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="red")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")
#Random Forest kullanarak "Age" özniteliğindeki eksik verileri tahmin etme

from sklearn.ensemble import RandomForestRegressor



def fill_missing_age(df):

    

    #Öznitelik seti

    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass','FamilySize','FsizeD','NameLength',"NlengthD",'Deck']]

    # Seti train ve test olarak böldüm

    train  = age_df.loc[ (df.Age.notnull()) ] #bilinen "Age" değerleri

    test = age_df.loc[ (df.Age.isnull()) ] # null olan "Age" değerleri

    

    # Tüm yaş değerleri bir hedef dizisinde saklanır

    y = train.values[:, 0]

    

    # Diğer tüm değerler öznitelik diziliminde saklanır

    x = train.values[:, 1::]

    

    #Model oluşturma uygulama

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(x, y)

    

    # Eksik değerleri tahmin etmek için uygun modeli kullandım

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Bu tahminleri tüm veri setine atadım

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
dataTitanic_test.info()
dataTitanic=fill_missing_age(dataTitanic)

dataTitanic_test=fill_missing_age(dataTitanic_test)
dataTitanic_test.info()
#Tamamlanmış "Age" özniteliğine bakalım:

with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(dataTitanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")

    plt.xlim((15,100))
#Lineer regresyon sınıfını import edelim:

from sklearn.linear_model import LinearRegression

#Sklearn'ın çapraz geçerliliği kolaylaştıran bir yardımıcısı vardır:

from sklearn.cross_validation import KFold



#Hedefi tahmin etmek için kullanacağım sütunlar

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare", "Embarked","NlengthD", "FsizeD", "Title","Deck"]

#Algoritma sınıfını başlatalım:

alg = LinearRegression()



#Veri seti için çapraz doğrulama katları oluşturalım. Bu train ve test'e karşılık gelen sıralı indeksleri döndürür.

#random_state'i her çalıştırdığımızda aynı sonuçları elde ettiğimizden emin olmak için sabitledik, ayarladık.

kf = KFold(dataTitanic.shape[0], n_folds=3, random_state=1)



predictions = []
for train, test in kf:

    #"train" ile algoritmayı eğitiyorum.

    train_predictors = (dataTitanic[predictors].iloc[train,:])

    #Algoritmayı eğitmek için "Survived" özniteliğini kullanıyoruz:

    train_target = dataTitanic["Survived"].iloc[train]

    #"predictors" ve "target" kullanarak algoritmayı eğitiyoruz:

    alg.fit(train_predictors, train_target)

    #Şimdi test ile ilgili tahmin yapabiliriz:

    test_predictions = alg.predict(dataTitanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

#Sonuçlara tahminler koyalım(Sonuçlar sadece 1 ve 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0





accuracy=sum(dataTitanic["Survived"]==predictions)/len(dataTitanic["Survived"])

accuracy
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",

              "FsizeD", "Title","Deck"]



#Oluşturduğumuz lineer regresyon algoritmasını başlatalım:

alg = LogisticRegression(random_state=1)

#Tüm çapraz doğrulama katları için doğruluk değerini hesaplayalım: 



scores = cross_validation.cross_val_score(alg, dataTitanic[predictors], 

                                          dataTitanic["Survived"], cv=3)

# Her kat için bir doğruluk değeri hesaplandığı için; bu doğruluk değerlerinin ortalamasını alalım

print(scores.mean())
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(dataTitanic[['Age', 'Fare']])

df_std = std_scale.transform(dataTitanic[['Age', 'Fare']])



#std_scale = preprocessing.StandardScaler().fit(dataTitanic_test[['Age', 'Fare']])

#df_std = std_scale.transform(dataTitanic_test[['Age', 'Fare']])
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold

import numpy as np

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch","Fare", "Embarked","NlengthD", "FsizeD", "Title","Deck"]



#Oluşturduğumuz algortimayı default değerlerle başlatalım:

# n_estimators yapmak istediğimiz ağaç sayısı

# min_samples_split bölme işlemini yapmamız gereken minimum satır sayısı

# min_samples_leaf bir ağaç dalının bittiği yerde alabileceğimiz minimum örnek sayısı (ağacaın alt dalları)

rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = KFold(dataTitanic.shape[0], n_folds=3, random_state=1)



scores = cross_validation.cross_val_score(rf, dataTitanic[predictors], dataTitanic["Survived"], cv=kf)

# Her kat için bir tane hesaplandığı için; bu doğruluk değerlerinin ortalamasını alalım

print(scores.mean())
#accuracy değerini arttırmak için n_estimators, min_samples_split ve min_samples_leaf değerlerini değiştirelim:

rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)

kf = KFold(dataTitanic.shape[0], n_folds=3, random_state=1)

rf.fit(dataTitanic[predictors], dataTitanic["Survived"])

scores = cross_validation.cross_val_score(rf, dataTitanic[predictors], dataTitanic["Survived"], cv=kf)

# Her kat için bir tane hesaplandığı için; bu doğruluk değerlerinin ortalamasını alalım

print(scores.mean())
# Oluşturduğum Random Forest Modeli ile feature'ların önemine, etkisine bakalım:

importances=rf.feature_importances_

std = np.std([rf.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])



plt.figure()

plt.title("Feature Importances By Random Forest Model")

plt.bar(range(np.size(predictors)), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')



plt.xlim([-1, np.size(predictors)])

plt.show()
#Bu grafikten yola çıkarak öznitelik seçimi yapalım:

import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.cross_validation import KFold

%matplotlib inline

import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FsizeD", "Title", "NlengthD","Deck"]



# Öznitelik seçimi

selector = SelectKBest(f_classif, k=5)

selector.fit(dataTitanic[predictors], dataTitanic["Survived"])



#Her öznitelik için işlem yapılmamış p değerlerini alalım ve p değerlerinden skorlara dönüştürelim:

scores = -np.log10(selector.pvalues_)



indices = np.argsort(scores)[::-1]



sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])



# "Pclass", "Sex", "Title", ve "Fare" öznitelikleri en etkilileri mi görmek için skorları gösterelim:

plt.figure()

plt.title("Feature Importances")

plt.bar(range(np.size(predictors)), scores[indices],

       color="seagreen", yerr=std[indices], align="center")

plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')



plt.xlim([-1, np.size(predictors)])

plt.show()

#Sadece en iyi 5 özniteliği seçelim:

predictors = ["Pclass", "Sex", "Fare", "Title","Age"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=8,min_samples_split=6, min_samples_leaf=4)



#alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

kf = KFold(dataTitanic.shape[0], n_folds=3, random_state=1)



scores = cross_validation.cross_val_score(alg, dataTitanic[predictors], dataTitanic["Survived"], cv=kf)

#Skorların ortalamasını alalım:

print(scores.mean())
#Tüm tarin veri setini kullarak bir algoritma geliştirelim:

alg.fit(dataTitanic[predictors], dataTitanic["Survived"])

#Test veri setini kullanarak tahmin edelim. Bir hata olmasını engellemek için tüm sütunları float'a çevirelim:

predictions = alg.predict_proba(dataTitanic_test[predictors].astype(float))[:,1]

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0



predictions=predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": dataTitanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("dataTitanic_submission.csv", index=False)