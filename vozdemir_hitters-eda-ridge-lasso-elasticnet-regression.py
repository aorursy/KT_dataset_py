from warnings import filterwarnings
filterwarnings('ignore')
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso , LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
# Hitters veri setini df_base isimli dataframe'e atıyoruz.
hitters=pd.read_csv("../input/hittersds/Hitters.csv")
df_base=hitters.copy()
df_base.head()
# base dataframe'in kopyasını alıp, df_eda isimli dataframe'e atıyoruz. Head ile veri setine bakıyoruz.
df_eda = df_base.copy()
df_eda.head()
df_eda.tail()
# info metotu ile veri setimize bakıyoruz. 
# 20 değşkeni olan 322 adet gözlem mevcut veri setinde. 
# Değişkenlerin 3 tanesinin(League , Division, NewLeague) veri tipi object, 
# 1 tanesi(Salary) float 
# diğerleri ise int.
# Salary haricinde NaN degeri olan değişken yok. 
df_eda.info()
# Salary değişkeninde 59 adet NaN değer var.
df_eda.isnull().sum()
df_eda.dropna(inplace=True)
df_eda.describe().T
sns.distplot(df_eda.Salary,bins=50, kde = False);
# Değişkenlerin birbirleri ile olan korelasyonuna bakalım.
# CAtBat, CHits, CHmRun, CRuns ve CRBI değişkenlerinin salary ile korelasyonları diğer değişkenlere göre daha yüksek.
df_eda.corr()
# Şimdi CAtBat, CHits, CHmRun, CRuns ve CRBI değişkenlerin salary'e karşı grafiklerine bakalım. Doğrusal bir ilişki mevcut mu diye.
# Salary & CAtBat grafiğini çizdirelim. Grafiğe göre doğrusal bir ilişki olduğunu söyleyemeyiz.
sns.lmplot(x="Salary",y="CAtBat",data=df_eda);
# Salary & CHits grafiğini çizdirelim. Grafiğe göre doğrusal bir ilişki olduğunu söyleyemeyiz.
sns.lmplot(x="Salary",y="CHits",data=df_eda);
# Salary & CHmRun grafiğini çizdirelim. Grafiğe göre doğrusal bir ilişki olduğunu söyleyemeyiz.
sns.lmplot(x="Salary",y="CHmRun",data=df_eda);
# Salary & CRuns grafiğini çizdirelim. Grafiğe göre doğrusal bir ilişki olduğunu söyleyemeyiz.
sns.lmplot(x="Salary",y="CRuns",data=df_eda);
# Salary & CRBI grafiğini çizdirelim. Grafiğe göre doğrusal bir ilişki olduğunu söyleyemeyiz.
sns.lmplot(x="Salary",y="CRBI",data=df_eda);
# Years değişkenini de kullanıp, years bazında CRBI ile Salary arasında nasıl bir ilişki olduğunu inceleyelim.
#Years değişkenini kategorik bir değişkene dönüştürüp, Years Group adı ile dataframe ekliyoruz.
df_eda["Years Group"] = pd.cut(df_eda.Years,bins=[0,4,8,16,24],labels=['0-4','4-8','8-16','16-24'])
sns.lmplot(x="Salary",y="CRBI", hue="Years Group", data=df_eda);
# Years Group ile Salary arasında da net bir doğrusal ilişkiden bahsetmek zor. Evet yıl arttıkça ücretlerde artma meydana geliyor ama 16-24 aralığı 
# en yüksek ücreti almıyor. 
sns.catplot(x="Years Group", y="Salary", data=df_eda)
# CRBI değişkenini de kategorik bir değişken yapıp üç değişkene bakalım. 
df_eda["CRBI Group"] = pd.cut(df_eda.CRBI,bins=[0,95.0,230.0,424.5,1659.0],labels=['First_Q','Second_Q','Third_Q','High'])
# CRBI nin artması ile beraber Salary de artıyor ama net bir doğrusal ilişki mevcut değil.
sns.catplot(x="CRBI Group", y="Salary", hue="Years Group", data=df_eda, palette=["Blue","Red","Yellow","Black"])
# CRuns değişkeninede aynı işlemi uygulayalım.
df_eda["CRuns Group"] = pd.cut(df_eda.CRBI,bins=[0,105.5,250.0,497.5,2165.0],labels=['First_Q','Second_Q','Third_Q','High'])
# Sonuc CRBI ile aynı diyebiliriz.
sns.catplot(x="CRuns Group", y="Salary", hue="Years Group", data=df_eda, palette=["Blue","Red","Yellow","Black"])
# Daha önce kendi oluşturmuş olduğumuz kategorik değişkenleri drop edelim.
df_eda = df_eda.drop(['Years Group','CRBI Group','CRuns Group'], axis=1)
# Dataset içindeki değişken tiplerine bakalım
df_eda.dtypes
#Object tipleri kategorik yapacagız.
df_eda.League = pd.Categorical(df_eda.League)
df_eda.Division = pd.Categorical(df_eda.Division)
df_eda.NewLeague = pd.Categorical(df_eda.NewLeague)
df_eda.select_dtypes(include=["category"]).head()
# Unique degerleri buluyoruz.
print("League ",df_eda.League.unique())
print("Division ",df_eda.Division.unique())
print("NewLeague ",df_eda.NewLeague.unique())
# Kategorik değişkenlerin frekanslarına bakıyoruz.
# Legaue ve NewLeague değişkenleri arasında fark olduğunu gördük.
print("Division\n",df_eda.Division.value_counts())
print("League\n",df_eda.League.value_counts())
print("NewLeague\n",df_eda.NewLeague.value_counts())
#League ve NewLeague sayıları lig bazıda farklılık gösteriyor. Bazı oyuncular lig değiştirmişe benziyor. Bunlara bakalım.
# N liginden A ligine geçen oyunculara bakıyoruz.
df_eda[((df_eda["League"] == "N") & (df_eda["NewLeague"] == "A"))]
# A liginden N ligine geçen oyunculara bakıyoruz.
df_eda[((df_eda["League"] == "A") & (df_eda["NewLeague"] == "N"))]
# A ligi N ligine gore bir tık daha kaliteli bir lige benziyor.
df_eda.groupby("League").aggregate(["mean","min","max"])
# E divisionında ucretler W'e gore daha iyi
df_eda.groupby("Division")["Salary"].aggregate(["mean","min","max","median"])
# W division'ında salary değeri 0-500 olan daha fazla gözlem mevcut.
(sns
 .FacetGrid(df_eda, 
               hue="Division", 
               height=5,
               xlim=(0,3000))
 .map(sns.kdeplot,"Salary",shade=True)
 .add_legend() # cut kategorilerinin yazılmasını istiyoruz
);
df_model = df_base.copy()
df_model.isnull().sum()
df_model.groupby("Division")["Salary"].aggregate(["mean"])
# Nan salary degerlerini, division ve years grupları uzerinden dolduruyoruz. E divisionı salary degerleri W divisiondan daha yuksek. Ayrıca years'ında
# az da olsa salary ile bir ilişkisi mevcut. Bu sebeple iki değişkeni grupladım.
df_model['Salary'] = df_model.groupby(['Division','Years'])['Salary'].transform(lambda x: x.fillna(x.mean()))
# W division'ında dolmayan bir deeri ortalama ile dolduruyorum.
df_model["Salary"].fillna(450.876873, inplace=True)

df_model = pd.get_dummies(df_model ,columns=["League","Division","NewLeague"],drop_first=True)

# C ile başlayan değişkenler oyuncunun kariyeri boyunca yaptığı istatistiki değerlerdir. Oyuncuların beyzbol oyandıkları süre farklı olduğundan (Years) istatistiki
# veriler arasında büyük farklar bulunmaktadır. Bu sebeple yıllık ortalamalarını alarak her oyuncu için değişkenleri aynı ölçekte ifade ettik. 
df_model["AtBat Per Year"] = df_model["CAtBat"] / df_model["Years"]
df_model["Hits Per Year"] = df_model["CHits"] / df_model["Years"]
df_model["HmRun Per Year"] = df_model["CHmRun"] / df_model["Years"]
df_model["Runs Per Year"] = df_model["CRuns"] / df_model["Years"]
df_model["CRBI Per Year"] = df_model["CRBI"] / df_model["Years"]
df_model["Walks Per Year"] = df_model["CWalks"] / df_model["Years"]
# Yeni değişkenlerden sonra bunları drop ediyoruz.
df_model.drop(["CAtBat","CHits","CHmRun","CRuns","CRBI","CWalks"],axis=1, inplace=True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df_model)
outlier_scores = clf.negative_outlier_factor_
outlier_scores_sorted = np.sort(outlier_scores)
outlier_scores_sorted[0:50]
# Outlier skorlarımızı bulduk. Bu değerler içerisinden bir eşik değer belirlememiz gerekiyor. 
df_outlier_scores = pd.DataFrame(outlier_scores_sorted)
df_outlier_scores.plot(style=".-", stacked=True, yticks = [-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.0,-0.9], figsize=(15,20));
#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#sns.boxplot(df_outlier_scores)
esik_deger = outlier_scores_sorted[11]
esik_deger
esik_gozlem = df_model[outlier_scores == esik_deger]
aykirilar = df_model[outlier_scores < esik_deger]
esik_gozlem
# aykırı gozlemleri indexlerinden kurtarıp bir recarray içerisine attık.
res = aykirilar.to_records(index= False)
# tüm aykiri degerleri esik_gozlem ile ezdik.
res[:] = esik_gozlem.to_records(index=False)
# düzenlenen aykiri gozlemlerin oldugu res recarray'i aykirilar dataframe'inin indexleri ile beraber bir dataframe yapıp asıl dataframe değerlerimizi güncelledik.
df_model[outlier_scores < esik_deger] = pd.DataFrame(res, index = aykirilar.index)
df_model[outlier_scores < esik_deger]
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
X = df_model.drop('Salary', axis=1)
y = df_model[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse
np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
#test hatasi
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
X = df_model.drop('Salary', axis=1)
y = df_model[["Salary"]]

scaled_features = X[["AtBat","Hits","HmRun","Runs","RBI","Walks","Years","PutOuts","Assists","Errors","AtBat Per Year","Hits Per Year","HmRun Per Year","Runs Per Year","CRBI Per Year","Walks Per Year"]]
categoric_features = X[["League_N","Division_W","NewLeague_N"]]

cols = scaled_features.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scaled_features)
X_scaled = pd.DataFrame(X_scaled, columns = cols)

X_scaled = pd.concat([X_scaled, categoric_features], axis=1, join="inner")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.20, 
                                                    random_state=46)

ridge_model=Ridge().fit(X_train,y_train)
y_pred=ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
ridge_model.coef_
from sklearn.linear_model import RidgeCV
alphas1 = np.random.randint(0,1000,1000)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas2, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train, y_train)
ridge_cv.alpha_
ridge_tuned = Ridge(alpha = 0.005).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.linear_model import Lasso , LassoCV
from sklearn.preprocessing import StandardScaler
X = df_model.drop('Salary', axis=1)
y = df_model[["Salary"]]

scaled_features = X[["AtBat","Hits","HmRun","Runs","RBI","Walks","Years","PutOuts","Assists","Errors","AtBat Per Year","Hits Per Year","HmRun Per Year","Runs Per Year","CRBI Per Year","Walks Per Year"]]
categoric_features = X[["League_N","Division_W","NewLeague_N"]]

cols = scaled_features.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scaled_features)
X_scaled = pd.DataFrame(X_scaled, columns = cols)

X_scaled = pd.concat([X_scaled, categoric_features], axis=1, join="inner")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.20, 
                                                    random_state=46)

lasso_model = Lasso().fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lasso_cv_model = LassoCV(alphas = alphas2, cv = 10).fit(X_train, y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso(alpha = 3.0679536367065814).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV

X = df_model.drop('Salary', axis=1)
y = df_model[["Salary"]]

scaled_features = X[["AtBat","Hits","HmRun","Runs","RBI","Walks","Years","PutOuts","Assists","Errors","AtBat Per Year","Hits Per Year","HmRun Per Year","Runs Per Year","CRBI Per Year","Walks Per Year"]]
categoric_features = X[["League_N","Division_W","NewLeague_N"]]

cols = scaled_features.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scaled_features)
X_scaled = pd.DataFrame(X_scaled, columns = cols)

X_scaled = pd.concat([X_scaled, categoric_features], axis=1, join="inner")

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=46)

enet_model = ElasticNet().fit(X_train, y_train)
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
elasticNet_cv_model = ElasticNetCV(alphas = alphas2, cv = 10).fit(X_train, y_train)
elasticNet_cv_model.alpha_
elasticNet_tuned = ElasticNet(alpha = 0.10772173450159389).fit(X_train, y_train)
y_pred = elasticNet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))