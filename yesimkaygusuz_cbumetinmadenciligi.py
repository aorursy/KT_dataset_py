import numpy as np # linear algebra
import pandas as pd # datayi CSV vb okutmak icin kullaniriz
import matplotlib.pyplot as plt

datapath = '../input/student-mat.csv' # dizini veriyolu olarak degiskene atadik 
dataset = pd.read_csv(datapath)
df = pd.DataFrame(dataset)
df




# Veri Keşfi ve Görselleştirme
#satır ve sütun sayısı
print(dataset.shape)

print("First 10 lines:")#ilk 10 satır --> deger belirtidliği ici, yoksa 5 satır
df.head(10)


print("Tail")#son 5 satır-->deger belirtilmediği icin 5
df.tail()


print("describe: ")#basit belirli istatistikler
df.describe()



print("info: ")#bellek kullanımı ve veri türleri
df.info()


#histogramı cizer
df.hist()
#histogramı cizer, parametrelerini degistirdik, korelasyonu en yüksek iki feature un hisyogramını cizdik
#hist = df.hist(bins=3)
ax = df.hist(column='G2', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

ax = ax[0]
for x in ax:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="True", which="True", bottom="False", top="False", labelbottom="True", left="True", right="False", labelleft="True")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("Histogram g2-g3")

    # Set x-axis label
    x.set_xlabel("G3", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("G2", labelpad=20, weight='bold', size=12)

    # Format y-axis label



#eksik veri yok, 395 de 395 hepsi dolu, "Ön İşleme" kısmında da göreceğiz
dataset.corr()#1 e yakın sonuc verenlerin korelasyonu yüksektir. 
#g1 g2 ve g2 g3 arasında, pozitif yönlü güçlü bir korelasyon görülüyor
#g2-g3 0.904868
#g1-g2 0.852118

import seaborn as sns

# Compute the correlation matrix
corr = dataset.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#en koyu olanların korelasyonları en yüksektir. g2 ve g3 ile g1 ve g2 burada da kendilerini göstermişler
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
import seaborn as sns#baska bir gösterim, bunda g1-g3 ile g2-g3 daha net
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df.plot(x='G2', y='G3', style='o') #g2-g3 korelasyon degeri: 0.904868
#net bir sekilde yüksek oldugu görülüyor
df.plot(x='G1', y='G2', style='o')#g1-g2 korelasyon degeri : 0.852118
#net bir sekilde yüksek oldugu görülüyor
# Ön İşleme
dataset.isnull().sum()#eksik veri olmadığı görülüyor hepsinde 0
dataset.isnull().sum().sum()#bu da toplam eksikleri veriyor. ikna olmazsak bunu deneyebiliriz.



#2.Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=df['G2'])

P = np.percentile(df.Medu, [10, 100])
P

#1 ve 4 uç değerler, bu iki değer arasındakiler uygun
new_df = df[(df.Medu > P[0]) & (df.Medu < P[1])]
new_df
#sns.boxplot(x=df['G3'])
#P = np.percentile(df.Medu, [10, 100])

#seklinde tüm featurelar icin uc degerleri bulabiliriz
#Yeni Öznitelik oluşturma--> 2 adet feature birlestirildi ve baska bir feature olarak atandı
df["NewFeature"] = df["Mjob"]+ df["Fjob"]
df

#Normalleştirme
from sklearn import preprocessing

#age özniteliğini normalleştirmek istiyoruz
x = df[['age']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['age2'] = pd.DataFrame(x_scaled)

df

print(dataset)
#Model Eğitimleri
#X = dataset.iloc[:, [2, 3]].values
#X = df.iloc[:, :-1].values son sürun haric tamamı
X = df.iloc[:, 30:32]#korelasyonları yüksek olan veriler feature olarak secildi
Y = df['G2'] #ismen sutun alma




X
Y
# Veri setini test ve eğitim olarak 2'ye ayırıyoruz. %25 e %75 seklinde
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest algoritmasını uyguluyoruz 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# X_test ile sonucu tahmin etmeye calısıyoruz
Y_pred = classifier.predict(X_test)

#Confusion matrisimizi oluşturuyoruz. Bu esnada Classification Raporunu da import ediyoruz. 
#Burada dogru ve yanlıs sayıları net bir sekilde gorunecek
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix:")
print(cm)

# Accuracy sonucu
from sklearn.metrics import accuracy_score
print("Accuracy:")
print("ACC: ",accuracy_score(Y_pred,Y_test))


print("Precision, Recall Değerleri:")
print(classification_report(Y_test, Y_pred))

#buradan gozlemlenen sonuclar oldukca kotu. bir de naive bayes i deneyelim.

#sonuc iyi görünüyor, bir de grafiksel olarak inceleyelim
# Eğitim sonuçları gözlemliyoruz
from matplotlib.colors import ListedColormap

X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Sınıflama (Eğitim seti)')
plt.xlabel('Feature')
plt.ylabel('G3')
plt.legend()
plt.show()

# Test sonuçlarını gözlemliyoruz.
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Sınıflama (Test seti)')
plt.xlabel('Feature')
plt.ylabel('G3')
plt.legend()
plt.show()
#2. MODEL ile eğitilmesi -Naive Bayes


# eğitim setine Naive Bayes uyguluyoruz 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Test veri setini kullanarak sonuçları tahmin ediyoruz
y_pred = classifier.predict(X_test)

# Confusion matrisimizi oluşturuyoruz.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


print("Confusion Matrix naive_bayes:")
print(cm)

# Accuracy sonucu
from sklearn.metrics import accuracy_score
print("naive_bayes-ACC: ",accuracy_score(Y_pred,Y_test))


print("Precision, Recall Değerleri: naive_bayes")
print(classification_report(Y_test, Y_pred))

#buradan gozlemlenen sonuclar oldukca kotu. bir de naive bayes i deneyelim.



# Eğitim sonuçları gözlemliyoruz
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Feature')
plt.ylabel('G3')
plt.legend()
plt.show()

# Test sonuçlarını gözlemliyoruz.
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Feature')
plt.ylabel(' G3')
plt.legend()
plt.show()