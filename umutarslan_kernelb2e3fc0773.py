import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from keras import backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import metrics
df_train = pd.read_csv('../input/train.csv', index_col=0)
df_train.head()
#Hangi fiyatın satış fiyatıyla doğrusal bir ilişkisi olduguna bakıyoruz.
df_train['SalePrice'].describe()
#SalePrice'ın ağırlıklı aralığına distplot visualization ile göz attık.
sns.distplot(df_train['SalePrice']);
#Skewness ve Kurtosis ölçümlerine göz atıyoruz. Skewness verinin ortalamanın sağında veya solunda mı olduguna,

#kurtosis gauss dağılımının çok küçük, dar veya çok geniş olup olmadığını açıklar.
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#Hangi attribute'ın price ile en güçlü korelasyonları gösterdiğini görmek için heatmap kullanıyoruz.
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#Sale price ile ilgili en önemli 10 kolonu belirliyoruz.
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#şimdi ise missing datalarımızı görüntülüyoruz.
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#outlier verilerimizi atıyoruz ve veriyi daha çok standartlaştırıyoruz.

df_train = df_train.fillna(df_train.mean())
#dağılımın aralıgını tespit ediyoruz.
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('Düşük dağılımdaki ağırlığımız :')

print(low_range)

print('Yüksek Dağılımdaki ağırlığımız :')

print(high_range)
#burada ise garage live area attribute'ının saleprice üzerindeki korelasyonunu plot.scatter ile görüntülüyoruz.

#standart verilerden sapan 2 veri gözüküyor ancak bunları göz ardı edebiliriz. Çünkü ücret tarım alanı olmasına göre de

#değişiklik gösterebilir.
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,700000));
#hangi verilerin bizim için faydalı, hangi verilerin bizim için gereksiz oldugunu az çok gördük. 

#şimdi tekrardan başlayarak veriyi hazırlayalım.
df_train = pd.read_csv('../input/train.csv')
cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']

df_train = df_train[cols]

# Kukla verilerimizi olusturuyoruz.

df_train = pd.get_dummies(df_train)

#Ortalama degerler train dataframe'ine yerlestirilir.

df_train = df_train.fillna(df_train.mean())

scale = StandardScaler()

X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]

X_train = scale.fit_transform(X_train)

# Y SalePrice kolonumuzu tespit etmektedir.

y = df_train['SalePrice'].values

seed = 7

np.random.seed(seed)

# train datamızı 28'e 72 olarak böldük. Daha önceki denemelere göre en doğru dağıtım bu şekilde belirlenmiştir.

X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.28, random_state=seed)
def create_model():

    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(40, activation='relu'))

    model.add(Dense(1))

    # Compile model

    model.compile(optimizer ='adam', loss = 'mean_squared_error', 

              metrics =[metrics.mae])

    return model
model = create_model()

model.summary()
# NN olarak kurdugumuz modelimizi 100 epoch kadar eğitiyoruz.
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=32)
# modelimizin başarısını ölçmek adına mean absolute error ve accuracy'sini epoch sayısına göre gösteriyoruz.

# epoch sayımız arttıkça modelimiz gitgide öğreniyor. Ancak 100 epochtan sonra yavaşça overfitting durumuna girdiğimiz için

# öğrenmeyi durdurduk.
plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
df_test = pd.read_csv('../input/test.csv')

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']

id_col = df_test['Id'].values.tolist()

df_test['GrLivArea'] = np.log1p(df_test['GrLivArea'])

df_test = pd.get_dummies(df_test)

df_test = df_test.fillna(df_test.mean())

X_test = df_test[cols].values

# NN için her zaman kullanılan Scale yöntemi.

scale = StandardScaler()

X_test = scale.fit_transform(X_test)
prediction = model.predict(X_test)

submission = pd.DataFrame()

submission['Id'] = id_col

submission['SalePrice'] = prediction
submission.to_csv('submission.csv', index=False)