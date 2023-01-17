#Kütüphanelerimizi yüklüyoruz.
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import keras
from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
#Eğitim ve test verilerimizi aktarıyoruz.
(inp_train, out_train),(inp_test, out_test)=fashion_mnist.load_data()
# Eğitim ve test verilerinin boyutlarının görüntülüyoruz.
print('inp_train shape:', inp_train.shape)
print(inp_train.shape[0], 'eğitim örnekleri')
print(inp_test.shape[0], 'test örnekleri')

print(inp_train.shape, out_train.shape)
print(inp_test.shape, out_test.shape)
#Inp_ olarak tanımlanan değişkenlerimizin boyutlarını düzenliyoruz.
inp_train=inp_train.reshape(-1,28,28,1)
inp_test=inp_test.reshape(-1,28,28,1)
X_train = inp_train
X_test = inp_test
y_train = out_train
y_test = out_test
#Daha sonra ondalık hale çeviriyoruz.
inp_train=inp_train.astype('float32')
inp_test=inp_test.astype('float32')
#Modelimizin daha optimize çalışması için değerlerimizi 0 ile 1 arasına indirgiyoruz.
inp_train=inp_train/255.0
inp_test=inp_test/255.0
#Out_ olarak tanımlanan değişkenleri ise one-hot-encoding haline getiriyoruz.
out_train=to_categorical(out_train)
out_test=to_categorical(out_test)
#Eğitim setimizdeki bir örneği görüntülüyoruz.
image = X_train[4000].reshape([28, 28])

plt.subplot(2, 2, 1)
plt.imshow(image, cmap=cm.gray_r)
plt.axis('off')

plt.show()               
model=Sequential()
# 32 filtreli 2x2'lik kernel_size'lı relu aktivatörlü katman 
model.add(Conv2D(filters = 32,
                 kernel_size = (2,2),
                 padding = 'Same',
                 input_shape=(28,28,1),
                 activation='relu'))
# 32 filtreli 3x3'lük kernel_size'lı relu aktivatörlü katman 
model.add(Conv2D(filters = 32,
                 kernel_size = (3,3),
                 padding = 'Same',
                 activation='relu'))
# 2x2'lik Pooling katmanı
model.add(MaxPooling2D(pool_size=(2,2)))
# Nöron bağlarımızın dörtte birini temizlemek için kullandığımız Dropout katmanı
model.add(Dropout(0.25))
# 64 filtreli 4x4'lük kernel_size'lı relu aktivatörlü katman 
model.add(Conv2D(filters = 64, 
                 kernel_size = (4,4),
                 padding = 'Same',
                 activation='relu'))
# 64 filtreli 5x5'lik kernel_size'lı relu aktivatörlü katman 
model.add(Conv2D(filters = 64, 
                 kernel_size = (5,5),
                 padding = 'Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#Modeli compile edelim.
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Artık modeli eğitebiliriz.
egitim = model.fit(inp_train,out_train,batch_size=250,verbose=1,epochs=150,validation_split=0.2)
#Correction olarak oluşturduğumuz değişken ile modelimizin doğruluk oranını ölçelim.
correction=model.evaluate(inp_test.reshape(-1,28,28,1),out_test, verbose=1)
print('Yitim değeri (loss): {}'.format(correction[0]))
print('Test başarısı (accuracy): {}'.format(correction[1]))

# Doğruluk oranlarımızın epoch sayısına bağlı değişim grafiklerini görüntülüyoruz. 
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(egitim.history['acc'], color = 'pink')
axarr[0].plot(egitim.history['val_acc'], color = 'purple')
axarr[0].legend(['train', 'test'])
axarr[0].set_title('acc - val_acc')
axarr[1].plot(egitim.history['loss'], color = 'blue')
axarr[1].plot(egitim.history['val_loss'], color = 'gray')
axarr[1].legend(['train', 'test'])
axarr[1].set_title('loss - val_loss')
plt.show()
(inp_train, out_train), (inp_test, out_test) = fashion_mnist.load_data()
print('inp_train shape:', inp_train.shape)
print(inp_train.shape[0], 'eğitim örnekleri')
print(inp_test.shape[0], 'test örnekleri')

print(inp_train.shape, out_train.shape)
print(inp_test.shape, out_test.shape)
# fashion mnist etiket isimlerini ekliyoruz.
fashion_mnist_labels = np.array([
    'Tişört/Üst',
    'Pantolon',
    'Kazak',
    'Elbise',
    'Ceket',
    'Sandalet',
    'Gömlek',
    'Spor ayakkabı',
    'Çanta',
    'Bilekte Bot'])
# Test verilerinden rassal olarak seçilmiş 100 örneğin test ediyoruz ve görselleştiriyoruz.
def convertMnistData(image):
    img = image.astype('float32')
    img /= 255
    return image.reshape(1,28,28,1)

plt.figure(figsize=(16,16))

right = 0
mistake = 0
predictionNum = 100

for i in range(predictionNum):
    index = random.randint(0, inp_test.shape[0])
    image = inp_test[index]
    data = convertMnistData(image)

    plt.subplot(10, 10, i+1)
    plt.imshow(inp_test[index], cmap=cm.gray_r)
    plt.axis('off')

    ret = model.predict(data, batch_size=1) 
    
    bestnum = 0.0
    bestclass = 0
    for n in [0,1,2,3,4,5,6,7,8,9]:
        if bestnum < ret[0][n]:
            bestnum = ret[0][n]
            bestclass = n

    if out_test[index] == bestclass:
        plt.title(fashion_mnist_labels[bestclass])
        right += 1
    else:
        plt.title(fashion_mnist_labels[bestclass] + "!=" + fashion_mnist_labels[out_test[index]], color='#ff0000')
        mistake += 1
                                                                   
plt.show()
print("Doğru tahminlerin sayısı:", right)
print("Hata sayısı:", mistake)
print("Doğru tahmin oranı:", right/(mistake + right)*100, '%')
# Modelimizin karmaşıklık matrisini görüntülüyoruz.
pred = model.predict_classes(X_test)
cm=confusion_matrix(y_test,pred)

f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Blues",linecolor="green", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
