from os import listdir

import matplotlib.pyplot as plt

from numpy import expand_dims, argmax, ravel



from tensorflow.keras import Input



from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.optimizers import RMSprop



from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# pengunduhan model - ukuran default model'e iku 9MB'an

pretrained_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False)



# jumlah label/kelas

CLASSES = 38



# nambah/ganti layer baru di output layernya

x = pretrained_model.output

x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.4)(x)



# jadikan layer yg terakhir itu pake fungsi aktivasi

# softmax biar dia bisa klasifikasi multi kelas

predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=(pretrained_model.input), outputs=predictions)



model.summary()
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
batch_size =32

base_dir = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"



# ketentuan2 praproses

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   validation_split=0.3,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



# praproses seluruh image pada tiap folder

train_generator = train_datagen.flow_from_directory(base_dir+'/train',

                                                    target_size=(256, 256),

                                                    batch_size=batch_size,

                                                    class_mode='categorical')



validation_generator = valid_datagen.flow_from_directory(base_dir+'/valid',

                                                         target_size=(256, 256),

                                                         batch_size=batch_size,

                                                         class_mode='categorical')
li = list(train_generator.class_indices.keys())



for i in li:

    print(i)
history = model.fit_generator(train_generator,

                              epochs=10, # dapat diganti bebas

                              steps_per_epoch=train_generator.samples//batch_size,

                              validation_data=validation_generator,

                              validation_steps=validation_generator.samples//batch_size)

# simpan model

model.save('coba.h5')
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



# akurasi

plt.plot(epochs, acc, label='Training Accuracy')

plt.plot(epochs, val_acc, label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()

plt.figure()



# loss

plt.plot(epochs, loss, label='Training Loss')

plt.plot(epochs, val_loss, label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
model_termuat = load_model('coba.h5')
test_data_nama = listdir('/kaggle/input/new-plant-diseases-dataset/test/test/')

pra_proses_citra = lambda x : [expand_dims(image.img_to_array(image.load_img('/kaggle/input/new-plant-diseases-dataset/test/test/'+i,

                               target_size=(256, 256))), axis=0)/255 for i in x]



test_data = pra_proses_citra(test_data_nama)
test_data_nama
hasil_prediksi = [argmax(model_termuat.predict(i)) for i in test_data]

ravel(hasil_prediksi)
for i in range(len(hasil_prediksi)):

    print("Hasil prediksi :", li[hasil_prediksi[i]],

          "\nNilai asli", test_data_nama[i], '\n')