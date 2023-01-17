import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import json
from fastai.vision import *
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, ELU, BatchNormalization, Flatten
from sklearn.decomposition import PCA
with open('/kaggle/input/ships-in-satellite-imagery/shipsnet.json') as f:
    loaded_file = json.load(f)
loaded_file['data'][0][:5] # что-то странное с тем, как передается картинка
def parse_list_image(data): # матрица с картинками передается плоской. нужно вернуть ей нужный формат
    return np.array(data, dtype=np.uint8).reshape((3,80,80)).T
from tqdm import trange
for i in trange(len(loaded_file['data'])): # возвращаю нужный формат
    loaded_file['data'][i] = parse_list_image(loaded_file['data'][i])
loaded_file['data'][0][0] # теперь это похоже на матрицу
plt.figure(figsize = (20, 20)) 

# вывожу то, как выглядит датасет
for i, j in enumerate(np.random.randint(0, len(loaded_file['data']), 20)):
    plt.subplot(5, 4, (i + 1))
    plt.imshow(loaded_file['data'][j])
    plt.title(loaded_file['labels'][j])
from collections import Counter

Counter(loaded_file['labels']) # 0 всего в 3 раза больше 1 => классы более-менее уравновешены
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X, y = shuffle(loaded_file['data'], loaded_file['labels'], random_state = 12) # сохраняю признаки и таргет
X_trn, X_tst, y_trn, y_test =  train_test_split(X, y, test_size = 0.1) # делю на трейн и тест
X_train, X_val, y_train, y_val = train_test_split(X_trn, y_trn, test_size = 0.1) # делю трейн на трейн и валидацию 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# функция аугментации для увеличения количества данных в датасете
datagen = ImageDataGenerator(
    horizontal_flip=True,
#     width_shift_range=0.1,
    vertical_flip=True,
#     height_shift_range=0.1
)
datagen.fit(X_train) # применение аугментации к трейну
# штука, которая позволяет обновлять сессия после каждого запуска (чтобы сетка обучалась сначала)
def reset_tf_session():
    
    curr_session = tf.compat.v1.get_default_session()
    if curr_session is not None:
        curr_session.close()
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4, 
                        allow_soft_placement=True
                       )
    config.gpu_options.allow_growth = True

    s = tf.compat.v1.InteractiveSession(config=config)
    tf.compat.v1.keras.backend.set_session(s)
    return s
pca = PCA(n_components=100)
values_train=np.asarray(X_train).reshape(-1,(80*80*3))
pca.fit(values_train)


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA')
plt.show()
NCOMPONENTS = 100

pca = PCA(n_components=NCOMPONENTS)

data_pca_train = pca.fit_transform(np.asarray(X_train).reshape(-1, (80*80*3)))
data_pca_val = pca.transform(np.asarray(X_val).reshape(-1, (80*80*3)))
data_pca_test = pca.transform(np.asarray(X_tst).reshape(-1, (80*80*3)))
pca_std = np.std(data_pca_train)
pca_std
labels_train = tf.keras.utils.to_categorical(y_train)
labels_test = tf.keras.utils.to_categorical(y_test)
labels_validation = tf.keras.utils.to_categorical(y_val)
def mlp_make_model(num_of_filters=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)):
    
    model = Sequential()
    model.add(Dense(num_of_filters, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.1))
    model.add(Dense(num_of_filters//2, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.1))
    model.add(Dense(num_of_filters//2, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax', kernel_regularizer=kernel_regularizer))
    
    return model
mlp_model = mlp_make_model()

mlp_model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

mlp_model.fit(
  data_pca_train,
  labels_train,
  validation_data=(data_pca_val, labels_validation),
  epochs=100,
  batch_size=200
)
mlp_pred = mlp_model.predict(data_pca_test)
baseline_model = log_loss(y_test, mlp_pred)
baseline_model # будет бейзлайном для следующей модели
mlp_classes = mlp_model.predict_classes(data_pca_test)

roc_auc_score(mlp_classes, y_test)
import seaborn as sn
from sklearn.metrics import confusion_matrix


conf_matrix = confusion_matrix(y_test, mlp_classes)
sn.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
X_train = np.array(X_train) / 255 - 0.5 # нормирую значения пикселей
X_val = np.array(X_val) / 255 - 0.5 # нормирую значения пикселей
X_test = np.array(X_tst) / 255 - 0.5 # нормирую значения пикселей
y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(y_val) # привожу к нужному типу данных
# Класс сохраняющий вид модели и веса
from tensorflow.keras.models import save_model, load_model

class ModelSaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, file_name):
        super().__init__()
        self.file_name = file_name
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        save_model(self.model, self.file_name.format(epoch))
        print('{}'.format(self.file_name.format(epoch)))
# функция, сохраняющая веса в файл
def load_from_file(model_filename, last_epoch):
    return load_model(model_filename.format(last_epoch))
# Придуманная архитектура (подбирала ее изначально ручками, т.е. сидела и смотрела на то, как меняется лосс)
#def make_model(initializer='lecun_uniform', num_of_filters = 256, num_of_layers=1):

#    model = Sequential()
 
#    model.add(Conv2D(filters=num_of_filters, padding='same',
#                     kernel_size=5,
#                     strides=2,
#                     input_shape=(80,80,3),
#                     kernel_initializer=initializer,
#                     activation='relu'))
#     model.add(BatchNormalization())
#    
#    for i in np.arange(num_of_layers):
#        model.add(Conv2D(filters=num_of_filters, padding='same', kernel_size=(5,5), kernel_initializer=initializer))
#         model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
#        model.add(Activation('relu'))
#         model.add(Dropout(0.1))
#    
#    model.add(Flatten())
#    model.add(Dense(num_of_filters, kernel_initializer=initializer))
#
#    model.add(Activation('relu'))
#     model.add(BatchNormalization())    
#    model.add(Dense(2, kernel_initializer=initializer))             
#    model.add(Activation("softmax"))
#    
#    return model
# Придуманная архитектура (подбирала ее изначально ручками, т.е. сидела и смотрела на то, как меняется лосс)
def make_model(initializer='he_normal', num_of_filters = 128):

    model = Sequential()
 
    model.add(Conv2D(filters=num_of_filters,
                     kernel_size=5,
                     strides=2,
                     input_shape=(80,80,3),
                     kernel_initializer=initializer,
                     activation='relu'))

    model.add(Conv2D(filters=num_of_filters,
                     kernel_size=3,
                     kernel_initializer=initializer,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=initializer, activation='relu'))
    model.add(Dense(2, kernel_initializer=initializer, activation='softmax')) 
    
    return model
# Функция для обучения модели
def train_gen_model(train_gen, val_data, make_model, initial_lr=1e-3, epochs=10, initial_epoch=0):
    model_filename = '\nweights_{0:02d}.hdf5'
    
    s = reset_tf_session() # заново запускает сессию
    model = make_model() # сохраняем модель
    model.compile(
        loss=tf.keras.losses.binary_crossentropy, # Функция потерь
        optimizer=tf.keras.optimizers.Adam(lr=initial_lr), # Оптимизатором выступаем градиентный спуск - Адам
        metrics=['accuracy'] # метрика
    )


    if initial_epoch != 0:
        model = load_from_file(model_filename, initial_epoch) # Загружаю веса в случае, когда надо дообучить модель
    
    hist = model.fit_generator(
        train_gen, #выше написан генератор 
        epochs=epochs,
        callbacks=[
                   ModelSaveCallback(model, model_filename),
                   tf.keras.callbacks.EarlyStopping( # Останавливается, если нет изменений в течение 3-х эпох
                      monitor='val_loss', patience=3)
                  ],
        validation_data=val_data, 
        shuffle=True,
        verbose=1,
        initial_epoch=initial_epoch
    )
    
     
    return hist, model
hist, model = train_gen_model(datagen.flow(X_train, y_train, batch_size=512),
                              val_data=(X_val, y_val),
                              make_model=make_model,
                              epochs=20, initial_lr=1e-3)
loss = hist.history['loss']

fig = plt.figure(0, figsize=(12,7), dpi=85)

plt.plot(range(1,len(loss) + 1),loss)

plt.xlabel('Номер итерации')
plt.ylabel('Loss')
plt.title('Зависимость функции потерь от номера итерации')
plt.show()
loss = hist.history['loss']

fig = plt.figure(0, figsize=(12,7), dpi=85)

plt.plot(range(4,len(loss) + 1),loss[3:])

plt.xlabel('Номер итерации')
plt.ylabel('Loss')
plt.title('Зависимость функции потерь от номера итерации')
plt.show()
pred_proba = model.predict(X_test) 
pred_proba[:10] #обхихикаться можно
pred_classes = model.predict_classes(X_test) 
roc_auc_score(y_test ,pred_classes) # работает на уровне константной
Counter(y_train) # Количество классов, на которых обучалась модель
log_loss(y_test, pred_proba) # работает хуже, чем mlp, лучше сделать у меня не получилось
import seaborn as sn
from sklearn.metrics import confusion_matrix


confusion_matrix = pd.crosstab(y_test, pred_classes, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix.values, annot=True, cmap="Blues")

plt.show()
new_baseline = log_loss(y_test, pred_proba)