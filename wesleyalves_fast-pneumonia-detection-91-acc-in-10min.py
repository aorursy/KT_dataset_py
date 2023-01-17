import os
import cv2
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import add
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(777)
tf.random.set_seed(777)
tf.__version__
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
BATCH_SIZE = 32
IMG_HEIGHT = 240
IMG_WIDTH = 240
ALPHA = 2e-4
labels = ['PNEUMONIA', 'NORMAL']
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) 
                resized_arr = cv2.resize(img_arr, (IMG_WIDTH, IMG_HEIGHT))
                data.append([resized_arr, class_num])
            except Exception as e:
                pass

    return np.array(data)
train = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
test = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
val = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
print(f"{[y for _, y in train].count(1)} PNEUMONIA IMAGES IN TRAINING SET")
print(f"{[y for _, y in train].count(0)} NORMAL IMAGES IN TRAINING SET")
print(f'Images in TRAINING SET: {train.shape[0]}')
print(f'Images in VALIDATION SET: {val.shape[0]}')
print(f'Images in TEST SET: {test.shape[0]}')
train = np.append(train, val, axis=0)
train, val = train_test_split(train, test_size=.20, random_state=777)
plt.figure(figsize=(10, 10))
for k, i in np.ndenumerate(np.random.randint(train.shape[0], size=9)):
    ax = plt.subplot(3, 3, k[0] + 1)
    plt.imshow(train[i][0], cmap='gray')
    plt.title(labels[train[i][1]])
    plt.axis("off")
def prepare_data(data):
    x = []
    y = []
    
    for feature, label in data:
        x.append(feature)
        y.append(label)
        
    x = (np.array(x) / 255).reshape(-1,IMG_WIDTH, IMG_HEIGHT, 1)
    y = np.array(y)
        
    return x, y

x_train, y_train = prepare_data(train)
x_val, y_val = prepare_data(val)
x_test, y_test = prepare_data(test)
datagen = ImageDataGenerator(
    rotation_range = 20, 
    zoom_range = 0.2, 
    width_shift_range=0.15,  
    height_shift_range=0.15,
    horizontal_flip = False,  
    vertical_flip=False)


datagen.fit(x_train)
weights = compute_class_weight('balanced', np.unique(y_train), y_train)
weights = {0: weights[0], 1: weights[1]}
print(weights)
def block(inputs, filters, stride):
    conv_0 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding='same', activation='relu')(inputs)
    conv_1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding='same', activation='relu')(conv_0)
    
    skip = layers.Conv2D(input_shape=input_size, filters=filters, kernel_size=(1, 1), strides=(stride**2, stride**2), padding='same', activation='relu')(inputs)
    
    pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(add([conv_1, skip]))
    
    return pool
input_size = (IMG_HEIGHT, IMG_WIDTH, 1)

inputs = tf.keras.Input(shape=input_size, name='input')

y_0 = block(inputs, 16, 2)
y_1 = block(y_0, 32, 1)
y_2 = block(y_1, 48, 1)
y_3 = block(y_2, 64, 1)
y_4 = block(y_3, 80, 1)

gap = layers.GlobalMaxPooling2D()(y_4)
dense = layers.Dense(2, activation='relu')(gap)

outputs = layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pneumonia_wnet")
model.summary()
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.7, min_delta=ALPHA, patience=7, verbose=1)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
CALLBACKS = [lr_reduce, early_stopping_cb]
METRICS = ['accuracy',
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall')]
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=ALPHA),
    loss='binary_crossentropy', 
    metrics=METRICS
)
history = model.fit(datagen.flow(x_train,y_train, batch_size = BATCH_SIZE),
                    steps_per_epoch=x_train.shape[0]/BATCH_SIZE, 
                    validation_data = (x_val, y_val),
                    validation_steps=x_val.shape[0]/BATCH_SIZE,
                    callbacks = CALLBACKS,
                    class_weight = weights,
                    epochs = 30)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
predictions = model.predict(x_test)
predictions = predictions.reshape(1,-1)[0]
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test,predictions)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

fig, ax = plot_confusion_matrix(conf_mat=cm , show_normed=True, figsize=(5, 5))
plt.show()