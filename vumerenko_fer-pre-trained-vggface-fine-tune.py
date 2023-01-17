import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import os
import gc

# TODO:
# 1. Use data augmentation
# 2. Fine-tune best model on data augmentation
# 3. Evaluate on test set
print(os.listdir("../input"))
data_fer = pd.read_csv('../input/fer2013/fer2013.csv')
data_fer.head()
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
idx_to_emotion_fer = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
X_fer_train, y_fer_train = np.rollaxis(data_fer[data_fer.Usage == "Training"][["pixels", "emotion"]].values, -1)
X_fer_train = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_train]).reshape((-1, 48, 48))
y_fer_train = y_fer_train.astype('int8')

X_fer_test_public, y_fer_test_public = np.rollaxis(data_fer[data_fer.Usage == "PublicTest"][["pixels", "emotion"]].values, -1)
X_fer_test_public = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_public]).reshape((-1, 48, 48))
y_fer_test_public = y_fer_test_public.astype('int8')

X_fer_test_private, y_fer_test_private = np.rollaxis(data_fer[data_fer.Usage == "PrivateTest"][["pixels", "emotion"]].values, -1)
X_fer_test_private = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_private]).reshape((-1, 48, 48))
y_fer_test_private = y_fer_test_private.astype('int8')
print(f"X_fer_train shape: {X_fer_train.shape}; y_fer_train shape: {y_fer_train.shape}")
print(f"X_fer_test_public shape: {X_fer_test_public.shape}; y_fer_test_public shape: {y_fer_test_public.shape}")
print(f"X_fer_test_private shape: {X_fer_test_private.shape}; y_fer_test_private shape: {y_fer_test_private.shape}")
class_counts = np.bincount(y_fer_train)
x_ticks = np.arange(len(class_counts))

plt.bar(x_ticks, class_counts)
plt.xticks(x_ticks, idx_to_emotion_fer.values())
plt.show()
plt.imshow(X_fer_train[10], interpolation='none', cmap='gray')
plt.title(idx_to_emotion_fer[y_fer_train[10]])
plt.show()
plt.imshow(X_fer_test_public[10], interpolation='none', cmap='gray')
plt.title(idx_to_emotion_fer[y_fer_test_public[10]])
plt.show()
plt.imshow(X_fer_test_private[10], interpolation='none', cmap='gray')
plt.title(idx_to_emotion_fer[y_fer_test_private[10]])
plt.show()
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
def one_hot(y):
    return to_categorical(y, 7)
def plot_history(history, metrics):
    fig, ax = plt.subplots(1, 1+len(metrics), figsize=(20, 5))
    ax[0].plot(history.history['loss'], label='Train loss')
    ax[0].plot(history.history['val_loss'], label='Validation loss')
    ax[0].legend()
        
    for i, metric in enumerate(metrics):
        ax[i+1].plot(history.history[metric], label='Train %s' % metric)
        ax[i+1].plot(history.history['val_%s' % metric], label='Validation %s' % metric)
        ax[i+1].legend()
    
    plt.show()

# TODO: delete sample history
# class History:
#     def __init__(self):
#         self.history = dict({
#             'categorical_accuracy': list(range(10)),
#             'val_categorical_accuracy': [-i*2 for i in range(10)],
#             'loss': list(range(10)),
#             'val_loss': [-i*2 for i in range(10)]
#         })

# plot_history(History(), metrics=['categorical_accuracy'])
!pip install git+https://github.com/rcmalli/keras-vggface.git
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
VGGFace(include_top = False, input_shape = (48,48,3),pooling = 'avg').summary()
X_train, y_train = X_fer_train.reshape((-1, 48, 48, 1)), one_hot(y_fer_train)
X_val, y_val = X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
# X_small_train, _, y_small_train, _ = train_test_split(X_train, y_train, train_size=0.9)

def create_normalize(mean, std):
    def normalize(X):
        return (X - mean) / std
    return normalize

# X_mean = X_train.mean(axis=0)
# X_std = X_train.std(axis=0)

# X_small_val, _, y_small_val, _ = train_test_split(X_val, y_val, train_size=0.9)

# normalize = create_normalize(X_mean, X_std)

# X_train_norm = normalize(X_train)
# X_val_norm = normalize(X_val)

# print(X_train_norm.shape, X_val_norm.shape)
# from keras.preprocessing.image import ImageDataGenerator

# gen = ImageDataGenerator(featurewise_center=True,
#                          samplewise_center=False,
#                          featurewise_std_normalization=True,
#                          samplewise_std_normalization=False,
#                          zca_whitening=False,
#                          brightness_range=(0.3, 0.8),
#                          horizontal_flip=True,
#                          vertical_flip=False,
#                          validation_split=0.1)

# gen.fit(X_train)
X_test, y_test = X_fer_test_private.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_private)

X_train_all = np.concatenate((X_train, X_val), axis=0)
y_train_all = np.concatenate((y_train, y_val), axis=0)

X_train_mix, X_val_mix, y_train_mix, y_val_mix = \
    train_test_split(X_train_all, y_train_all, test_size=0.1)

print(X_train_mix.shape, y_train_mix.shape, X_val_mix.shape, y_val_mix.shape)

X_mix_mean = X_train_mix.mean(axis=0)
X_mix_std = X_train_mix.std(axis=0)

normalize_mix = create_normalize(X_mix_mean, X_mix_std)
X_train_mix_norm = normalize_mix(X_train_mix)
X_val_mix_norm = normalize_mix(X_val_mix)

X_test_norm = normalize_mix(X_test)

print(X_train_mix_norm.shape, y_train_mix.shape, X_val_mix_norm.shape, y_val_mix.shape)
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.layers import Flatten, Dense, Input, Concatenate, Dropout, BatchNormalization, ReLU
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
# X_train_small, X_rest, y_train_small, y_rest = train_test_split(X_train_mix_norm, y_train_mix, train_size=0.2)
# X_val_small, _, y_val_small, _ = train_test_split(X_rest, y_rest, train_size=0.02)

# print(X_train_small.shape, X_val_small.shape)
def compose_model(feature_extractor, reg=0.0):
    conv_output = feature_extractor(img_conc)

    dense_1   = Dense(1024, kernel_regularizer=l2(reg))(conv_output)
    bn_1      = BatchNormalization()(dense_1)
    relu_1    = ReLU()(bn_1)
    dropout_1 = Dropout(0.5)(relu_1)
    
    dense_2   = Dense(1024, activation='relu', kernel_regularizer=l2(reg))(dropout_1)
    out       = Dense(7, activation='softmax')(dense_2)

    return Model(inputs=img_input, outputs=out)

def train(params):
    print('training with {} params'.format(params))
    
    vgg_features = VGGFace(weights='vggface', include_top=False, input_shape=(48,48,3), pooling='max')
    # for x in vggfeatures.layers[:-5]: # [:-5] [:-9]
    #     x.trainable = False
    model = compose_model(vgg_features, reg=params['reg'])

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=params['lr']), 
                  metrics=['categorical_accuracy'])

    batch_size = params['batch_size']

    cat_weights = class_weight.compute_class_weight(
        'balanced', np.unique(y_fer_train), y_fer_train)

    history = model.fit(
        X_train_mix_norm,
        y_train_mix,
        batch_size=batch_size,
        epochs=10,
        validation_data=(X_val_mix_norm, y_val_mix),
        class_weight=cat_weights)
    
    return model, history

params = {
    'reg': 0.5,
    'lr': 5e-5,
    'batch_size': 64
}

model, history = train(params)

# params = {
#     'reg': [0.2, 0.4, 0.6, 0.8],
#     'lr': [1e-3, 3e-4, 1e-4, 1e-5],
#     'batch_size': [64, 128, 256]
# }

# scores = []

# for reg in params['reg']:
#     for lr in params['lr']:
#         for batch_size in params['batch_size']:
#             config = {
#                 'reg': reg,
#                 'lr': lr,
#                 'batch_size': batch_size
#             }

#             score = train(config)
#             scores.append([score, config])
# max_score = 0
# best_config = None

# for pair in scores:
#     score, config = pair
#     if score > max_score:
#         max_score = score
#         best_config = config
        
# print(max_score, best_config)
plot_history(history, metrics=['categorical_accuracy'])
# TODO: visualize weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# X_test, y_test = X_fer_test_private.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_private)


def evaluate(model):
    labels = idx_to_emotion_fer.values()
    
    y_pred = model.predict(X_test_norm)

    y_true_cat = np.argmax(y_test, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true_cat, y_pred_cat)
    print(report)

    conf = confusion_matrix(y_true_cat, y_pred_cat)
    conf = conf / np.max(conf)

    _, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(conf, annot=True, cmap='YlGnBu', 
                     xticklabels=labels, 
                     yticklabels=labels)
    plt.show()
    
    return report


report = evaluate(model)
import json

model_json = model.to_json()
with open('model.json', 'w') as f:
    f.write(model_json)

with open('params.json', 'w') as f:
    f.write(json.dumps({
        'reg': 0.5,
        'lr': 5e-5,
        'batch_size': 64
    }))
    
with open('report.txt', 'w') as f:
    f.write(report)
    
model.save_weights('model.h5')
