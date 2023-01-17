import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
!pip install -U efficientnet
# !mkdir ../input/data/images
# !mv ../input/data/images_001 ../input/data/images
# !mv ../input/data/images_002 ../input/data/images
# !mv ../input/data/images_003 ../input/data/images
# !mv ../input/data/images_004 ../input/data/images
# !mv ../input/data/images_005 ../input/data/images
# !mv ../input/data/images_006 ../input/data/images
# !mv ../input/data/images_007 ../input/data/images
# !mv ../input/data/images_008 ../input/data/images
# !mv ../input/data/images_009 ../input/data/images
# !mv ../input/data/images_010 ../input/data/images
# !mv ../input/data/images_011 ../input/data/images

all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join(os.getcwd(), "../input", 'data', "images*",'images', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)


all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
# print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('|', ','))

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.1, 
                                   random_state = 2137,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (256, 256)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)


train_gen=core_idg.flow_from_dataframe(
                        dataframe=train_df,
                        directory=os.path.join(os.getcwd(), "../input", 'data', "images_001",'images'),
                        x_col='Image Index',
                        y_col='Finding Labels',
                        batch_size=32,
                        color_mode = 'rgb',
                        class_mode='categorical',
                        classes=all_labels,
                        target_size=IMG_SIZE)

valid_gen=core_idg.flow_from_dataframe(
                        dataframe=valid_df,
                        directory=os.path.join(os.getcwd(), "../input", 'data', "images_002",'images'),
                        x_col='Image Index',
                        y_col='Finding Labels',
                        batch_size=32,
                        color_mode = 'rgb',
                        class_mode='categorical',
                        classes=all_labels,
                        target_size=IMG_SIZE)

test_X, test_Y = next(valid_gen)

from efficientnet.keras import EfficientNetB1
import tensorflow as tf

from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential


base_model = EfficientNetB1(input_shape =  (256,256,3), 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_model)
multi_disease_model.add(GlobalAveragePooling2D())
# multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(128))
multi_disease_model.add(BatchNormalization())
multi_disease_model.add(Activation('relu'))
multi_disease_model.add(Dense(128))
multi_disease_model.add(Activation('relu'))
# multi_disease_model.add(Dropout(0.5))

multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))

#metrics mae (need to be checked)
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy'
                            ,metrics = ['categorical_accuracy', "accuracy"])
# multi_disease_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path='simple_color_first_aproach-{epoch:02d}-{val_accuracy:.2f}.h5'

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15)
callbacks_list = [checkpoint, early]
multi_disease_model.load_weights(os.path.join(os.getcwd(), "../input", 'weights', "xray_class_weights.best.hdf5"))
pred_Y = multi_disease_model.predict(test_X, batch_size = 1, verbose = True)
for c_label, p_count, t_count in zip(all_labels, 
                                     100*np.mean(pred_Y,0), 
                                     100*np.mean(test_Y,0)):
    print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))
from sklearn.metrics import roc_curve, auc

avg = []

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    avg.append(auc(fpr, tpr))
    
print(sum(avg) / len(avg))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=50,
                                  validation_data = valid_gen, 
                                  validation_steps = 20,
                                  epochs = 50, 
                                  callbacks = callbacks_list)