import numpy as np

np.random.seed(42)

import pandas as pd



import cv2

from sklearn.model_selection import KFold



from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# Load data

train = pd.read_json("../input/train.json")

test = pd.read_json("../input/test.json")
# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_1']])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_2']])



X_train = np.concatenate([x_band1[:, :, :, np.newaxis],

                          x_band2[:, :, :, np.newaxis],

                          ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)



target_train=train['is_iceberg']



del train
# Test data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test['band_1']])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test['band_2']])



X_test = np.concatenate([x_band1[:, :, :, np.newaxis],

                         x_band2[:, :, :, np.newaxis],

                         ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)



id_test = test['id'].values



del test; del x_band1; del x_band2
# Define CNN Model Architecture (Kaggle can't access the weights file)

img_height = 224

img_width = 224

img_channels = 3

img_dim = (img_height, img_width, img_channels)



def inceptionv3(img_dim=img_dim):

    input_tensor = Input(shape=img_dim)

    base_model = InceptionV3(include_top=False,

                   weights='imagenet',

                   input_shape=img_dim)

    bn = BatchNormalization()(input_tensor)

    x = base_model(bn)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    

    return model



model = inceptionv3()

model.summary()
# Train Model and predict

def train_model(model, batch_size, epochs, img_size, x, y, test, n_fold, kf):

        

    train_scores = []; valid_scores = []

    preds_test = np.zeros(len(test), dtype = np.float)



    i = 1



    for train_index, test_index in kf.split(x):

        x_train = x[train_index]; x_valid = x[test_index]

        y_train = y[train_index]; y_valid = y[test_index]



        def augment(src, choice):

            if choice == 0:

                # Rotate 90

                src = np.rot90(src, 1)

            if choice == 1:

                # flip vertically

                src = np.flipud(src)

            if choice == 2:

                # Rotate 180

                src = np.rot90(src, 2)

            if choice == 3:

                # flip horizontally

                src = np.fliplr(src)

            if choice == 4:

                # Rotate 90 counter-clockwise

                src = np.rot90(src, 3)

            if choice == 5:

                # Rotate 180 and flip horizontally

                src = np.rot90(src, 2)

                src = np.fliplr(src)

            return src



        def train_generator():

            while True:

                for start in range(0, len(x_train), batch_size):

                    x_batch = []

                    end = min(start + batch_size, len(x_train))

                    y_batch = y_train[start:end]

                    for img in x_train[start:end]:

                        new_img = cv2.resize(img, img_size)

                        new_img = augment(new_img, np.random.randint(6))

                        x_batch.append(new_img)

                    x_batch = np.array(x_batch, np.float32) / 255.

                    y_batch = np.array(y_batch, np.uint8)

                    yield x_batch, y_batch



        def valid_generator():

            while True:

                for start in range(0, len(x_valid), batch_size):

                    x_batch = []

                    end = min(start + batch_size, len(x_valid))

                    y_batch = y_valid[start:end]

                    for img in x_valid[start:end]:

                        new_img = cv2.resize(img, img_size)

                        x_batch.append(new_img)

                    x_batch = np.array(x_batch, np.float32) / 255.

                    y_batch = np.array(y_batch, np.uint8)

                    yield x_batch, y_batch



        def test_generator():

            while True:

                for start in range(0, len(test), n_fold):

                    x_batch = []

                    end = min(start + n_fold, len(test))

                    for img in test[start:end]:

                        new_img = cv2.resize(img, img_size)

                        x_batch.append(new_img)

                    x_batch = np.array(x_batch, np.float32) / 255.

                    yield x_batch

                    

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),

             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 

                               verbose=1, min_lr=1e-7),

             ModelCheckpoint(filepath='inception.fold_' + str(i) + '.hdf5', verbose=1,

                             save_best_only=True, save_weights_only=True, mode='auto')]



        train_steps = len(x_train) / batch_size

        valid_steps = len(x_valid) / batch_size

        test_steps = len(test) / n_fold

        

        model = model



        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])



        model.fit_generator(train_generator(), train_steps, epochs=epochs, verbose=1, 

                            callbacks=callbacks, validation_data=valid_generator(), 

                            validation_steps=valid_steps)



        model.load_weights(filepath='inception.fold_' + str(i) + '.hdf5')



        

        print('----------------------------------------')

        print('Running train evaluation on fold {}'.format(i))

        train_score = model.evaluate_generator(train_generator(), steps=train_steps)        

        print('Running validation evaluation on fold {}'.format(i))

        valid_score = model.evaluate_generator(valid_generator(), steps=valid_steps)

        print('----------------------------------------')   

        

        print('Train loss: {:0.5f}\n Train acc: {:0.5f} for fold {}'.format(train_score[0],

                                                                            train_score[1], i))

        print('Valid loss: {:0.5f}\n Valid acc: {:0.5f} for fold {}'.format(valid_score[0],

                                                                            valid_score[1], i))

        print('----------------------------------------')



        train_scores.append(train_score[1])

        valid_scores.append(valid_score[1])

        print('Avg Train Acc: {:0.5f}\nAvg Valid Acc: {:0.5f} after {} folds'.format

              (np.mean(train_scores), np.mean(valid_scores), i))

        print('----------------------------------------')

        

        print('Running test predictions with fold {}'.format(i))        

        preds_test_fold = model.predict_generator(generator=test_generator(),

                                              steps=test_steps, verbose=1)[:, -1]



        preds_test += preds_test_fold



        print('\n\n')



        i += 1



        if i <= n_fold:

            print('Now beginning training for fold {}\n\n'.format(i))

        else:

            print('Finished training!')



    preds_test /= n_fold



    return preds_test
batch_size = 6

epochs = 50

n_fold = 3

img_size = (img_height, img_width)

kf = KFold(n_splits=n_fold, shuffle=True)



prediction = train_model(model, batch_size, epochs, img_size, X_train, 

                                target_train, X_test, n_fold, kf)



submit = pd.DataFrame({'id': id_test, 'is_iceberg': prediction.reshape((prediction.shape[0]))})

submit.to_csv('./submission.csv', index=False)