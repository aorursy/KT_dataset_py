# Setup
import numpy as np
import keras
from keras.applications import inception_v3, inception_resnet_v2, xception
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
PREDICT_DIR = "TestImages"
NUM_TRAIN, NUM_TEST, NUM_PREDICT = 34398, 3813, 16111
IMG_WIDTH, IMG_HEIGHT = 256, 256  # On hindsight, we realised these weren't the default image sizes to the keras pre-trained models
NB_CLASSES = 18
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = 'nearest',
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    rotation_range = 30)
test_datagen = ImageDataGenerator(
    rescale = 1./255)
predict_datagen = ImageDataGenerator(
    rescale = 1./255)
# Hyperparams
NUM_EPOCHS = 25
BATCH_SIZE = 54
FC_SIZE = 1024

xception_model = xception.Xception(include_top=False, weights='imagenet', pooling='avg', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
# Freeze the old layers
for layer in xception_model.layers:
    layer.trainable = False
# Attach our top layer classifier
x = xception_model.output
x = Dense(FC_SIZE, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(NB_CLASSES, activation='softmax')(x)
xception_final_model = Model(inputs=xception_model.input, outputs=pred)
xception_final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks and generators
checkpoint = ModelCheckpoint("xception.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")

# Start training
xception_final_model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
    epochs = NUM_EPOCHS,
    validation_data = test_generator,
    callbacks = [checkpoint, early])
# We had to reduce batch size due to GPU memory constraints
BATCH_SIZE = 18

for layer in xception_final_model.layers:
    layer.trainable = True
# We switch to SGD with low LR for transfer learning, so as to not wreck the previously learned weights
xception_final_model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# We also added a reduce LR callback, which may or may not have helped
reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.4, patience=4)
# Start fine tuning
xception_final_model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
    epochs = NUM_EPOCHS,
    validation_data = test_generator,
    callbacks = [checkpoint, early, reduceLR])
predict_generator = predict_datagen.flow_from_directory(
    PREDICT_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = "categorical",
    shuffle = False)
model1 = load_model("inception_v3_83.h5")
model2 = load_model("inception_v3_84.h5")
model3 = load_model("inception_v4_84.h5")
model4 = load_model("xception_84.h5")

results1 = model1.predict_generator(predict_generator, verbose=True)
results2 = model2.predict_generator(predict_generator, verbose=True)
results3 = model3.predict_generator(predict_generator, verbose=True)
results4 = model4.predict_generator(predict_generator, verbose=True)

ensemble = np.argmax(results1 + results2 + results3 + results4, axis = 1)
image_ids = [int(f[10:-4]) for f in predict_generator.filenames]

# Create the CSV
sub = pd.DataFrame({'id':image_ids, 'category':ensemble})
sub = sub.sort_values(by=['id'])
# Place id column first
sub = sub[['id', 'category']]
sub.to_csv("submission.csv", index=False)