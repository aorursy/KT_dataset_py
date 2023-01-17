import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator
IMG_SIZE = 256



TRAINING_DIR = "../input/xmrfishplus/images/train"

training_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(IMG_SIZE,IMG_SIZE), 

                                                       batch_size=64, class_mode='categorical' ,shuffle=True)





TEST_DIR = "../input/xmrfishplus/images/test"

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=(IMG_SIZE,IMG_SIZE) , batch_size=64 , shuffle=False)
import tensorflow_hub as hub



URL = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4' 

feature_extractor = hub.KerasLayer(URL, input_shape=(256,256,3))

feature_extractor.trainable = False
model = tf.keras.models.Sequential([    

    feature_extractor,

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(20,activation='softmax')

    

])

model.summary()
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self,epoch,logs={}):

    if(logs['accuracy']>=1):

      self.model.stop_training=True



callbacks=myCallback()

METRICS = [

        'accuracy',

        tf.keras.metrics.Precision(name='precision'),

        tf.keras.metrics.Recall(name='recall')

    ]

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=METRICS )



history = model.fit(train_generator , epochs=50 , callbacks=[callbacks] )
model.evaluate(test_generator)