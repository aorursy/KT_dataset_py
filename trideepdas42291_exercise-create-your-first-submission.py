!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.ex_tpu import *

step_1.check()
from petal_helper import *
import tensorflow_addons as tfa
# Detect TPU, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
ds_train = get_training_dataset()

ds_valid = get_validation_dataset()

ds_test = get_test_dataset()



print("Training:", ds_train)

print ("Validation:", ds_valid)

print("Test:", ds_test)
print("Number of classes: {}".format(len(CLASSES)))



print("First five classes, sorted alphabetically:")

for name in sorted(CLASSES)[:5]:

    print(name)



print ("Number of training images: {}".format(NUM_TRAINING_IMAGES))
print("Training data shapes:")

for image, label in ds_train.take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())
print("Test data shapes:")

for image, idnum in ds_test.take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
one_batch = next(iter(ds_train.unbatch().batch(20)))

display_batch_of_images(one_batch)
def convert(image, label):

  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

  return image, label
def augment(image, label):

    image, label = convert(image, label)

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_with_crop_or_pad(image, 518, 518)

    image = tf.image.random_crop(image, size=[128, 512, 512, 3])

    image = tf.image.random_brightness(image, max_delta=0.5)

    image = tf.image.random_flip_left_right(image)

    image = tfa.image.rotate(image, tf.constant(np.pi/8))

    image = tfa.image.transform(image, [1.0, 1.0, -250, 0.0, 1.0, 0.0, 0.0, 0.0])

    image = tf.image.random_saturation(image, 0, 2)

    return image, label
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

AUTOTUNE = tf.data.experimental.AUTOTUNE
augmented_train_batches=(

ds_train

    .map(augment, num_parallel_calls=AUTOTUNE)

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTOTUNE)    

)
validation_batches = (

ds_valid

    .map(convert, num_parallel_calls=AUTOTUNE)

    .cache()

    .batch(BATCH_SIZE)

    .prefetch(AUTOTUNE)

)
test_batches = (

ds_test

    .batch(BATCH_SIZE)

    .prefetch(AUTO)



)
with strategy.scope():

    pretrained_model = tf.keras.applications.Xception(

        weights='imagenet',

        include_top=False ,

        input_shape=[*IMAGE_SIZE, 3]

    )

    pretrained_model.trainable = False

    

    model = tf.keras.Sequential([

        # To a base pretrained on ImageNet to extract features from images...

        pretrained_model,

        # ... attach a new head to act as a classifier.

        #tf.keras.layers.Conv2D(512, [3,3], activation='relu'),

        #tf.keras.layers.MaxPooling2D(2,2),

        #tf.keras.layers.Conv2D(512, [3,3], activation='relu'),

        #tf.keras.layers.MaxPooling2D(2,2),

        #tf.keras.layers.Flatten(),

        #tf.keras.layers.Dense(1024, activation='relu'),

        #tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    model.compile(

        optimizer='adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['accuracy'],

    )



model.summary()
# Define the batch size. This will be 16 with TPU off and 128 with TPU on



# Define training epochs for committing/submitting. (TPU on)

EPOCHS = 35

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

)
display_training_curves(

    history.history['loss'],

    history.history['val_loss'],

    'loss',

    211,

)

display_training_curves(

    history.history['accuracy'],

    history.history['val_accuracy'],

    'accuracy',

    212,

)
cmdataset = get_validation_dataset(ordered=True)

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()



cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)



labels = range(len(CLASSES))

cmat = confusion_matrix(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

)

cmat = (cmat.T / cmat.sum(axis=1)).T # normalize
score = f1_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

precision = precision_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

recall = recall_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

display_confusion_matrix(cmat, score, precision, recall)
dataset = get_validation_dataset()

dataset = dataset.unbatch().batch(20)

batch = iter(dataset)
images, labels = next(batch)

probabilities = model.predict(images)

predictions = np.argmax(probabilities, axis=-1)

display_batch_of_images((images, labels), predictions)
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)
print('Generating submission.csv file...')



# Get image ids from test set and convert to integers

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



# Write the submission file

np.savetxt(

    'submission.csv',

    np.rec.fromarrays([test_ids, predictions]),

    fmt=['%s', '%d'],

    delimiter=',',

    header='id,label',

    comments='',

)



# Look at the first few predictions

!head submission.csv