import math, re, os

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import datetime

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
from tensorflow.keras.applications import Xception

from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2

from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.applications import InceptionResNetV2
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
!ls -l '/kaggle/input'
GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"

!gsutil ls -l $GCS_DS_PATH
# At the size 512, a GPU will run out of memory. Use the TPU.

# We use the size 192 here in order to save time.

IMAGE_SIZE = [512, 512] 



EPOCHS = 40

WARMUP_EPOCHS = 8

BATCH_SIZE = 8 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()



def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap='Reds')

    ax.set_xticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    ax.set_yticks(range(len(CLASSES)))

    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    titlestring = ""

    if score is not None:

        titlestring += 'f1 = {:.3f} '.format(score)

    if precision is not None:

        titlestring += '\nprecision = {:.3f} '.format(precision)

    if recall is not None:

        titlestring += '\nrecall = {:.3f} '.format(recall)

    if len(titlestring) > 0:

        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

    plt.show()

    

def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    # dataset = dataset.repeat() # Since we use custom training loop, we don't need to use repeat() here.

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
# data dump

print("Training data shapes:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())

print("Validation data shapes:")

for image, label in get_validation_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("Test data shapes:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
# run this cell again for next set of images

display_batch_of_images(next(train_batch))
# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
# run this cell again for next set of images

display_batch_of_images(next(test_batch))
# Enable this if you want to use `EfficientNetB7` model



# !pip install -q efficientnet

# from efficientnet.tfkeras import EfficientNetB7
class Flower_Classifier(tf.keras.models.Model):

    

    def __init__(self):



        super(Flower_Classifier, self).__init__()

        

        self.image_embedding_layers = []

        

        self.image_embedding_layers.append(Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3)))

        self.image_embedding_layers.append(ResNet152V2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3)))

        self.image_embedding_layers.append(InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3)))

        

        # self.image_embedding_layers.append(DenseNet201(weights='imagenet', include_top=False ,input_shape=(*IMAGE_SIZE, 3)))

        # self.image_embedding_layers.append(EfficientNetB7(weights='imagenet', include_top=False ,input_shape=(*IMAGE_SIZE, 3)))

        

        self.pooling_layer = tf.keras.layers.GlobalAveragePooling2D()



        self.layer_normalization_layers = []

        self.prob_dist_layers = []

        for model_idx, image_embedding_layer in enumerate(self.image_embedding_layers):

            

            self.layer_normalization_layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

            

            self.prob_dist_layers.append(

                tf.keras.layers.Dense(

                    len(CLASSES),

                    activation='softmax',

                    name='prob_dist_{}'.format(model_idx)

                )

            )

            

        # These values are obtained by previous training.

        kernel_init = tf.constant_initializer(np.array([0.86690587, 1.0948032, 1.1121726])) 

        bias_init = tf.constant_initializer(np.array([-0.13309559, 0.09480964, 0.11218266]))            

            

        self.prob_dist_weight = tf.keras.layers.Dense(

            len(self.image_embedding_layers), activation="softmax",

            kernel_initializer=kernel_init,

            bias_initializer=bias_init,

            name='prob_dist_weight'

        )



    def call(self, inputs, training=False):

        

        all_model_outputs = []

        for i in range(len(self.image_embedding_layers)):

            

            embedding = self.image_embedding_layers[i](inputs, training=training)

            pooling = self.pooling_layer(embedding, training=training)



            pooling_normalized = self.layer_normalization_layers[i](pooling, training=training)

            

            # shape = (batch_size, nb_classes)

            model_output = self.prob_dist_layers[i](pooling_normalized, training=training)



            all_model_outputs.append(model_output)

               

        # stack the outputs from different models

        # shape = (batch_size, nb_models, nb_classes)

        all_model_outputs = tf.stack(all_model_outputs, axis=1)

        

        # Get the model's current prob_dist_weights

        # shape = (1, nb_models)

        prob_dist_weight = self.prob_dist_weight(tf.constant(1, shape=(1, 1)), training=training)



        # Get the weighted prob_dist

        # shape = (batch_size, 1, nb_classes)

        prob_dist = tf.linalg.matmul(prob_dist_weight, all_model_outputs)

        

        # Remove axis 1

        # shape = (batch_size, nb_classess)

        prob_dist = prob_dist[:, 0, :]

        

        return prob_dist





with strategy.scope():

    

    flower_classifier = Flower_Classifier()
class CustomExponentialDecaySchedule(tf.keras.optimizers.schedules.ExponentialDecay):

    

    def __init__(self,

      initial_learning_rate,

      decay_steps,

      decay_rate,

      staircase=False,

      cycle=False,

      name=None,        

      num_warmup_steps=1000):

        

        # Since we have a custom __call__() method, we pass cycle=False when calling `super().__init__()` and

        # in self.__call__(), we simply do `step = step % self.decay_steps` to have cyclic behavior.

        super(CustomExponentialDecaySchedule, self).__init__(initial_learning_rate, decay_steps - num_warmup_steps, decay_rate, staircase, name=name)

        

        self.num_warmup_steps = num_warmup_steps

        

        self.cycle = tf.constant(cycle, dtype=tf.bool)

        

    def __call__(self, step):

        """ `step` is actually the step index, starting at 0.

        """

        

        # For cyclic behavior

        step = tf.cond(self.cycle and step >= self.decay_steps, lambda: step % self.decay_steps, lambda: step)

        

        # learning_rate = super(CustomExponentialDecaySchedule, self).__call__(step, war)



        # Copy (including the comments) from original bert optimizer with minor change.

        # Ref: https://github.com/google-research/bert/blob/master/optimization.py#L25

        

        # Implements linear warmup: if global_step < num_warmup_steps, the

        # learning rate will be `global_step / num_warmup_steps * init_lr`.

        if self.num_warmup_steps > 0:

            

            steps_int = tf.cast(step, tf.int32)

            warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)



            steps_float = tf.cast(steps_int, tf.float32)

            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)



            # The first training step has index (`step`) 0.

            # The original code use `steps_float / warmup_steps_float`, which gives `warmup_percent_done` being 0,

            # and causing `learning_rate` = 0, which is undesired.

            # For this reason, we use `(steps_float + 1) / warmup_steps_float`.

            # At `step = warmup_steps_float - 1`, i.e , at the `warmup_steps_float`-th step, 

            #`learning_rate` is `self.initial_learning_rate`.

            warmup_percent_done = (steps_float + 1) / warmup_steps_float

            

            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done



            is_warmup = tf.cast(steps_int < warmup_steps_int, tf.float32)



            learning_rate = super(CustomExponentialDecaySchedule, self).__call__(steps_int - warmup_steps_int)



            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

                        

        return learning_rate

    



with strategy.scope():

    

    # number of training steps

    decay_steps = int(NUM_TRAINING_IMAGES * EPOCHS / BATCH_SIZE)



    num_warmup_steps = int(NUM_TRAINING_IMAGES * WARMUP_EPOCHS / BATCH_SIZE)

    

    print("decay_steps = {}".format(decay_steps))

    print("num_warmup_steps = {}".format(num_warmup_steps))

    

    learning_rate = CustomExponentialDecaySchedule(

        initial_learning_rate=1e-4,

        decay_steps=decay_steps,

        decay_rate=0.1,

        cycle=False,

        name=None,

        num_warmup_steps=num_warmup_steps

)



%matplotlib inline

if decay_steps <= 20000:

    xs = tf.range(decay_steps)

    ys = [learning_rate(x) for x in xs]

    plt.plot(xs, ys)

    plt.ylabel("Learning Rate")

    plt.xlabel("Train Step")

    plt.show()
with strategy.scope():

    

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
from tensorflow.python.ops import clip_ops

from tensorflow.python.framework import constant_op





def _constant_to_tensor(x, dtype):

    return constant_op.constant(x, dtype=dtype)





with strategy.scope():



    num_classes = len(CLASSES)

    

    # About why we set `reduction` to 'none', please check this tutorial

    # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function

    # In particular, read the paragraph

    #     <<If using tf.keras.losses classes (as in the example below), the loss reduction needs to be explicitly specified to be one of NONE or SUM. AUTO and SUM_OVER_BATCH_SIZE are disallowed when used with tf.distribute.Strategy.>>

    

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none', label_smoothing=0.1)



    def loss_function(labels, prob_dists, sample_weights=None):



        if not sample_weights:

            sample_weights = 1.0



        # While trained with BATCH_SIZE = 8 * strategy.num_replicas_in_sync, I got `nan` values.

        # Since we pass probability distribution to `CategoricalCrossentropy` with `from_logits` = False,

        # which has numerical unstability issue,

        # we use the same trick in the source code to avoid such unstabiltiy.

        epsilon_ = _constant_to_tensor(tf.keras.backend.epsilon(), prob_dists.dtype.base_dtype)

        prob_dists = clip_ops.clip_by_value(prob_dists, epsilon_, 1 - epsilon_)

        

        labels = tf.keras.backend.one_hot(labels, num_classes)

        

        loss = loss_object(labels, prob_dists)

        

        # About why we use `tf.nn.compute_average_loss`, please check this tutorial

        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function

        

        loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)



        return loss
def get_metrics(name):



    loss = tf.keras.metrics.Mean(name=f'{name}_loss')

    acc = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc')

    

    return loss, acc





with strategy.scope():

    

    train_loss_obj, train_acc_obj = get_metrics("train")

    valid_loss_obj, valid_acc_obj = get_metrics("valid")
train_input_signature = [

    tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),

    tf.TensorSpec(shape=(None,), dtype=tf.int32)

]



# They have the same input format

valid_input_signature = train_input_signature



test_input_signature = [

    tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)

]



with strategy.scope():



    @tf.function(input_signature=train_input_signature)

    def train_step(images, labels):



        with tf.GradientTape() as tape:



            prob_dists = flower_classifier(images, training=True)

            loss = loss_function(labels, prob_dists)

            train_acc_obj(labels, prob_dists)



        gradients = tape.gradient(loss, flower_classifier.trainable_variables)

        

        gradients, global_norm = tf.clip_by_global_norm(

            gradients,

            clip_norm=1.0

        )

        

        optimizer.apply_gradients(zip(gradients, flower_classifier.trainable_variables))



        return loss



    @tf.function

    def distributed_train_step(inputs):



        (images, labels) = inputs

        loss = strategy.experimental_run_v2(train_step, args=(images, labels))

        

        return loss



    @tf.function(input_signature=valid_input_signature)

    def valid_step(images, labels):



        prob_dists = flower_classifier(images, training=False)

        loss = loss_function(labels, prob_dists, sample_weights=None)

        valid_acc_obj(labels, prob_dists)

        

        return loss, prob_dists





    @tf.function

    def distributed_valid_step(inputs):



        (images, labels) = inputs

        loss, prob_dists = strategy.experimental_run_v2(valid_step, args=(images, labels))

     

        return loss, prob_dists



    

    @tf.function(input_signature=test_input_signature)

    def test_step(images):



        prob_dists = flower_classifier(images, training=False)



        return prob_dists



    

    @tf.function

    def distributed_test_step(inputs):



        images = inputs

        prob_dists = strategy.experimental_run_v2(test_step, args=(images,))



        return prob_dists
validation_dataset = get_validation_dataset(ordered=True)

validation_dist_dataset = strategy.experimental_distribute_dataset(validation_dataset)



test_dataset = get_test_dataset(ordered=True)

test_image_dataset = test_dataset.map(lambda image, idnum: image)

test_dist_dataset = strategy.experimental_distribute_dataset(test_image_dataset)
labels_ds = validation_dataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch



# Save correct labels

with open("correct_labels", "w", encoding="UTF-8") as fp:

    np.savetxt(fp, cm_correct_labels, delimiter=',', fmt='%10.6f')
history = {

    "train_loss": [],

    "valid_loss": [],

    "train_acc": [],

    "valid_acc": [],

    "valid_f1": [],

    "valid_precision": [],

    "valid_recall": [],

    "model_coefs": []

}



print("start training at {}".format(datetime.datetime.now()))



for epoch in range(EPOCHS):

    

    epoch_start_time = datetime.datetime.now()



    # We need to shuffle the training dataset in every epoch.

    # I don't know if there is a better way than the following way.

    training_dataset = get_training_dataset()

    training_dist_dataset = strategy.experimental_distribute_dataset(training_dataset)

    

    train_loss_obj.reset_states()

    train_acc_obj.reset_states()

    valid_loss_obj.reset_states()

    valid_acc_obj.reset_states()

    

    for batch_idx, inputs in enumerate(training_dist_dataset):

                

        batch_start_time = datetime.datetime.now()

        

        # See explaination below in validation part.

        per_replica_train_loss = distributed_train_step(inputs)

        train_loss = tf.stack(per_replica_train_loss.values, axis=0)

        train_loss = tf.math.reduce_sum(train_loss)

        train_loss_obj(train_loss)        

        

        batch_end_time = datetime.datetime.now()

        batch_elapsed_time = (batch_end_time - batch_start_time).total_seconds()

        

        if (batch_idx + 1) % 50 == 0:

            

            # print training results

            print('Epoch {} | Batch {} | Timing {}'.format(epoch + 1, batch_idx + 1, batch_elapsed_time))

            print('Train Loss: {:.6f}'.format(train_loss_obj.result()))

            print('Train  Acc: {:.6f}'.format(train_acc_obj.result()))

            print('Model Coef: {}'.format(flower_classifier.prob_dist_weight(tf.constant(1, shape=(1, 1))).numpy()[0]))

            print("-" * 40)

            

    history['train_loss'].append(train_loss_obj.result())        

    history['train_acc'].append(train_acc_obj.result())        

        

    # print training results

    print('\nEpoch {}'.format(epoch + 1))

    print('Train Loss: {:.6f}'.format(train_loss_obj.result()))

    print('Train  Acc: {:.6f}'.format(train_acc_obj.result()))



    print('\nComputing valid predictions...')

    

    all_valid_preds = []

    for batch_idx, inputs in enumerate(validation_dist_dataset):

    

        """

        # The return values of `strategy.experimental_run_v2 ` are actually `PerReplica` objects.

        # For valid_step we defined above, the return value is a tuple of 2 `PerReplica` object.

        # The 2nd element in the return value is for valid_preds, and it looks like. 

        

        PerReplica:{

            0 /job:worker/replica:0/task:0/device:TPU:0: tf.Tensor(..., shape=(16, 104), dtype=float32),

            1 /job:worker/replica:0/task:0/device:TPU:0: tf.Tensor(..., shape=(16, 104), dtype=float32),

            ...

            7 /job:worker/replica:0/task:0/device:TPU:0: tf.Tensor(..., shape=(16, 104), dtype=float32)

        }

        

        # The 1st element in the return value is for valid_loss. It's similar to the above, but just scalar.

        

        So we have to convert each of them to a single tf.Tensor manually.

        

        Remark: Maybe there is a method to do this automatically. If you find it, let me know!

        """

        per_replica_valid_results = distributed_valid_step(inputs)

        per_replica_valid_loss, per_replica_valid_preds = per_replica_valid_results

        

        valid_preds = tf.concat(per_replica_valid_preds.values, axis=0)

        valid_preds = valid_preds.numpy()        

   

        all_valid_preds.append(valid_preds)



        valid_loss = tf.stack(per_replica_valid_loss.values, axis=0)

        valid_loss = tf.math.reduce_sum(valid_loss)

        valid_loss_obj(valid_loss)

    

    # print validation results

    print('\nEpoch {}'.format(epoch + 1))

    print('Valid Loss: {:.6f}'.format(valid_loss_obj.result()))

    print('Valid  Acc: {:.6f}'.format(valid_acc_obj.result()))

    

    history['valid_loss'].append(valid_loss_obj.result())        

    history['valid_acc'].append(valid_acc_obj.result())     

    

    # Make the whole valid predictions as a numpy array

    all_valid_preds = np.concatenate(all_valid_preds, axis=0, out=None)



    # Save valid predictions

    with open("valid_preds_epoch_{}".format(epoch), "w", encoding="UTF-8") as fp:

        np.savetxt(fp, all_valid_preds, delimiter=',', fmt='%10.6f')



    cm_predictions = np.argmax(all_valid_preds, axis=-1)

    f1 = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')    



    print('\nf1       : {:.6f}'.format(f1))

    print('precision: {:.6f}'.format(precision))

    print('recall   : {:.6f}'.format(recall))    

    

    history['valid_f1'].append(f1)        

    history['valid_precision'].append(precision)

    history['valid_recall'].append(recall)

    

    history['model_coefs'].append(flower_classifier.prob_dist_weight(tf.constant(1, shape=(1, 1))).numpy()[0])

    

    epoch_end_time = datetime.datetime.now()

    epoch_elapsed_time = (epoch_end_time - epoch_start_time).total_seconds()        

    print('\nTime taken for 1 epoch: {} secs'.format(epoch_elapsed_time))

    print("\n" + "=" * 80 + "\n")

    

print('Computing predictions...')

prediction_start_time = datetime.datetime.now()



all_test_preds = []

for batch_idx, inputs in enumerate(test_dist_dataset):

    

    per_replica_test_preds = distributed_test_step(inputs)



    test_preds = tf.concat(per_replica_test_preds.values, axis=0)

    test_preds = test_preds.numpy()        



    all_test_preds.append(test_preds)

    

# Make the whole test predictions as a numpy array

all_test_preds = np.concatenate(all_test_preds, axis=0, out=None)



# Save test predictions

with open("test_preds", "w", encoding="UTF-8") as fp:

    np.savetxt(fp, all_test_preds, delimiter=',', fmt='%10.6f')

    

print('Computing predictions finished.')



prediction_end_time = datetime.datetime.now()

prediction_elapsed_time = (prediction_end_time - prediction_start_time).total_seconds()        

print('\nTime taken for prediction: {} secs'.format(prediction_elapsed_time))
display_training_curves(history['train_loss'], history['valid_loss'], 'loss', 211)

display_training_curves(history['train_acc'], history['valid_acc'], 'accuracy', 212)
for epoch in range(EPOCHS - 1, EPOCHS):

    

    # Load valid predictions

    with open("valid_preds_epoch_{}".format(epoch), "r", encoding="UTF-8") as fp:

        cm_probabilities = np.loadtxt(fp, delimiter=',')

        

    cm_predictions = np.argmax(cm_probabilities, axis=-1)

    

    print("Epoch {}\n".format(epoch))

    print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

    print("Predicted labels: ", cm_predictions.shape, cm_predictions)

    

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    #cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

    display_confusion_matrix(cmat, score, precision, recall)

    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

    print("-"  * 80)
# Save confusion matrix

with open("confusion_matrix", "w", encoding="UTF-8") as fp:

    np.savetxt(fp, cmat, delimiter=',', fmt='%d')



cmat_normalized = (cmat.T / cmat.sum(axis=1)).T # normalized

with open("confusion_matrix_normalized", "w", encoding="UTF-8") as fp:

    np.savetxt(fp, cmat_normalized, delimiter=',', fmt='%d')

    

print(cmat_normalized)



display_confusion_matrix(cmat_normalized, score, precision, recall)
def display_scores(f1, precision, recall, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(f1)

    ax.plot(precision)

    ax.plot(recall)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['f1', 'precision', 'recall'])

        

display_scores(history['valid_f1'], history['valid_precision'], history['valid_recall'], 'score', 211)
def display_model_coefs(model_coefficients, title, subplot):

    

    model_coefficients = np.transpose(np.array(model_coefficients))

    

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    for model_coef in model_coefficients:

        ax.plot(model_coef)



    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['model_coefs_{}'.format(idx) for idx in range(len(flower_classifier.image_embedding_layers))])

        

display_model_coefs(history['model_coefs'], "model_coefs", 211)
# Load test predictions

with open("test_preds", "r", encoding="UTF-8") as fp:

    probabilities = np.loadtxt(fp, delimiter=',')

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

!head submission.csv
!ls -l