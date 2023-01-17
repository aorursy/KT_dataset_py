# Data loading and processing imports
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk
from skimage.morphology import label
import gc
import datetime

# Hp optimization imports
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# Modelling imports
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
# Some useful Keras callbacks
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# TODO: Make this work
# !pip install mlflow and using the Packages installer in the kernel aren't enough. :(
# Enable garbage collection
gc.enable()
# Global constants
SEED = 31415
# Set it to a small number so that it can run in this notebook.
# Notice that if it is too small, hyperopt behaves as random selection.
MAX_EVALS = 1
CUSTOM_DICE_LOSS_EPSILON = 1e-3
# TODO: Use pathlib later
BASE_DATA_PATH = "../input/"
MASKS_DATA_PATH = os.path.join(BASE_DATA_PATH, 'train_ship_segmentations.csv')
TRAIN_IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, 'train')
TEST_IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, 'test')
# According to the data description, some files from the test folder shoud be ignore
TEST_IMGS_TO_IGNORE = ['13703f040.jpg',
 '14715c06d.jpg',
 '33e0ff2d5.jpg',
 '4d4e09f2a.jpg',
 '877691df8.jpg',
 '8b909bb20.jpg',
 'a8d99130e.jpg',
 'ad55c3143.jpg',
 'c8260c541.jpg',
 'd6c7f17c7.jpg',
 'dc3e7c901.jpg',
 'e44dffe88.jpg',
 'ef87bad36.jpg',
 'f083256d8.jpg']
# These two patiences thresholds are small so that this notebook can run with limited resources
REDUCE_LR_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 2
# Fraction of the validation size (compared to the total train size)
VALID_SIZE = 0.3
# Minimum size (in KB) of files to keep
FILE_SIZE_KB_THRESHOLD = 50
# The original size of the image
# TODO: Check if it is really 3 channels.
IMG_SIZE = (768, 768, 3)
# downsampling in preprocessing
# TODO: Should these be hp to optimize as well?
IMG_SCALING = (4, 4)
EDGE_CROP = 16
# downsampling inside the network
NET_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 600
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 150
MAX_TRAIN_EPOCHS = 10
# The hyperparameters space over which to search. 
# TODO: Improve the ranges and the used distributions to sample.
HYPERPARAMETERS_SPACE = {
        # TODO: What is the best scale for Gaussian noise?
        'gaussian_noise': hp.choice('gaussian_noise', [0.1, 0.2, 0.3]),
        'batch_size':  hp.choice('batch_size', [8, 16, 32, 64, 128]),
        'upsample_mode': hp.choice('upsmaple_mode', ["SIMPLE", "DECONV"]),
        'augment_brightness': hp.choice('augment_brightness', [True, False]),
        'max_train_steps': MAX_TRAIN_STEPS, 
        'max_train_epochs': MAX_TRAIN_EPOCHS, 
        'valid_img_count': VALID_IMG_COUNT,
        'img_scaling': IMG_SCALING,
        'edge_crop': EDGE_CROP,
        'net_scaling': NET_SCALING
    }

# The classic RLE encoding code, from the great: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_threshold=1e-3, max_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_threshold:
        return '' ## no need to encode if it's all zeros
    if max_threshold and np.mean(img) > max_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    # TODO: This mask size shouldn't be hardcoded (768, 768).
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    # TODO: This mask size shouldn't be hardcoded (768, 768).
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

# TODO: Add some documentation
def make_image_gen(input_df, batch_size, img_scaling):
    df = input_df.copy()
    all_batches = list(df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_IMAGES_FOLDER, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            c_mask = np.expand_dims(c_mask, axis=-1)
            if img_scaling is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
# TODO: Add some documentation for the augmentation pipeline as well.
# TODO: Finish this and add some documentation.
def build_image_generator(augment_brightness):
    """ Build an image data generator (for images and labels). 
    For more details about this class, check the documentation here: 
    https://keras.io/preprocessing/image/.
    """
    # TODO: Describe what each data augementation parameter does.
    data_generator_dict = dict(featurewise_center = False, 
                               samplewise_center = False,
                               rotation_range = 45, 
                               width_shift_range = 0.1, 
                               height_shift_range = 0.1, 
                               shear_range = 0.01,
                               zoom_range = [0.9, 1.25],  
                               horizontal_flip = True, 
                               vertical_flip = True,
                               fill_mode = 'reflect',
                               data_format = 'channels_last')
    # brightness can be problematic since it seems to change the labels differently from the images 
    if augment_brightness:
        data_generator_dict['brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**data_generator_dict)

    if augment_brightness:
        data_generator_dict.pop('brightness_range')
    label_gen = ImageDataGenerator(**data_generator_dict)
    return image_gen, label_gen

# TODO: Add some documentation and improve variables names.
def create_aug_gen(in_gen, augment_brightness, seed = None):
    image_gen, label_gen = build_image_generator(augment_brightness)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
URI = "your/remote/instance"
EXPERIMENT_NAME = "airbus_ship_detection_u_net"
try:
    import mlflow
    mlflow.set_tracking_uri(URI)
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
except (ImportError, ModuleNotFoundError):
    print("Unfortunately, MLflow isn't available. :(")
class HyperoptHPOptimizer(object):
    
    def __init__(self, generate_model_history, hyperparameters_space, max_evals):
        # TODO: Add some documentation
        self.generate_model_history = generate_model_history
        self.trials = Trials()
        self.max_evals = max_evals
        self.hyperparameters_space = hyperparameters_space

    def _get_loss_with_mlflow(self, hyperparameters):
        # MLflow will track and save hyperparameters, loss, and scores. 
        with mlflow.run(experiment_id=EXPERIMENT_ID):
            print("Training with the following hyperparameters: ")
            print(hyperparameters)
            for k, v in hyperparameters.iteritems():
                mlflow.log_param(k, v)
            history = self.generate_model_history(hyperparameters)
            # Log the various losses and metrics (on train and validation)
            for k, v in history.history.items():
                mlflow.log_metric(k, v[-1])
            # Use the last validation loss from the history object to optimize
            loss = history.history["val_loss"][-1]
            return {'loss': loss, 'status': STATUS_OK}
        
    def _get_loss_without_mlflow(self, hyperparameters):
            print("Training with the following hyperparameters: ")
            print(hyperparameters)
            history = self.generate_model_history(hyperparameters)
            # Use the last validation loss from the history object to optimize
            loss = history.history["val_loss"][-1]
            return {'loss': loss, 'status': STATUS_OK}
    
    def get_loss(self, hyperparameters):
        try:
            import mlflow
            return self._get_loss_with_mlflow(hyperparameters)
        except (ImportError, ModuleNotFoundError):
            return self._get_loss_without_mlflow(hyperparameters)

    def optimize(self):
        """
        This is the optimization function that given a space of 
        hyperparameters and a scoring function, finds the best hyperparameters.
        """
        # Use the fmin function from Hyperopt to find the best hyperparameters
        # Here we use the tree-parzen estimator method. 
        best = fmin(self.get_loss, self.hyperparameters_space, algo=tpe.suggest, 
                    trials=self.trials,  max_evals=self.max_evals)
        return best
# TODO: Use the https://www.kaggle.com/yassinealouini/idiomatic-pandas-processing?scriptVersionId=5308185
# to cleanup this part. 


def get_data(file_size_kb_threshold = FILE_SIZE_KB_THRESHOLD):
    """ Load and process train and validation data (images and masks).
    """
    # Two vectorized functions
    _v_path_join = np.vectorize(os.path.join)
    _v_file_size = np.vectorize(lambda fp: (os.stat(fp).st_size) / 1024)

    # Read the masks DataFrame
    masks_df = pd.read_csv(MASKS_DATA_PATH)
    print(masks_df.shape)
    ships_df = (masks_df.groupby('ImageId')["EncodedPixels"]
                        .count()
                        .reset_index()
                        .rename(columns={"EncodedPixels": "ships"})
                        .assign(has_ship=lambda df: np.where(df['ships'] > 0, 1, 0))
                        .assign(file_path=lambda df: _v_path_join(TRAIN_IMAGES_FOLDER, 
                                                                  df.ImageId.astype(str)))
                        .assign(file_size_kb=lambda df: _v_file_size(df.file_path))
                        .loc[lambda df: df.file_size_kb > file_size_kb_threshold, :])

    print(ships_df.head())
    train_ids, valid_ids = train_test_split(ships_df, 
                     test_size = VALID_SIZE, 
                     stratify = ships_df['ships'])
    train_df = pd.merge(masks_df, train_ids)
    valid_df = pd.merge(masks_df, valid_ids)


    print(train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')
    
    # Rebalance the training DataFrame.
    # TODO: Improve the rebalancing code.
    train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+2)//3)
    balanced_train_df = train_df.groupby('grouped_ship_count').apply(lambda x: x.sample(1500))
    return balanced_train_df, valid_df
# Garbage collection time!
gc.collect()
# Build U-Net model

# TODO: Document the various used hyperparameters
def build_u_net_model(input_shape, upsample_mode="DECONV", gaussian_noise=0.1, 
                      padding="same", net_scaling=None, img_scaling=IMG_SCALING, *args, **kargs):
    # TODO: Move these to the utils section?
    def _upsample_conv(filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters, kernel_size, strides=strides, 
                                      padding=padding)
    def _upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)    
    
    upsample_dict = {"DECONV": _upsample_conv, "SIMPLE": _upsample_simple}
    
    upsample = upsample_dict.get(upsample_mode, _upsample_simple)

    input_img = layers.Input(input_shape, name = 'RGB_Input')
    pp_in_layer = input_img
    
    # TODO: Add dropout for regularization?

    # Some preprocessing
    # TODO: Add explanation of the different stes.
    if net_scaling is not None:
        pp_in_layer = layers.AvgPool2D(net_scaling)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(gaussian_noise)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding) (pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding) (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding) (p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding) (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding) (p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding) (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding) (p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding) (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding=padding) (p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding=padding) (c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding=padding) (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding) (u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding) (c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding=padding) (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding) (u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding) (c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding=padding) (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding) (u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding) (c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding=padding) (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding) (u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding) (c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    # TODO: Why is this commented
    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if net_scaling is not None:
        d = layers.UpSampling2D(net_scaling)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    if img_scaling is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(img_scaling, 
                                           input_shape = (None, None, 3)))
        fullres_model.add(seg_model)
        fullres_model.add(layers.UpSampling2D(img_scaling))
    else:
        fullres_model = seg_model
    fullres_model.summary()
    return fullres_model
def dice_metric(y_true, y_pred, smooth=1):
    """
    Also known as the Sorensen-Dice coeffecient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient),
    this is the F1 score (i.e. harmonic mean of precision and recall). 
    Notice that this metric has a smoothness parameter (smooth) to avoid division by 0.
    """
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    # Compute the dice metric and then take the average over the samples.
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def custom_dice_loss(in_gt, in_pred):
    """ This is a custom loss function that has two contributions: binary crossentropy 
    (this is the usual metric used for binary classification) and - the dice metric (to turn it into a loss).
    """
    return CUSTOM_DICE_LOSS_EPSILON * binary_crossentropy(in_gt, in_pred) - dice_metric(in_gt, in_pred)

def true_positive_rate_metric(y_true, y_pred):
    """ TPR (true positive rate) measures the ratio of true positives over positives.
    Notice the round step so that the predicted values are transformed into 0 or 1 values instead of floats
    in the range [0, 1]. 
    """
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)



def f2_metric():
    # TODO: Add the F2 metric. 
    pass

METRICS = [true_positive_rate_metric, 
           dice_metric, 
           "binary_accuracy"]
def get_compiled_model(hyperparameters, input_shape=IMG_SIZE):
    model = build_u_net_model(input_shape, **hyperparameters)
    # TODO: This should be in the hp list as well.
    learning_rate = 1e-3
    adam_optimizer = Adam(learning_rate, decay=1e-6)
    model.compile(optimizer=adam_optimizer, loss=custom_dice_loss, metrics=METRICS)
    return model
# TODO: Add some documentation for these callbacks.

weight_path = "best_weights.h5"

# TODO: Move some of the hyperparameters to the constants list
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', 
                             verbose=1, save_best_only=True, 
                             mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=REDUCE_LR_PATIENCE, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)
# probably needs to be more patient, but kaggle time is limited
early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=EARLY_STOPPING_PATIENCE)

CALLBACKS = [checkpoint, early, reduceLROnPlat]
# TODO: Add some documentation
def _generate_model_history(input_train_df, input_valid_df, hyperparameters, n_samples, input_shape):
    # Copy input DataFrames to avoid side-effects
    train_df = input_train_df.copy()
    valid_df = input_valid_df.copy()
    max_train_steps = hyperparameters["max_train_steps"]
    batch_size = hyperparameters["batch_size"]
    img_scaling = hyperparameters["img_scaling"]
    max_train_epochs = hyperparameters["max_train_epochs"]
    valid_img_count =  hyperparameters["valid_img_count"]
    augment_brightness = hyperparameters["augment_brightness"]
    steps_per_epoch = min(max_train_steps, n_samples//batch_size)
    print("Using {} steps per epoch.".format(steps_per_epoch))
    img_genarator = make_image_gen(train_df, batch_size, img_scaling)
    augmented_img_generator = create_aug_gen(img_genarator, augment_brightness)
    # TODO: Improve the names of these returned values.
    valid_x, valid_y = next(make_image_gen(valid_df, valid_img_count, img_scaling))
    model = get_compiled_model(hyperparameters, input_shape)
    # Use only one worker for thread-safety reason. 
    # TODO: Investigate this claim.
    return model.fit_generator(augmented_img_generator, steps_per_epoch=steps_per_epoch,
                                epochs=max_train_epochs, validation_data=(valid_x, valid_y),
                                callbacks=CALLBACKS, workers=1)



train_df, valid_df = get_data()
n_samples = train_df.shape[0]
input_shape = IMG_SIZE
# We wrap the _generate_model_history so that the train_df DataFrame isn't reconstructed for each iteration.
generate_model_history = lambda hp: _generate_model_history(train_df, valid_df, hp, n_samples, input_shape)
hp_optimizer = HyperoptHPOptimizer(generate_model_history, hyperparameters_space=HYPERPARAMETERS_SPACE, 
                                   max_evals=MAX_EVALS)
optimal_hyperparameters = hp_optimizer.optimize()
print(optimal_hyperparameters)
def save_optimal_model(input_train_df, input_valid_df, hyperparameters, n_samples, input_shape, 
                       model_path):
    # Copy input DataFrames to avoid side-effects
    train_df = input_train_df.copy()
    valid_df = input_valid_df.copy()
    max_train_steps = hyperparameters["max_train_steps"]
    batch_size = hyperparameters["batch_size"]
    img_scaling = hyperparameters["img_scaling"]
    max_train_epochs = hyperparameters["max_train_epochs"]
    valid_img_count =  hyperparameters["valid_img_count"]
    augment_brightness = hyperparameters["augment_brightness"]
    steps_per_epoch = min(max_train_steps, n_samples//batch_size)
    print("Using {} steps per epoch.".format(steps_per_epoch))
    img_genarator = make_image_gen(train_df, batch_size, img_scaling)
    augmented_img_generator = create_aug_gen(img_genarator, augment_brightness)
    # TODO: Improve the names of these returned values.
    valid_x, valid_y = next(make_image_gen(valid_df, valid_img_count, img_scaling))
    model = get_compiled_model(hyperparameters, input_shape)
    model.fit_generator(augmented_img_generator, steps_per_epoch=steps_per_epoch,
                        epochs=max_train_epochs, validation_data=(valid_x, valid_y),
                        callbacks=CALLBACKS, workers=1)
    model.save(model_path)
# It is usually a good idea to add the creation datetime in the model name
current_datetime = datetime.datetime.now()
model_path = "optimal_custom_u_net_model.h5".format(current_datetime.strftime("%Y-%m-%d-%H-%M"))
save_optimal_model(train_df, valid_df, optimal_hyperparameters, n_samples, input_shape, model_path)