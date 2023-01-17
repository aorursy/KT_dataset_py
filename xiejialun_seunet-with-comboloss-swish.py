import numpy as np 

import pandas as pd 

import os



import keras

import keras.backend as K

import keras.layers as klayers

from keras.preprocessing.image import load_img

import albumentations as albu

from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE , GridDistortion)

from sklearn.model_selection import train_test_split



import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import gc

import random

main_dir = '../input/severstal-steel-defect-detection/'

os.listdir(main_dir)
pretrain_weights_path = '../input/steel-pretrain/unet_out.hdf5'

batch_size = 32

epochs = 15

reshape_rgb = (256, 512, 3)

reshape_mask = (256, 512)

mask_threshold = 3500

mask_bound = 0.8

lr = 1e-4
train_fns = os.listdir(main_dir + 'train_images')

test_fns = os.listdir(main_dir + 'test_images')

train_seg = pd.read_csv(main_dir + 'train.csv')



print(len(train_fns))

print(len(test_fns))

print(train_seg.shape)



train_seg.head(5)
train_seg['ImageId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[0])

train_seg['ClassId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[1])

train_seg = train_seg.drop(['ImageId_ClassId'], axis = 1)
train_seg['has_label'] = train_seg['EncodedPixels'].map(lambda x : 1 if isinstance(x,str) else 0)

train_seg.head(5)
Image_with_label = train_seg.groupby(['ImageId'])['has_label'].sum().value_counts()

print(Image_with_label)

plt.figure(figsize = (6,4))

plt.bar(Image_with_label.index, Image_with_label.values)

plt.xlabel('label number')

plt.ylabel('count')

plt.title('Count of label number in single image')

plt.show()
class_with_label = train_seg.groupby(['ClassId'])['has_label'].sum().reset_index()

plt.figure(figsize = (6,4))

plt.bar(class_with_label.ClassId.values, class_with_label.has_label.values)

plt.xlabel('class id')

plt.ylabel('count')

plt.title('Count of each class id who has labeled')

plt.show()
def rle_encoding(mask):

    

    pixels = mask.T.flatten()

    pixels = np.concatenate([[0], pixels,[0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    if len(runs) % 2:

        runs = np.append(runs,len(pixels))

    runs[1::2] -= runs[0::2]

    

    return ' '.join(str(x) for x in runs)



def rle_decoding(rle, mask_shape = (256,1600)):

    strs = rle.split(' ')

    starts = np.asarray(strs[0::2], dtype = int) - 1

    lengths = np.asarray(strs[1::2], dtype = int)

    ends = starts + lengths

    

    mask = np.zeros(mask_shape[0] * mask_shape[1], dtype = np.uint8)

    for s,e in zip(starts, ends):

        mask[s:e] = 1

    return mask.reshape(mask_shape, order = 'F')



def merge_masks(image_id, df, mask_shape = (256,1600), reshape = None):

    

    rles = df[df['ImageId'] == image_id].EncodedPixels.iloc[:]

    depth = rles.shape[0]

    if reshape:

        masks = np.zeros((*reshape, depth), dtype = np.uint8)

    else:

        masks = np.zeros((mask_shape[0], mask_shape[1],depth), dtype = np.uint8)

    

    for idx in range(depth):

        if isinstance(rles.iloc[idx], str):

            if reshape:

                cur_mask = rle_decoding(rles.iloc[idx], mask_shape)

                cur_mask = cv2.resize(cur_mask, (reshape[1], reshape[0]))

                masks[:,:,idx] += cur_mask

            else:         

                masks[:,:,idx] += rle_decoding(rles.iloc[idx], mask_shape)

    return masks   
#Check rle_encoding and rle_decoding



rle_1 = train_seg['EncodedPixels'].iloc[0]

mask_1 = rle_decoding(rle_1)

rle_2 = rle_encoding(mask_1)

mask_2 = rle_decoding(rle_2)



plt.figure(figsize = (16,8))

plt.imshow(mask_1)

plt.show()

plt.figure(figsize = (16,8))

plt.imshow(mask_2)

plt.show()
def display_img_masks(img, masks, image_id = "", title = ""):

    for idx in range(masks.shape[-1]):

        plt.figure(figsize = (24,6))

        plt.imshow(img)

        plt.imshow(masks[:,:,idx], alpha = 0.35, cmap = 'gray')

        plt.title(image_id + '_class_' + str(idx+1) + title)

        plt.show()
image_dir = main_dir + 'train_images/' + train_seg['ImageId'].iloc[0]

image_id = train_seg['ImageId'].iloc[0]

img = plt.imread(image_dir)

masks = merge_masks(image_id, train_seg)



display_img_masks(img,masks, image_id)
gamma = 1.2

inverse_gamma = 1.0 / gamma

look_up_table = np.array([((i/255.0) ** inverse_gamma) * 255.0 for i in np.arange(0,256,1)]).astype("uint8")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))



def contrast_enhancement(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img[:,:,0] = clahe.apply(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

    return img



def gamma_correction(img):

    return cv2.LUT(img.astype('uint8'), look_up_table)



def load_target_image(path, grayscale = False, color_mode = 'rgb', target_size = reshape_rgb,

                     interpolation = 'nearest'):

    

    return load_img(path = path, grayscale = grayscale, color_mode = color_mode,

                   target_size = target_size, interpolation = interpolation)



def input_gen(filenames, segs, aug, batch_size = 4, reshape = (256,1600)):

    

    load_dir = main_dir + 'train_images/'

    

    batch_rgb = []

    batch_mask = []

    

    while True:

        fns = random.sample(filenames, batch_size)

        seed = np.random.choice(range(999))

        for fn in fns:

            cur_img = np.asarray(load_target_image(path = load_dir + fn))

            cur_img = gamma_correction(cur_img)

            cur_img = contrast_enhancement(cur_img)

            masks = merge_masks(fn, segs, reshape = reshape)            

            processed = aug(image = cur_img, mask = masks)



            batch_rgb.append(processed['image']/255.0)

            batch_mask.append(processed['mask'])

        

        batch_rgb, batch_mask = np.stack(batch_rgb), np.stack(batch_mask)

        

        yield batch_rgb, batch_mask

        batch_rgb = []

        batch_mask = []

        gc.collect()
train_x, valid_x = train_test_split(train_fns, test_size = 0.2, random_state = 2019)

print(len(train_x))

print(len(valid_x))
aug_for_train = albu.Compose([HorizontalFlip(p=0.5),

                             ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=15, p=0.5),

                             GridDistortion(p=0.5)])

aug_for_valid = albu.Compose([])





train_aug_gen = input_gen(train_x, train_seg, aug_for_train, batch_size = batch_size,reshape = reshape_mask)

valid_aug_gen = input_gen(valid_x, train_seg, aug_for_valid, batch_size = batch_size,reshape = reshape_mask)
class Unet:

    

    def __init__(self, input_shape = (256,1600,3), output_units = 4):

        

        self.input_shape = input_shape

        self.output_units = output_units

        

    def _swish(self,x):

        return K.sigmoid(x)*x

    

    def _SEBlock(self, se_ratio=16, activation = "relu"):

        

        def f(input_x):



            input_channels = input_x._keras_shape[-1]

            reduced_channels = max(input_channels // se_ratio, 8)

            

            x = klayers.GlobalAveragePooling2D()(input_x)

            x = klayers.Dense(units = reduced_channels, kernel_initializer = "he_normal")(x)

            x = klayers.Activation(activation)(x)

            x = klayers.Dense(units = input_channels, activation = 'sigmoid', kernel_initializer = "he_normal")(x)

            

            return klayers.multiply([input_x, x])

        

        return f    

    

    def _cn_bn_act(self, filters = 64, kernel_size = (3,3), bn_flag = False, activation = "relu"):

        

        def f(input_x):

            

            x = input_x

            x = klayers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1,1), padding = "same", kernel_initializer = "he_normal")(x)

            x = klayers.BatchNormalization()(x) if bn_flag == True else x

            x = klayers.Activation(activation)(x)

            

            return x

        

        return f

    

    def _UpSamplingBlock(self, filters = 64, kernel_size = (3,3), upsize = (2,2), bn_flag = True, up_flag = False, se_flag = True):

        

        def f(up_c, con_c):

            

            if up_flag:

                x = klayers.UpSampling2D(size = upsize, interpolation = 'bilinear')(up_c)

            else:

                x = klayers.Conv2DTranspose(filters = filters, kernel_size = (2,2), strides = upsize, padding = "same", kernel_initializer = "he_normal")(up_c)

            

            x = klayers.concatenate([x,con_c])

            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)

            x = self._SEBlock(activation = self._swish)(x) if se_flag == True else x

            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)

            

            return x

        return f

    

    def _DownSamplingBlock(self, filters = 64, kernel_size = (3,3), downsize = (2,2), bn_flag = True, is_bottom = False, se_flag = True):

        

        def f(input_x):

            

            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(input_x)

            x = self._SEBlock(activation = self._swish)(x) if se_flag == True else x

            c = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)

            return c if (is_bottom == True) else (c,klayers.MaxPooling2D(pool_size = downsize)(c))

            

        return f

    

    def build_unet(self):

        

        

        input_x = klayers.Input(shape = self.input_shape)

        #encoder region

        c1,p1 = self._DownSamplingBlock(filters = 16)(input_x)

        c2,p2 = self._DownSamplingBlock(filters = 32)(p1)

        c3,p3 = self._DownSamplingBlock(filters = 32)(p2)

        c4,p4 = self._DownSamplingBlock(filters = 64)(p3)

        c5,p5 = self._DownSamplingBlock(filters = 128)(p4)

        

        c6 = self._DownSamplingBlock(filters = 256, is_bottom = True)(p5)

        

        #decoder region

        u7 = self._UpSamplingBlock(filters = 128)(c6,c5)

        u8 = self._UpSamplingBlock(filters = 64)(u7,c4)

        u9 = self._UpSamplingBlock(filters = 32)(u8,c3)

        u10 = self._UpSamplingBlock(filters = 32)(u9,c2)

        u11 = self._UpSamplingBlock(filters = 16)(u10,c1)

        output_x = klayers.Conv2D(filters = self.output_units, kernel_size = (1,1), padding = "same", activation = "sigmoid", kernel_initializer = "he_normal")(u11)

        model = keras.models.Model(inputs = [input_x], outputs = [output_x])

        return model

        
unet_builder = Unet(input_shape = reshape_rgb)

unet = unet_builder.build_unet()

unet.summary()
if pretrain_weights_path != None:

    unet.load_weights(pretrain_weights_path)
def Tversky_Loss(y_true, y_pred, smooth = 1, alpha = 0.3, beta = 0.7, flatten = False):

    

    if flatten:

        y_true = K.flatten(y_true)

        y_pred = K.flatten(y_pred)

    

    TP = K.sum(y_true * y_pred)

    FP = K.sum((1-y_true) * y_pred)

    FN = K.sum(y_true * (1-y_pred))

    

    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    

    return 1 - tversky_coef



def Focal_Loss(y_true, y_pred, alpha = 0.8, gamma = 2.0, flatten = False):

    

    if flatten:

        y_true = K.flatten(y_true)

        y_pred = K.flatten(y_pred)    

    

    bce = keras.losses.binary_crossentropy(y_true, y_pred)

    bce_exp = K.exp(-bce)

    

    loss = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)

    return loss



def weighted_bce(weight = 0.6):

    

    def convert_2_logits(y_pred):

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        return tf.log(y_pred / (1-y_pred))

    

    def weighted_binary_crossentropy(y_true, y_pred):

        y_pred = convert_2_logits(y_pred)

        loss = tf.nn.weighted_cross_entropy_with_logits(logits = y_pred, targets = y_true, pos_weight = weight)

        return loss

    

    return weighted_binary_crossentropy



def Combo_Loss(y_true, y_pred, a = 0.4, b = 0.2, c= 0.4):

    

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    

    return a*weighted_bce()(y_true, y_pred) + b*Focal_Loss(y_true_f, y_pred_f) + c*Tversky_Loss(y_true_f, y_pred_f)



def Dice_coef(y_true, y_pred, smooth = 1):



    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

 

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def Dice_loss(y_true, y_pred):   

    return  1.0 - Dice_coef(y_true, y_pred) 
optimizer = keras.optimizers.Adam(lr = lr, decay = 1e-6)

unet.compile(loss = Combo_Loss, optimizer = optimizer, metrics = [Dice_coef])



reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 2, mode = 'min', factor = 0.5, verbose = 1)

cp = keras.callbacks.ModelCheckpoint('unet_out.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')

es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min')

training_callbacks = [reduce_lr, cp, es]



steps_per_epoch = len(train_x) // batch_size

validation_steps = len(valid_x) // batch_size
'''

history = unet.fit_generator(train_aug_gen, steps_per_epoch = steps_per_epoch, epochs = epochs,

                              validation_data = valid_aug_gen, validation_steps = validation_steps, verbose = 1, callbacks = training_callbacks)





unet.load_weights('unet_out.hdf5')

'''
'''

def plot_training_result(history):

    

    plt.figure(figsize = (8,6))

    plt.plot(history.history['loss'], '-', label = 'train_loss', color = 'g')

    plt.plot(history.history['val_loss'], '--', label = 'valid_loss', color ='r')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Loss on unet')

    plt.legend()

    plt.show()

    

    plt.figure(figsize = (8,6))

    plt.plot(history.history['Dice_coef'], '-', label = 'train_Dice_coef', color = 'g')

    plt.plot(history.history['val_Dice_coef'], '--', label = 'valid_Dice_coef', color ='r')

    plt.xlabel('epoch')

    plt.ylabel('Dice_Coef')

    plt.title('Dice_Coef on unet')

    plt.legend()

    plt.show()



plot_training_result(history)

'''


def predict_masks(img):

    

    masks = unet.predict(np.expand_dims(img, axis = 0))

    masks = masks > mask_bound

    masks = np.squeeze(masks, axis = 0)

    

    return masks



valid_aug_gen = input_gen(valid_x, train_seg, aug_for_valid, batch_size = 8, reshape = reshape_mask)



valid_data, valid_label = next(valid_aug_gen)

for x,y in zip(valid_data, valid_label):

    display_img_masks(x,y,title = "_ground truth")

    prediction = predict_masks(x)

    display_img_masks(x,prediction, title = '_prediction')



valid_count = 1600

valid_predictions = np.zeros((valid_count,256,1600))

valid_truths = []

valid_aug_gen = input_gen(valid_x, train_seg, aug_for_valid, batch_size = 1, reshape = reshape_mask)



for idx in tqdm_notebook(range(int(valid_count/4))):

    img, true_masks = next(valid_aug_gen)  

    true_masks = true_masks[0]

    pred_masks = np.squeeze( unet.predict(img), axis = 0)

    pred_masks = cv2.resize(pred_masks, (1600, 256), interpolation = cv2.INTER_LINEAR)

    

    for i in range(pred_masks.shape[-1]):

        valid_predictions[idx*4+i, :, :] = pred_masks[:,:,i]

    

    for tm_id in range(true_masks.shape[-1]):

        valid_truths.append(cv2.resize(true_masks[:,:,tm_id], dsize = (1600,256), interpolation = cv2.INTER_LINEAR))

    

    gc.collect()


def Dice(y_true, y_pred, smooth = 1):



    y_true_f = y_true.flatten()

    y_pred_f = y_pred.flatten()

 

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)



def single_mask_reduce(mask, minSize):

    label_num, label_mask = cv2.connectedComponents(mask.astype(np.uint8))

    reduced_mask = np.zeros(mask.shape, np.float32)

    for label in range(1, label_num):

        single_label_mask = (label_mask == label)

        if single_label_mask.sum() > minSize:

            reduced_mask[single_label_mask] = 1

            

    return reduced_mask



class_params = {}



for class_id in range(4):

    print(class_id)

    attemps = []

    for th in range(0,100,5):

        th = th / 100

        d = 0

        for ms in [3500]:

            for i in range(class_id, len(valid_predictions), 4):

                valid_pred = valid_predictions[i]

                valid_pred = np.array(valid_pred > th, np.uint8)

                valid_pred = single_mask_reduce(valid_pred, ms)

                if valid_pred.sum() == 0 and np.sum(valid_truths[i]) == 0:

                    d += 1

                else:

                    d += Dice(valid_pred, valid_truths[i])

            attemps.append((th, ms, d/(len(valid_predictions)//4) ))

            

    attempts_df = pd.DataFrame(attemps, columns = ['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending = False)

    print(attempts_df.head())

    best_threshold = attempts_df['threshold'].values[0]

    best_size = attempts_df['size'].values[0]

    

    class_params[class_id] = (best_threshold, best_size)



sub = pd.DataFrame(columns = ['ImageId_ClassId', 'EncodedPixels'])



def mask_reduce(masks, class_params):

    

    for idx in range(masks.shape[-1]):

        masks[:,:,idx] = np.array(masks[:,:,idx] > class_params[idx][0])

        

        label_num, labeled_mask = cv2.connectedComponents(masks[:,:,idx].astype(np.uint8))

        reduced_mask = np.zeros(masks.shape[:2],np.float32)

        

        for label in range(1, label_num):

            single_label_mask = (labeled_mask == label)

            if single_label_mask.sum() >  class_params[idx][1]:

                reduced_mask[single_label_mask] = 1

        

        masks[:,:,idx] = reduced_mask

        

    return masks





def prediction_encoding(fn, img_dir, submission, target_shape = (256,1600)):

    img = np.asarray(load_target_image(path = os.path.join(img_dir,fn)))

    img = gamma_correction(img)

    img = contrast_enhancement(img)/255.0

    masks = unet.predict(np.expand_dims(img, axis = 0))

    masks = np.squeeze( np.round(masks), axis = 0)

    masks = cv2.resize(masks, (target_shape[1], target_shape[0]))

    masks = mask_reduce(masks, class_params)

    

    ImageId_ClassId = np.asarray([ fn+'_'+str(id) for id in range(1,5) ])

    for idx in range(masks.shape[-1]):

        submission = submission.append(pd.DataFrame([[ImageId_ClassId[idx], rle_encoding(masks[:,:,idx])]], columns = ["ImageId_ClassId", "EncodedPixels"]))

        

    return submission



load_dir = main_dir + 'test_images/'



for fn in tqdm_notebook(test_fns):

    sub = prediction_encoding(fn, load_dir, sub)

    gc.collect()

    

sub.head(10)



sub_sample = pd.read_csv(main_dir + 'sample_submission.csv')

sub_sample = sub_sample.drop(['EncodedPixels'], axis = 1)



submission = sub_sample.merge(sub, on = ['ImageId_ClassId'])

submission.to_csv('submission.csv', index = False)

submission.head(10)
