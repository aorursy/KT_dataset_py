import numpy as np

import os

from sklearn.metrics import confusion_matrix

import seaborn as sn; sn.set(font_scale=1.4)

from sklearn.utils import shuffle           

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from tqdm import tqdm

import pandas as pd

from sklearn import decomposition

from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys

import imgaug as ia

import imgaug.augmenters as iaa

from collections import Counter

print(f'tensor version:{tf.__version__}')

print(f'sys python/os version:{sys.version}')



import albumentations

from PIL import Image, ImageOps, ImageEnhance

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.augmentations import functional as F

from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,DualTransform, IAAAffine,IAAPerspective

)

from albumentations.augmentations import functional as Func

from tensorflow_addons.optimizers import SWA,CyclicalLearningRate, AdamW

from tensorflow.keras.optimizers import Adam,SGD
class_names = ['mountain', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
data_root = '/kaggle/input/sceneimage/scene-image/'

def load_data(path):

    datasets = ['./seg_train/seg_train','./seg_valid/seg_valid' ]

    output = []

    

    # Iterate through training and test sets

    for data in datasets:

        dataset = data_root + data

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        # Iterate through each folder corresponding to a category

        for folder in os.listdir(dataset):

            label = class_names_label[folder]

            

            # Iterate through each image in our folder

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(dataset, folder), file)

                

                # Open and resize the img

                image = cv2.imread(img_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(image)

                labels.append(label)

                

        images = np.array(images, dtype = 'float32')

        labels = np.array(labels, dtype = 'int32')   

        

        output.append((images, labels))



    return output
test_root= data_root +'seg_test/seg_test/'

test_path = os.listdir(test_root)

test_images_ = []

for img_path in test_path:

    image = cv2.imread(test_root+img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE) 

    test_images_.append(image)

test_images = np.array(test_images_, dtype = 'float32')

test_images /= 255.0
(train_images, train_labels), (valid_images, valid_labels) = load_data(data_root)
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
n_train = train_labels.shape[0]

n_valid = valid_labels.shape[0]



print ("Number of training examples: {}".format(n_train))

print ("Number of validation examples: {}".format(n_valid))

print ("Each image is of size: {}".format(IMAGE_SIZE))
_, train_counts = np.unique(train_labels, return_counts=True)

_, valid_counts = np.unique(valid_labels, return_counts=True)

pd.DataFrame({'train': train_counts,

                    'valid': valid_counts}, 

             index=class_names

            ).plot.bar()

plt.show()
plt.pie(train_counts,

        explode=(0, 0, 0, 0, 0) , 

        labels=class_names,

        autopct='%1.1f%%')

plt.axis('equal')

plt.title('Proportion of each observed category')

plt.show()
# Good practice: scale the data

train_images = train_images / 255.0 

valid_images = valid_images / 255.0
# Visualize the data

def display_random_image(class_names, images, labels):

    """

        Display a random image from the images array and its correspond label from the labels array.

    """

    

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title('Image #{} : '.format(index) + class_names[labels[index]])

    plt.show()
display_random_image(class_names, train_images, train_labels)
def display_examples(class_names, images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[labels[i]])

    plt.show()
display_examples(class_names, train_images, train_labels)
# model goes here

!pip install efficientnet

import efficientnet.tfkeras as efn

from tensorflow_addons.layers import GroupNormalization

recipes = []



# recipes.append({'backbone':efn.EfficientNetB4,"batch_size":36

#         ,'name':'Efb4','val_sel':0,'input_shape':(150,150,3), 'group':True})

# recipes.append({'backbone':efn.EfficientNetB5, "batch_size":22

#         ,'name':'Efb5','val_sel':1,'input_shape':(150,150,3), 'group':True})

# recipes.append({'backbone':efn.EfficientNetB6, "batch_size":18

#         ,'name':'Efb6','val_sel':2,'input_shape':(150,150,3), 'group':True})

# recipes.append({'backbone':efn.EfficientNetB7, "batch_size":14

#         ,'name':'Efb7','val_sel':3,'input_shape':(150,150,3), 'group':True})



recipes.append({'backbone':efn.EfficientNetB0,"batch_size":52

        ,'name':'Efb0','val_sel':4,'input_shape':(150,150,3), 'group':False})

recipes.append({'backbone':efn.EfficientNetB1, "batch_size":48

        ,'name':'Efb1','val_sel':5,'input_shape':(150,150,3), 'group':False})

# recipes.append({'backbone':efn.EfficientNetB2, "batch_size":44

#         ,'name':'Efb2','val_sel':6,'input_shape':(150,150,3), 'group':False})

# recipes.append({'backbone':efn.EfficientNetB3, "batch_size":40

#         ,'name':'Efb3','val_sel':7,'input_shape':(150,150,3), 'group':False})





def build_model(backbone= efn.EfficientNetB4 ,input_shape = (128,128,3)

                ,use_imagenet = 'imagenet', group=True):

    base_model = backbone(input_shape=input_shape,weights=use_imagenet,include_top= False)

    if group==True:

        print('replace bn to gn')

        for i, layer in enumerate(base_model.layers):

            if "_bn" in layer.name:

                base_model.layers[i]=GroupNormalization(groups=16)

                

    x = base_model.output

    x = tf.keras.layers.GlobalAvgPool2D(name='gap')(x)

    predictions = tf.keras.layers.Dense(len(class_names),activation='softmax'

                    ,name='prediction')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    return model
strategy = tf.distribute.MirroredStrategy(  

                tf.config.experimental.list_logical_devices('GPU')) 

gpus= strategy.num_replicas_in_sync  # 가용 gpu 수 

print('use gpus:',gpus)
class GridMask(DualTransform):

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = Func.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')
def strong_aug(p=0.5):

    return Compose([

        RandomRotate90(),

        Flip(),

        Transpose(),

        OneOf([

            MotionBlur(p=0.2),

            MedianBlur(blur_limit=3, p=0.1),

            Blur(blur_limit=3, p=0.1),

        ], p=0.2),

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

        OneOf([

            OpticalDistortion(p=0.3),

            GridDistortion(p=0.1),

            IAAPiecewiseAffine(p=0.3),

        ], p=0.2),

        OneOf([

            RandomBrightnessContrast(),

        ], p=0.3),

        

        OneOf([   

            GridMask(num_grid=(4,4), rotate=(-15,15), mode=0),

            GridMask(num_grid=(10,10), rotate=(-15,15), mode=0),

            GridMask(num_grid=(30,30), rotate=(-15,15), mode=0),

            ],p = 0.1)

    ], p=p)

augmentation = strong_aug(p=0.8)



class ScenGenerator(tf.keras.utils.Sequence):

    def __init__(self,imgs,labels,input_shape,batchsize,state='Train'):

        self.imgs = imgs

        print('Data Check', Counter(labels))

        self.labels = labels

        self.batchsize = batchsize

        self.input_shape = input_shape

        self.label_num = 5

        self.state = state

        self.on_epoch_end()

        self.len = -(-imgs.shape[0]//self.batchsize) 



    def __len__(self):

        return self.len



    def __getitem__(self, index):

        # batch size 만큼 index를 뽑음

        batch_idx = self.idx[index*self.batchsize:(index+1)*self.batchsize] 

        h,w,ch = self.input_shape

        X = np.zeros((len(batch_idx), h,w,ch)) #batch

        y = np.zeros((len(batch_idx), self.label_num))

        

        for i in range(self.batchsize):

            if self.state == 'Train':

                data = {"image":self.imgs[batch_idx[i]]}

                augmented = augmentation(**data)

                X[i, :, :, ] = augmented["image"]

            else:

                X[i, :, :, ] = self.imgs[batch_idx[i]]

            y[i, :] = tf.keras.utils.to_categorical(self.labels[batch_idx[i]],num_classes=self.label_num)

        return X, y

            

    def smooth_labels(self,labels, factor=0.1): # mix up augmentation 사용시 사용

        labels *= (1 - factor)

        print(len(labels))

        labels += (factor / labels.shape[0])

        return labels

    

    def on_epoch_end(self):

        self.idx = np.arange(self.imgs.shape[0]) #init

        if self.state == 'Train':

            cnt = Counter(self.labels)

            max_cnt = max(cnt.values())

            over_idxs=[]

            over_idxs.append(self.idx)

            for k,v in cnt.items():

                dif = max_cnt - v

                if dif>0:

                    over_idx = np.random.choice(np.array(np.where(self.labels==k)).squeeze(),dif)

                    over_idxs.append(over_idx)

            self.idx = np.concatenate(over_idxs)

            np.random.shuffle(self.idx) 

            sla = pd.Series(self.labels)

            print('Balanced Sampling Train Set',Counter(sla.loc[self.idx]))

        else:

            self.idx = np.tile(self.idx,2)



            

    

class CutMixGenerator(tf.keras.utils.Sequence):

    def __init__(self, generator1, generator2, cut_p = 0.2, maxcut= 0.5, mixup_p = 0.2, maxmix = 0.5):

        self.generator1 = generator1

        self.generator2 = generator2

        self.cut_p = cut_p

        self.mixup_p = mixup_p

        self.batch_size = self.generator1.batchsize

        self.maxcut = maxcut

        self.maxmix = maxmix

        self.on_epoch_end()  

        

    def __len__(self):

        return self.generator1.__len__()

        

    def get_rand_bbox(self,width, height, l):

        wcut = np.random.random()*l

        hcut = np.random.random()*l

        r_w = np.int(width * wcut)

        r_h = np.int(height * hcut)

        x = np.random.randint(width - r_w)

        y = np.random.randint(height - r_h)

        return x, y, r_w, r_h

    

    def smooth_labels(self,labels, factor=0.1):

        labels *= (1 - factor)

        labels += (factor / labels.shape[0])

        return labels



    def cutmix(self,X1, X2, y1, y2):

        width = X1.shape[1]

        height = X1.shape[0]

        x, y, r_w, r_h = self.get_rand_bbox(width, height, self.maxcut)

        X1[ y:y+r_h, x:x+r_w, :] = X2[ y:y+r_h, x:x+r_w, :]

        ra = (r_w*r_h) / (width*height)

        ysm1 = self.smooth_labels(y1)

        ysm2 = self.smooth_labels(y2)

        return X1, (ysm1*(1.0-ra)) + (ysm2*ra)

    

    def mixup(self, X1, X2, y1, y2):

        X = np.zeros(X1.shape)

        ra = np.random.random()*self.maxmix

        X = X1*(1-ra) + X2*ra

        ysm1 = self.smooth_labels(y1)

        ysm2 = self.smooth_labels(y2)

        return X,(ysm1*(1.0-ra)) + (ysm2*ra)

    

    def __getitem__(self, index):

        Data, Target = self.generator1.__getitem__(index)

        cutmix_idx = np.random.choice(np.arange(self.batch_size),int(self.batch_size*self.cut_p), replace=False)

        

        for idx in cutmix_idx:

            srcidx = np.random.randint(self.generator2.__len__())

            orgD, orgT= Data[idx,:], Target[idx,:]

            srcD, srcT = self.generator2.__getitem__(srcidx)

            

            mD, mT = self.cutmix(orgD,srcD[0], orgT, srcT[0])

            Data[idx,:], Target[idx,:]= mD, mT

        mixup_idx = np.random.choice(np.arange(self.batch_size),int(self.batch_size*self.mixup_p), replace=False)

        

        for idx in mixup_idx:

            if idx in cutmix_idx:

                continue

            srcidx = np.random.randint(self.generator2.__len__())

            orgD, orgT = Data[idx,:], Target[idx,:]

            srcD, srcT = self.generator2.__getitem__(srcidx)

            mD, mT1 = self.mixup(orgD,srcD[0], orgT, srcT[0])

            Data[idx,:], Target[idx,:] = mD, mT    

        

        return Data, Target



    def on_epoch_end(self):

        self.generator1.on_epoch_end()

        self.generator2.on_epoch_end()
recipe_sel = 0

recipe = recipes[recipe_sel]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2

                                                  , random_state=recipe_sel, stratify =train_labels )



train_gen_base = ScenGenerator(X_train, y_train, input_shape=recipe['input_shape']

                         ,batchsize=recipe['batch_size'],state='Train')



cut_src_gen = ScenGenerator(X_train, y_train, input_shape=recipe['input_shape']

                         ,batchsize=1,state='CutAndMixSrc')

train_gen = CutMixGenerator(

        generator1=train_gen_base,

        generator2=cut_src_gen,

        mixup_p = 0.5,

        cut_p=0.5

    )



val_gen = ScenGenerator(X_val, y_val, input_shape=recipe['input_shape']

                       ,batchsize=recipe['batch_size'],state='Valid')

test_gen = ScenGenerator(valid_images, valid_labels, input_shape=recipe['input_shape']

                        ,batchsize=recipe['batch_size'],state='Test')
tt = train_gen.__getitem__(0)

fig = plt.figure(figsize=(12,12))

fig.suptitle('Generator Check Augmentation!! ')

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(tt[0][i])

    plt.xlabel(class_names[np.argmax(tt[1][i])]+str([round(p,2) for p in tt[1][i]]))

plt.show()

print(tt[0].shape)
# train the model

train_model_list = []

historys =[]

skip_train = False

for i, recipe in enumerate(recipes):

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2

                                                      , random_state=recipe["val_sel"], stratify =train_labels )

    

    train_gen_base = ScenGenerator(X_train, y_train, input_shape=recipe['input_shape']

                             ,batchsize=recipe['batch_size'],state='Train')



    cut_src_gen = ScenGenerator(X_train, y_train, input_shape=recipe['input_shape']

                             ,batchsize=1,state='CutAndMixSrc')

    train_gen = CutMixGenerator(

            generator1=train_gen_base,

            generator2=cut_src_gen,

            mixup_p = 0.05,

            cut_p=0.1

        )

    

    val_gen = ScenGenerator(X_val, y_val, input_shape=recipe['input_shape']

                           ,batchsize=recipe['batch_size'],state='Valid')



    model_name = recipe["name"] + '_val_' + str(recipe["val_sel"]) 

    best_save_model_file = model_name + '.h5' 

    train_model_list.append(best_save_model_file)

    if skip_train == True and os.path.isfile(best_save_model_file):

        print(f'skip train and reuse weight {best_save_model_file}')

        continue

        

    print('best_save_model_file path : ',best_save_model_file)



    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5

                ,verbose=1, patience=2,min_lr=0.00005,min_delta=0.001, mode='max') 

    check_point=tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy',verbose=1

                                                   ,filepath=best_save_model_file,save_best_only=True,mode='max') 

    estop = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='max')

    



    with strategy.scope():   

        model = build_model(backbone= recipe['backbone']

        ,input_shape=recipe['input_shape'], use_imagenet = 'imagenet', group=recipe['group'])

        model.compile(optimizer=Adam(learning_rate=0.0005*gpus)

        ,loss='categorical_crossentropy', metrics=['accuracy'])

        

        if os.path.isfile(best_save_model_file):

            print(f'weights exist so re-train file path {best_save_model_file}')

            model.load_weights(best_save_model_file)        

        history = model.fit(train_gen ,validation_data=val_gen,epochs=30,callbacks =[reduce_lr,check_point, estop], verbose=1)

    historys.append(history)        

        

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5

                ,verbose=1, patience=2,min_lr=0.00005,min_delta=0.001, mode='max') 

    estop = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='max')



    with strategy.scope():   

        opt = Adam(lr=0.00005*gpus)

        opt = SWA(opt)

        model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])           

        swa_history = model.fit(train_gen ,validation_data=val_gen,epochs=30,callbacks =[reduce_lr,check_point, estop], verbose=1)

    historys.append(swa_history)  
def plot_accuracy_loss(history, text=None):

    """

enumerate    Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(10,5))

    if text is not None:

        fig.suptitle(text)

    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['accuracy'],'bo--', label = "acc")

    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")

    plt.title("train_acc vs val_acc")

    plt.ylabel("accuracy")

    plt.xlabel("epochs")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

    plt.title("train_loss vs val_loss")

    plt.ylabel("loss")

    plt.xlabel("epochs")



    plt.legend()

    plt.show()
historys[1]
# plot accuracy loss

for idx, history in enumerate(historys):

    if idx%2==0:

        plot_accuracy_loss(history, recipes[idx//2]['name'] )

    else:

        plot_accuracy_loss(history, recipes[idx//2]['name'] +' w/swa')
def tta_predict(model, imgs):

    pred_val = model.predict(imgs,verbose=1)

    pred_val_lr = model.predict(np.flip(imgs,axis=2), verbose=1)

    return (pred_val+pred_val_lr)/2.0



def average_ensemble(recipes, imgs, test=False):

    res = np.zeros((len(imgs),len(class_names)))

    for recipe in recipes:

        model = build_model(backbone= recipe['backbone']

                            ,input_shape=recipe['input_shape'], use_imagenet = None, group=recipe['group'])

        path = recipe["name"] + '_val_' + str(recipe["val_sel"]) +'.h5'

        model.load_weights(path)

        pred_val = tta_predict(model, imgs)

        if test == False:

            np.save(path.replace('.h5','.npy'),pred_val)

        else:

            np.save(path.replace('.h5','_test.npy'),pred_val)



        res += pred_val

    res /= len(recipes)

    return res

    

pred_val = average_ensemble(recipes,valid_images)
pred_val_label = np.argmax(pred_val,axis=1)

print(f'accuracy_score : {accuracy_score(valid_labels, pred_val_label)}')

print(f'f1_score : {f1_score(valid_labels, pred_val_label,average="macro")}')

cm = confusion_matrix(valid_labels, pred_val_label)



df_cm = pd.DataFrame(cm, index = [i for i in class_names],

                  columns = [i for i in class_names])

plt.figure(figsize = (10,7))



sn.heatmap(df_cm, annot=True,  fmt='d')

plt.ylabel('Actual')

plt.xlabel('Predict')

plt.show()
preds = []

for recipe in recipes:

    path = recipe["name"] + '_val_' + str(recipe["val_sel"]) +'.npy'

    pred_val = np.load(path)

    preds.append(pred_val)
ensemble_arr = np.stack(preds,axis=-1)

xin = tf.keras.layers.Input((5,len(recipes)))

x = tf.keras.layers.Convolution1D(1,kernel_size=1,activation='linear',use_bias=False)(xin)

x = tf.keras.layers.Reshape(target_shape=(5,))(x)

wensemble_model = tf.keras.Model(inputs=xin, outputs=x)

wensemble_model.summary()
wensemble_model.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mse')

wensemble_model.fit(x=ensemble_arr,y=tf.keras.utils.to_categorical(valid_labels)

                    , epochs=8000, batch_size=len(ensemble_arr),verbose=0)
wpred = wensemble_model.predict(ensemble_arr)

wensemble_model.get_weights()
pred_val_label = np.argmax(wpred,axis=1)

print(f'accuracy_score : {accuracy_score(valid_labels, pred_val_label)}')

print(f'f1_score : {f1_score(valid_labels, pred_val_label,average="macro")}')

cm = confusion_matrix(valid_labels, pred_val_label)



df_cm = pd.DataFrame(cm, index = [i for i in class_names],

                  columns = [i for i in class_names])

plt.figure(figsize = (10,7))



sn.heatmap(df_cm, annot=True,  fmt='d')

plt.ylabel('Actual')

plt.xlabel('Predict')

plt.show()
def display_examples_wprd(class_names, images, labels, prds):

    """

        Display 25 images from the images array with its corresponding labels

    """

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[labels[i]] + ',' + class_names[prds[i]], fontsize=10 )

    plt.show()
sample_idx = np.random.randint(0, valid_images.shape[0], 25)

sample_images = valid_images[sample_idx]

sample_labels = valid_labels[sample_idx]

sample_prd = pred_val_label[sample_idx]

display_examples_wprd(class_names, sample_images, sample_labels, sample_prd)
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    """

        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels

    """

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]

    org_labels = test_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"

    display_examples_wprd(class_names,  mislabeled_images,org_labels, mislabeled_labels)
print_mislabeled_images(class_names, valid_images, valid_labels, pred_val_label )
def tta_predict_feature(recipe, imgs):

    model = build_model(backbone= recipe['backbone']

                        ,input_shape=recipe['input_shape'], use_imagenet = None, group=recipe['group'])

    path = recipe["name"] + '_val_' + str(recipe["val_sel"]) +'.h5'

    model.load_weights(path)

    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('gap').output)

    pred_val = feature_model.predict(imgs,verbose=1)

    pred_val_lr = feature_model.predict(np.flip(imgs,axis=2), verbose=1)

    return (pred_val+pred_val_lr)/2.0
valid_features = tta_predict_feature(recipes[0],valid_images)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, random_state=0)

tsne_results = tsne.fit_transform(valid_features)
sn.set(style="whitegrid")

scplot = sn.scatterplot(x= tsne_results[:,0], y=tsne_results[:,1]

               , hue=[class_names[i] for  i in valid_labels], hue_order = class_names,legend=False)
base_model = recipes[0]['backbone'](input_shape=recipe['input_shape'],weights='imagenet',include_top= False)

x = base_model.output

x = tf.keras.layers.GlobalAvgPool2D(name='gap')(x)

notrain_model_features = tf.keras.Model(inputs=base_model.input, outputs=x)
no_valid_features = notrain_model_features.predict(valid_images,verbose=1)
from sklearn.manifold import TSNE

notrain_tsne = TSNE(n_components=2, verbose=1, random_state=0)

notrain_tsne_results = notrain_tsne.fit_transform(no_valid_features)
sn.scatterplot(x= notrain_tsne_results[:,0], y=notrain_tsne_results[:,1]

               , hue=[class_names[i] for  i in valid_labels], hue_order = class_names,legend=False)
print(test_images.shape)

plt.imshow(test_images[0])

plt.show()
pred_test = average_ensemble(recipes,test_images, test=True)
preds = []

for recipe in recipes:

    path = recipe["name"] + '_val_' + str(recipe["val_sel"]) +'_test.npy'

    pred_test = np.load(path)

    preds.append(pred_test)
ensemble_arr = np.stack(preds,axis=-1)

wpred = wensemble_model.predict(ensemble_arr)
pred_test_label = np.argmax(wpred,axis=1)
pred_names = [class_names[label] for label in pred_test_label]

sub_df = pd.DataFrame(test_path, columns=['image'])

sub_df['label']=pred_names

sub_df.to_csv('submission_4team.csv',index=None,header=None)
fig = plt.figure(figsize=(10,10))

fig.suptitle("Some examples of images of the dataset", fontsize=16)

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel(pred_names[i])

plt.show()