import numpy as np

import pandas as pd

import os

import gc



from sklearn.model_selection import train_test_split



from tqdm import tqdm_notebook



from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = [16, 10]

plt.style.use('fivethirtyeight')



import warnings

warnings.simplefilter('ignore')
import tensorflow as tf



from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam



from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout

from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
def read_image(location, size=(224, 224)):

    img = image.load_img(location, target_size=size)

    img = image.img_to_array(img)

    

    return img



class DataLoader(object):

    

    def __init__(self, image_size=(224, 224)):

        

        self.set_seed()

        self.image_size=image_size

        

        self.CATEGORIES = os.listdir("../input/plant-seedlings-classification/train/")

        self.N_CLASSES = len(self.CATEGORIES)

        print(f"Length of Categories : {self.N_CLASSES}")

        self.data_dir = "../input/plant-seedlings-classification/"

        self.train_dir = os.path.join(self.data_dir, "train")

        self.test_dir = os.path.join(self.data_dir, "test")

        self.sub = pd.read_csv("../input/plant-seedlings-classification/sample_submission.csv")



        print(f"Sub Shape : {self.sub.shape}")

    

    def set_seed(self, seed=13):

        

        self.seed = seed

        np.random.seed(self.seed)

    

    def plot_distribution(self, print_cat_wise=False):

        

        distribution = {}



        for category in self.CATEGORIES:

            num_samples = len(os.listdir(os.path.join(self.train_dir, category)))

            distribution[category] = num_samples

            if print_cat_wise:

                print(f"{category} has {num_samples} samples.")



        plt.figure(figsize=(24, 12))

        plt.xlabel("Category")

        plt.ylabel("Count")

        plt.title("Target Distribution")

        sns.barplot(list(distribution.keys()), list(distribution.values()))

        plt.show()

    

    def retrieve_data(self):

        

        # Making the datasets

        # Schema  : file_location | category | category_id

        

        self.train = []

        self.test = []

        self.class_names = {}

        

        for category_id, category in tqdm_notebook(enumerate(self.CATEGORIES)):

            category_path = os.path.join(self.train_dir, category)

            cur_cat_files = os.listdir(category_path)

            cur_cat_files = [[os.path.join(category_path, i), category, category_id] for i in cur_cat_files]

            

            if not self.class_names.get(category):

                self.class_names[category] = category_id

            

            self.train.extend(cur_cat_files)



        print(f"Total Train Samples : {len(self.train)}")

        self.train = pd.DataFrame(self.train, columns=['location', 'target', 'target_id'])

        

        for file in tqdm_notebook(os.listdir(self.test_dir)):

            cur_item = os.path.join(self.test_dir, file)

            self.test.append(cur_item)



        print(f"Total Test Samples : {len(self.test)}")

        self.test = pd.DataFrame(self.test, columns=['location'])

        

        self.split_data()

#         return self.train, self.test

    

    def split_data(self):



        self.train, self.valid, self.y_train, self.y_valid = train_test_split(self.train, self.train['target_id'], test_size=0.2, random_state=self.seed)

        print(f"Train Shape : {self.train.shape}\nValid Shape : {self.valid.shape}\n")

        

#         return self.get_data()

    

    def make_data_gens(self, batch_size=32):



        self.batch_size = batch_size

        

        self.train_gen = self.data_gen.flow_from_dataframe(

            dataframe=self.train, 

            x_col='location',

            y_col='target',

            batch_size=self.batch_size,

            seed=dataloader.seed,

            shuffle=False, # True

            class_mode='categorical',

            target_size=self.image_size,

        )



        self.valid_gen = self.data_gen.flow_from_dataframe(

            dataframe=self.valid, 

            x_col='location',

            y_col='target',

            batch_size=self.batch_size,

            seed=dataloader.seed,

            shuffle=False, # True

            class_mode='categorical',

            target_size=self.image_size,

        )



        self.test_gen = self.data_gen.flow_from_dataframe(

            dataframe=self.test, 

            x_col='location',

            y_col=None,

            batch_size=1, # 397 

            seed=dataloader.seed,

            shuffle=False,

            class_mode=None,

            target_size=self.image_size,

        )



        self.train_stepsize = self.train_gen.n // self.train_gen.batch_size

        self.valid_stepsize = self.valid_gen.n // self.valid_gen.batch_size

        

        return (self.train_gen, self.train_stepsize), (self.valid_gen, self.valid_stepsize), self.test_gen

    

    def get_data_gens(self):

        

        return self.train_gen, self.valid_gen, self.test_gen

    

    def get_stepsizes(self):

        

        return self.train_stepsize, self.valid_stepsize

    

    def get_data(self):

        

        return self.train, self.valid, self.test

    

    def set_datagen(self, data_gen=ImageDataGenerator(rescale=1./255)):

        self.data_gen = data_gen

        

    

    def set_image_size(self, image_size=(224, 224)):

        

        self.image_size = image_size

        

    def get_labels(self):

        

        return self.y_train, self.y_valid
dataloader = DataLoader(image_size=(224, 224))

dataloader.retrieve_data()

dataloader.plot_distribution()
class PreTrainedModels(object):

    

    def __init__(self):

        

        self.model_utils = {

            'resnet_50': {

                'model': ResNet50,

                'preprocessor': resnet_preprocess_input,

            },

            'vgg_16': {

                'model': VGG16,

                'preprocessor':vgg_preprocess_input,

            },

            'inception_v3': {

                'model': InceptionV3,

                'preprocessor': inception_preprocess_input,

            },

        }

        

        self.loss_history = {

            'resnet_50': {},

            'vgg_16': {},

            'inception_v3': {},

        }

    

    

    def tune_model(self, dataloader, choice, params):

        

        self.params = params

        

        self.pre_trained_model = self.model_utils[choice]['model']

        self.pre_trained_model_preprocessor = self.model_utils[choice]['preprocessor']

        

        self.datagen = ImageDataGenerator(preprocessing_function=self.pre_trained_model_preprocessor)

        dataloader.set_datagen(self.datagen)

        

        (self.train_gen, self.train_stepsize), (self.valid_gen, self.valid_stepsize), self.test_gen = dataloader.make_data_gens(self.params['batch_size'])

        

        # 1. Check if higher/lower input sizes help?

        

        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=3)

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=0.00001)

        

        print("\nDownloading & Compiling the model ... ")

        self.model_name = choice

        input_shape = (dataloader.image_size[0], dataloader.image_size[1], 3)

        if choice == 'vgg_16':



            file_path = "best_vgg16.hdf5"

            checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

            

            # VGG16

            model_vgg16 = self.pre_trained_model(include_top=False, input_shape=input_shape)



            # Training only the newly connected Dense Layer

            for layer in model_vgg16.layers:

                layer.trainable = False

                

            x = Flatten()(model_vgg16.output)

            x = Dense(512)(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            x = Dropout(0.8)(x)

            x = BatchNormalization()(x)

            x = Dense(dataloader.N_CLASSES, activation='softmax')(x)



            self.model = Model(inputs=model_vgg16.input, outputs=x)

            

        elif choice == 'resnet_50':



            file_path = "best_resnet50.hdf5"

            checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)



            # ResNet50

            model_resnet50 = self.pre_trained_model(include_top=False, input_shape=input_shape)



            for layer in model_resnet50.layers:

                layer.trainable = False



            x = Flatten()(model_resnet50.output)

            x = Dense(512)(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            x = Dropout(0.8)(x)

            x = BatchNormalization()(x)

            x = Dense(dataloader.N_CLASSES, activation='softmax')(x)



            self.model = Model(inputs=model_resnet50.input, outputs=x)

            

        elif choice == 'inception_v3':

            

            file_path = "best_inceptionv3.hdf5"  # {epoch:02d}

            checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

            

            # Inception_v3

            model_inception = self.pre_trained_model(include_top=False, input_shape=input_shape)



            for layer in model_inception.layers:

                layer.trainable = False



            x = Flatten()(model_inception.output)

            x = Dense(512)(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            x = Dropout(0.8)(x)

            x = BatchNormalization()(x)

            x = Dense(dataloader.N_CLASSES, activation='softmax')(x)



            self.model = Model(inputs=model_inception.input, outputs=x)

            

        else:

            return "Choose correct model."

        

        # lr = 1e-4

        optimizer = Adam() # learning_rate=lr

        self.model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])



        print("Compiled.")

        print(f"Fitting Model : {self.model_name} .. ", end='\n\n')

        

        hist = self.model.fit_generator(

            generator = self.train_gen,

            validation_data = self.valid_gen,

            steps_per_epoch = self.train_stepsize,

            validation_steps = self.valid_stepsize,

            epochs = self.params['epochs'],

            verbose = 1,

            callbacks=[checkpoint, early_stopping, reduce_lr],

        )

        

        self.loss_history[choice] = hist.history

        

        return self.model

    

    def plot_single_metric(self, choice):

        

        history = self.loss_history[choice]



        epoch_range = [i+1 for i, loss in enumerate(history['loss'])]

        xticks = range(0, max(epoch_range), 2)



        max_loss = max([max(history['loss']), max(history['val_loss'])])

        min_loss = min([min(history['loss']), min(history['val_loss'])])

        max_acc = max([max(history['accuracy']), max(history['val_accuracy'])])

        min_acc = min([min(history['accuracy']), min(history['val_accuracy'])])



        plt.figure(figsize=(18, 7))



        plt.subplot(1, 2, 1)

        plt.plot(epoch_range, history['loss'], color='red', label='Train')

        plt.plot(epoch_range, history['val_loss'], color='green', label='Valid')

        plt.xticks(xticks)

        plt.yticks(np.linspace(min_loss, max_loss, 10))

        plt.grid(False)

        plt.legend(loc='best')

        plt.xlabel('Epochs')

        plt.ylabel('Loss')

        plt.title(f"{self.model_name} | Loss Curve")



        plt.subplot(1, 2, 2)

        plt.plot(epoch_range, history['accuracy'], color='red', label='Train')

        plt.plot(epoch_range, history['val_accuracy'], color='green', label='Valid')

        plt.xticks(xticks)

        plt.yticks(np.linspace(min_acc, max_acc, 10))

        plt.grid(False)

        plt.legend(loc='best')

        plt.xlabel('Epochs')

        plt.ylabel('Accuracy')

        plt.title(f"{self.model_name} | Accuracy Curve")



        plt.tight_layout()

        plt.show()

        

    def plot_multiple_metric(self):

        

        history_1 = self.loss_history['vgg_16']

        history_2 = self.loss_history['resnet_50']

        history_3 = self.loss_history['inception_v3']



        epoch_range = [i+1 for i, loss in enumerate(history_1['loss'])]

        xticks = range(0, max(epoch_range), 2)



        plt.figure(figsize=(18, 18))



        # Valid Loss

        plt.subplot(2, 1, 1)

        plt.plot(epoch_range, history_1['val_loss'], color='red', label='VGG-16')

        plt.plot(epoch_range, history_2['val_loss'], color='green', label='Resnet-50')

        plt.plot(epoch_range, history_3['val_loss'], color='purple', label='Inception-v3')

        plt.xticks(xticks)

        plt.grid(False)

        plt.legend(loc='best')

        plt.xlabel('Epochs')

        plt.ylabel('Loss')

        plt.title(f"Validation Loss Comparision")



        # Valid Accuracy

        plt.subplot(2, 1, 2)

        plt.plot(epoch_range, history_1['val_accuracy'], color='red', label='VGG-16')

        plt.plot(epoch_range, history_2['val_accuracy'], color='green', label='Resnet-50')

        plt.plot(epoch_range, history_3['val_accuracy'], color='purple', label='Inception-v3')

        plt.xticks(xticks)

        plt.grid(False)

        plt.legend(loc='best')

        plt.xlabel('Epochs')

        plt.ylabel('Accuracy')

        plt.title(f"Validation Accuracy Comparision")



        plt.tight_layout()

        plt.show()

        

    def plot_confusion_matrix(self, dataloader, model, model_name):

        

        y_pred = model.predict_generator(dataloader.valid_gen)

        y_pred = np.argmax(y_pred, axis=1)



        con_mat = tf.math.confusion_matrix(labels=dataloader.valid_gen.classes, predictions=y_pred).numpy()

        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        con_mat_df = pd.DataFrame(con_mat_norm,

                             index = list(dataloader.class_names.keys()), 

                             columns = list(dataloader.class_names.keys()))



        fig = plt.figure(figsize=(16, 16))

        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)

        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')

        plt.title(f'{model_name} Confusion Matrix')



        plt.show()
params = {

    'batch_size': 64,

    'epochs': 15,

}
trans_models = PreTrainedModels()
model_inception = trans_models.tune_model(dataloader=dataloader, choice='inception_v3', params=params)

trans_models.plot_single_metric(choice='inception_v3')

trans_models.plot_confusion_matrix(dataloader=dataloader, model=model_inception, model_name='inception_v3')
model_inception.save("inception_v3.h5")
model_vgg = trans_models.tune_model(dataloader=dataloader, choice='vgg_16', params=params)

trans_models.plot_single_metric(choice='vgg_16')

trans_models.plot_confusion_matrix(dataloader=dataloader, model=model_vgg, model_name='vgg_16')
model_vgg.save("vgg_16.h5")
model_resnet = trans_models.tune_model(dataloader=dataloader, choice='resnet_50', params=params)

trans_models.plot_single_metric(choice='resnet_50')

trans_models.plot_confusion_matrix(dataloader=dataloader, model=model_resnet, model_name='resnet_50')
model_resnet.save("resnet_50.h5")
train = dataloader.train.copy()
# First Misclassification 



print("51% of Loose Silky-bent are mis-classified as Common wheat.")



plt.figure(figsize=(18, 18))

print("Loose Silky-bent")



for i in range(5):

    

    plt.subplot(1, 5, i+1)

    

    img = read_image(train['location'][train['target'] == 'Loose Silky-bent'].values[i], size=(400, 400))

    img = img/255.

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()



plt.show()



plt.figure(figsize=(18, 18))

print("Common wheat")



for i in range(5):

    

    plt.subplot(1, 5, i+1)

    

    img = read_image(train['location'][train['target'] == 'Common wheat'].values[i], size=(400, 400))

    img = img/255.

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()





plt.show()
# Second Misclassification 



print("16%% of Small-flowered Cranesbill are mis-classified as Black-grass.")

print("5%% of Small-flowered Cranesbill are mis-classified as Charlock.", end='\n\n')





plt.figure(figsize=(18, 18))

print("Small-flowered Cranesbill")



for i in range(5):

    

    plt.subplot(1, 5, i+1)

    

    img = read_image(train['location'][train['target'] == 'Small-flowered Cranesbill'].values[i], size=(400, 400))

    img = img/255.

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()



plt.show()



plt.figure(figsize=(18, 18))

print("Black-grass")



for i in range(5):

    

    plt.subplot(1, 5, i+1)

    

    img = read_image(train['location'][train['target'] == 'Black-grass'].values[i], size=(400, 400))

    img = img/255.

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()





plt.show()



plt.figure(figsize=(18, 18))

print("Charlock")



for i in range(5):

    

    plt.subplot(1, 5, i+1)

    

    img = read_image(train['location'][train['target'] == 'Charlock'].values[i], size=(400, 400))

    img = img/255.

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()





plt.show()
trans_models.plot_multiple_metric()
# y_pred = model_resnet.predict_generator(dataloader.test_gen, verbose=0)

# y_pred = np.argmax(y_pred, axis=1)

# print(y_pred.shape)



# id_to_target = {}

# for key, value in dataloader.class_names.items():

#     id_to_target[value] = key



# sub = dataloader.sub.copy()

# test = dataloader.test.copy()



# test['location'] = test['location'].apply(lambda x: x.split("/")[-1])

# test['species'] = np.nan

# test['species'] = y_pred

# test['species'] = test['species'].apply(lambda x: id_to_target[x])

# test.rename({

#     "location": "file",

# }, axis=1, inplace=True)



# # test.head()



# sub.drop(['species'], axis=1, inplace=True) 

# sub = pd.merge(sub, test, on=['file'], how='outer')

# # sub.shape

# # sub.head()

# sub.to_csv("sub.csv", index=False)