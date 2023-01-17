import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from glob import glob

import os

import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker

sns.set_style('whitegrid')

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
import os

inpath = "../input/224_v2/224_v2/"

print(os.listdir("../input/224_v2/224_v2"))
# reading the data

data = pd.read_csv("../input/224_v2/224_v2/Data_Entry_2017.csv")

data.head()
data.shape
data.describe()
#drop unused columns

data = data[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]



# removing the rows which have patient_age >100

total = len(data)

print('No. of rows before removing rows having age >100 : ',len(data))

data = data[data['Patient Age']<100]

print('No. of rows after removing rows having age >100 : ',len(data))

print('No. of datapoints having age > 100 : ',total-len(data))
# rows having no. of disease

data['Labels_Count'] = data['Finding Labels'].apply(lambda text: len(text.split('|')) if(text != 'No Finding') else 0)
label_counts = data['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
#plt.figure(figsize=(20,15))

sns.FacetGrid(data,hue='Patient Gender',size=5).map(sns.distplot,'Patient Age').add_legend()

plt.show()
g = sns.factorplot(x="Patient Age", col="Patient Gender",data=data, kind="count",size=10, aspect=0.8,palette="GnBu_d");

g.set_xticklabels(np.arange(0,100));

g.set_xticklabels(step=10);

g.fig.suptitle('Age distribution by sex',fontsize=22);

g.fig.subplots_adjust(top=.9)
f, axarr = plt.subplots(7, 2, sharex=True,figsize=(15, 20))

pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

df = data[data['Finding Labels'] != 'No Finding']

i=0

j=0

x=np.arange(0,100,10)

for pathology in pathology_list :

    index = []

    for k in range(len(df)):

        if pathology in df.iloc[k]['Finding Labels']:

            index.append(k)

    g=sns.countplot(x='Patient Age', hue="Patient Gender",data=df.iloc[index], ax=axarr[i, j])

    axarr[i, j].set_title(pathology)   

    g.set_xlim(0,90)

    g.set_xticks(x)

    g.set_xticklabels(x)

    j=(j+1)%2

    if j==0:

        i=(i+1)%7

f.subplots_adjust(hspace=0.3)
for pathology in pathology_list :

    data[pathology] = data['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
plt.figure(figsize=(15,10))

gs = gridspec.GridSpec(8,1)

ax1 = plt.subplot(gs[:7, :])

ax2 = plt.subplot(gs[7, :])

data1 = pd.melt(data,

             id_vars=['Patient Gender'],

             value_vars = list(pathology_list),

             var_name = 'Category',

             value_name = 'Count')

data1 = data1.loc[data1.Count>0]

g=sns.countplot(y='Category',hue='Patient Gender',data=data1, ax=ax1, order = data1['Category'].value_counts().index)

ax1.set( ylabel="",xlabel="")

ax1.legend(fontsize=20)

ax1.set_title('X Ray partition (total number = 121120)',fontsize=18);



data['Nothing']=data['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)



data2 = pd.melt(data,

             id_vars=['Patient Gender'],

             value_vars = list(['Nothing']),

             var_name = 'Category',

             value_name = 'Count')

data2 = data2.loc[data2.Count>0]

g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2)

ax2.set( ylabel="",xlabel="Number of decease")

ax2.legend('')

plt.subplots_adjust(hspace=.5)
f, (ax1,ax2) = plt.subplots( 2, figsize=(15, 10))



df = data[data['Follow-up #']<15]

g = sns.countplot(x='Follow-up #',data=df,palette="GnBu_d",ax=ax1);



ax1.set_title('Follow-up distribution');

df = data[data['Follow-up #']>14]

g = sns.countplot(x='Follow-up #',data=df,palette="GnBu_d",ax=ax2);

x=np.arange(15,100,10)

g.set_ylim(15,450)

g.set_xlim(15,100)

g.set_xticks(x)

g.set_xticklabels(x)

f.subplots_adjust(top=1)
df=data.groupby('Finding Labels').count().sort_values('Patient ID',ascending=False)

df1=df[['|' in index for index in df.index]].copy()

df2=df[['|' not in index for index in df.index]]

df2=df2[['No Finding' not in index for index in df2.index]]

df2['Finding Labels']=df2.index.values

df1['Finding Labels']=df1.index.values
f, ax = plt.subplots(sharex=True,figsize=(15, 10))

sns.set_color_codes("pastel")

g=sns.countplot(y='Category',data=data1, ax=ax, order = data1['Category'].value_counts().index,color='b',label="Multiple Pathologies")

sns.set_color_codes("muted")

g=sns.barplot(x='Patient ID',y='Finding Labels',data=df2, ax=ax, color="b",label="Simple Pathology")

ax.legend(ncol=2, loc="center right", frameon=True,fontsize=20)

ax.set( ylabel="",xlabel="Number of decease")

ax.set_title("Comparaison between simple or multiple decease",fontsize=20)      

sns.despine(left=True)
#we just keep groups of pathologies which appear more than 30 times

df3=df1.loc[df1['Patient ID']>30,['Patient ID','Finding Labels']]



for pathology in pathology_list:

    df3[pathology]=df3.apply(lambda x: x['Patient ID'] if pathology in x['Finding Labels'] else 0, axis=1)



df3.head(20)
#'Hernia' has not enough values to figure here

df4=df3[df3['Hernia']>0]  # df4.size == 0

#remove 'Hernia' from list

pat_list=[elem for elem in pathology_list if 'Hernia' not in elem]



f, axarr = plt.subplots(13, sharex=True,figsize=(10, 140))

i=0

for pathology in pat_list :

    df4=df3[df3[pathology]>0]

    if df4.size>0:  #'Hernia' has not enough values to figure here

        axarr[i].pie(df4[pathology],labels=df4['Finding Labels'], autopct='%1.1f%%')

        axarr[i].set_title('main desease : '+pathology,fontsize=14)   

        i +=1

data = pd.read_csv(inpath + 'Data_Entry_2017.csv')

data = data[data['Patient Age']<100] #removing datapoints which having age greater than 100

data_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join(inpath, 'images', '*.png'))}

print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])

data['path'] = data['Image Index'].map(data_image_paths.get)

data['Patient Age'] = data['Patient Age'].map(lambda x: int(x))

data.sample(3)
data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

from itertools import chain

all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))

all_labels = [x for x in all_labels if len(x)>0]

print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

data.sample(3)
# keep at least 1000 cases

MIN_CASES = 1000

all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]

print('Clean Labels ({})'.format(len(all_labels)), 

      [(c_label,int(data[c_label].sum())) for c_label in all_labels])

# since the dataset is very unbiased, we can resample it to be a more reasonable collection

# weight is 0.04 + number of findings

sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2

sample_weights /= sample_weights.sum()

data = data.sample(40000, weights=sample_weights)



label_counts = data['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)

# creating vector of diseases

data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

print(data.iloc[0:5]['disease_vec'])

print(data.shape)

data[['Finding Labels','Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax','disease_vec']].head()
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(data, 

                                   test_size = 0.25, 

                                   random_state = 2018,

                                   stratify = data['Finding Labels'].map(lambda x: x[:4]))

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)

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
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen

train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 32)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 256) # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 1024)) # one big batch
t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)

    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 

                             if n_score>0.5]))

    c_ax.axis('off')
from keras.applications.mobilenet import MobileNet

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.models import Sequential

mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 

                                 include_top = False, weights = None)

multi_disease_model = Sequential()

multi_disease_model.add(mobilenet_model)

multi_disease_model.add(GlobalAveragePooling2D())

multi_disease_model.add(Dropout(0.5))

multi_disease_model.add(Dense(512))

multi_disease_model.add(Dropout(0.5))

multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))

multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',

                           metrics = ['accuracy','binary_accuracy', 'mae'])

multi_disease_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import keras.callbacks as kcall

weight_path="{}_weights.best.hdf5".format('xray_class')



class LossHistory(kcall.Callback):

    def on_train_begin(self, logs={}):

        self.batch_losses = []

        self.batch_acc = []

        self.epochs_losses = []

        self.epochs_acc = []

        self.epochs_val_losses = []

        self.epochs_val_acc = []

    

    def on_batch_end(self, batch, logs={}):

        self.batch_losses.append(logs.get('loss'))

        self.batch_acc.append(logs.get('acc'))

    

    def on_epoch_end(self, epoch, logs={}):

        self.epochs_losses.append(logs.get('loss'))

        self.epochs_acc.append(logs.get('acc'))

        self.epochs_val_losses.append(logs.get('val_loss'))

        self.epochs_val_acc.append(logs.get('val_acc'))



history = LossHistory()



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=3)

callbacks_list = [checkpoint, early, history]
multi_disease_model.fit_generator(train_gen, 

                                  steps_per_epoch=100,

                                  validation_data = (test_X, test_Y), 

                                  epochs = 10, 

                                  callbacks = callbacks_list)
#Plots of losses

plt.figure(figsize=[16,8])

plt.subplot(2, 2, 1)

plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)





plt.xlabel('# of batches trained')

plt.ylabel('Training loss')



plt.title('1) Training loss vs batches trained')

plt.legend()

plt.ylim(0,1)

plt.grid(True)





plt.subplot(2, 2, 2)

plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)

plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)



plt.xlabel('# of epochs trained')

plt.ylabel('Training loss')



plt.title('2) Training loss vs epochs trained')

plt.legend()

plt.ylim(0,0.5)

plt.grid(True)





#Plots of acc

plt.figure(figsize=[16,8])

plt.subplot(2, 2, 1)

plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)



plt.xlabel('# of batches trained')

plt.ylabel('Training accuracy')



plt.title('3) Training accuracy vs batches trained')

plt.legend(loc=3)

plt.ylim(0,1.1)

plt.grid(True)





plt.subplot(2, 2, 2)

plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)

plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)



plt.xlabel('# of epochs trained')

plt.ylabel('Training accuracy')



plt.title('4) Training accuracy vs epochs trained')

plt.legend(loc=3)

plt.ylim(0.5,1)

plt.grid(True)
for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):

    print('%s: %2.2f%%' % (c_label, s_count))
pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_labels):

    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
multi_disease_model.fit_generator(train_gen, 

                                  steps_per_epoch = 100,

                                  validation_data =  (test_X, test_Y), 

                                  epochs = 5, 

                                  callbacks = callbacks_list)
# load the best weights

multi_disease_model.load_weights(weight_path)
pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)
#Plots of losses

plt.figure(figsize=[16,8])

plt.subplot(2, 2, 1)

plt.plot(history.batch_losses,'b--',label='Training',alpha=0.7)





plt.xlabel('# of batches trained')

plt.ylabel('Training loss')



plt.title('Training loss vs batches trained')

plt.legend()

plt.ylim(0,1.5)

plt.grid(True)





plt.subplot(2, 2, 2)

plt.plot(history.epochs_losses,'b--',label='Training',alpha=0.7)

plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)



plt.xlabel('# of epochs trained')

plt.ylabel('Training loss')



plt.title('Training loss vs epochs trained')

plt.legend()

plt.ylim(0,1.5)

plt.grid(True)





#Plots of acc

plt.figure(figsize=[16,8])

plt.subplot(2, 2, 1)

plt.plot(history.batch_acc,'b--',label= 'Training', alpha=0.7)



plt.xlabel('# of batches trained')

plt.ylabel('Training accuracy')



plt.title('Training accuracy vs batches trained')

plt.legend(loc=3)

plt.ylim(0,1.1)

plt.grid(True)





plt.subplot(2, 2, 2)

plt.plot(history.epochs_acc,'b--',label= 'Training', alpha=0.7)

plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)



plt.xlabel('# of epochs trained')

plt.ylabel('Training accuracy')



plt.title('Training accuracy vs epochs trained')

plt.legend(loc=3)

plt.ylim(0,1.1)

plt.grid(True)
# look at how often the algorithm predicts certain diagnoses 

for c_label, p_count, t_count in zip(all_labels, 

                                     100*np.mean(pred_Y,0), 

                                     100*np.mean(test_Y,0)):

    print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_labels):

    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('trained_net.png')
row = 670

print(*['{})'.format(i) for i in range(13)], sep='\t')

print(*test_Y[row], sep='\t', end='\tTrue\n')

sumatoria = np.sum(pred_Y[row])

print(sumatoria)

pred = ["%.2f"%item for item in pred_Y[row]]

print(*pred, sep='\t', end='\tPredict\n\n')

print(*['{}){}'.format(i,l) for i,l in enumerate(all_labels)], sep='\n')
pred_Y[660]
sickest_idx = np.argsort(np.sum(test_Y, 1)<1)

fig, m_axs = plt.subplots(10, 4, figsize = (16, 32))

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):

    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')

    stat_str = [n_class[:4] for n_class, n_score in zip(all_labels, 

                                                                  test_Y[idx]) 

                             if n_score>0.5]

    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, 

                                                                  test_Y[idx], pred_Y[idx]) 

                             if (n_score>0.5) or (p_score>0.5)]

    c_ax.set_title('{} Dx: '.format(idx)+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png')