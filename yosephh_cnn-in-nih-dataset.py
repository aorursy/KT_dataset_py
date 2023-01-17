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
# reading the data
data = pd.read_csv("../input/Data_Entry_2017.csv")
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
data = pd.read_csv('../input/Data_Entry_2017.csv')
data = data[data['Patient Age']<100] #removing datapoints which having age greater than 100
data_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input', 'images*', '*', '*.png'))}
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
                             color_mode = 'rgb',
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
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
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]
multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 10, 
                                  callbacks = callbacks_list)
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
#!pip install lime eli5 scikit-image xgboost --upgrade
#!pip install scikit-learn==0.20
#!pip install shap==0.31
import shap
import numpy as np

# select a set of background examples to take an expectation over
from scipy import misc
image_path=train_df.iloc[5,12]
background = misc.imread(image_path)




# explain predictions of the model on three images
e = shap.DeepExplainer(multi_disease_model, t_x[0:5])
# ...or pass tensors directly
#e = shap.DeepExplainer((multi_disease_model.layers[0].input, model.layers[-1].output), test_X)
shap_values = e.shap_values(t_x[0:5])
shap.image_plot(shap_values, -t_x[1:5])
shap_values[0][0,:]
shap.initjs()
shap.force_plot(e.expected_value[1], shap_values[1][0,0])
def convert_to_1channel(images):
    return images

def new_predict_fn(images):
    image = convert_to_1channel(images)
    return multi_disease_model.predict(images)

from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from skimage.util import img_as_float
from sklearn.preprocessing import Normalizer
foo=t_x[0, :,:,0]
#foo = np.expand_dims(foo, axis=0)
transformer = Normalizer().fit(foo)
foo2=transformer.transform(foo)
foo3=img_as_float(foo2)
#foo3 = np.expand_dims(foo3, axis=-1)
#foo3 = np.expand_dims(foo3, axis=0)
#foo3 = np.stack((foo3,)*3, axis=-1)
#foo3.reshape(128,128,3)

print(foo3.shape)
print('shape')
print(len(foo))


#img = image.load_img(image_path, target_size=(128, 128))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = inc_net.preprocess_input(x)

import lime
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
#foo[foo < -1.0] = 0.1
#foo[foo > 1.0] = 0.1
#print(foo.shape)
explanation = explainer.explain_instance(foo2,new_predict_fn, top_labels=5, hide_color=0,batch_size='None')
explanation

pics= [test_X[x,:,:,0] for x in range(100)]
res=[]
                  
for i in pics:
    transformer= Normalizer().fit(i)
    foo2=transformer.transform(i)
    foo3=img_as_float(foo2)
    res.append(foo3)
from lime.wrappers.scikit_image import SegmentationAlgorithm
random_seed = 1
segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200,
                                  ratio=0.2, random_seed=random_seed)
def bulk_exp(explainer, images):
    return [explainer.explain_instance(image, new_predict_fn, top_labels=5,
                                        hide_color=0, num_samples=1000
                                        )   for image in images]
def calc_identity(img1, img2, verbose=True):
    dis = np.array([np.array_equal(img1[i],img2[i]) for i in range(len(img1))])
    total = dis.shape[0]
    true = np.sum(dis)
    score = (total-true)/total
    if verbose:
        print('true: ',true, 'wrong: ', total-true, 'total: ', total)
    return score*100, true, total

def calc_separability(exp):
    wrong = 0
    for i in range(exp.shape[0]):
        for j in range(exp.shape[0]):
            if i == j:
                continue
            eq = np.array_equal(exp[i],exp[j])
            if eq:
                wrong = wrong + 1
    total = exp.shape[0]
    score = 100*abs(wrong)/(total**2-total)
    print('true: ', total**2-total-wrong, 'wrong: ', wrong, 'total: ', total**2-total)
    return wrong,total,total**2-total,score
import sklearn
import sklearn.cluster
from sklearn.manifold import TSNE


def calc_stability2(exp, labels):
    total = labels.shape[0]
    label_values = np.unique(labels)
    print(label_values)
    n_clusters = label_values.shape[0]
    #init = np.array([[np.average(exp[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, n_jobs=5, random_state=1)
    #ct=TSNE(n_components=2)
    ct.fit(exp)
    print(ct.labels_)
    error = np.sum(np.abs(labels-ct.labels_))
    if error/total > 0.5:
        error = total-error
    return error, total
print(calc_stability2(picList,ansNum))
pics_y
picList=[]
for i in range(100):
    picList.append(test_X[i,:,:,:].flatten())
len(t_x)
X_train = np.array(res).reshape(len(t_x),-1)
X_train.shape
[test_X[0:6:,:,1]].shape

m=shap_values[0]
m.shape
np.array([2,3,4]).shape
#pics_y= np.array([test_Y[x] for x in range(5)])

ansList=[]
ansNum=[]
j=0
for i in range(100):
    result=np.where(test_Y[i]==1.0)
    #result=np.where(test_Y[i]=1.0,test_Y[i])
    if result[0].tolist() not in ansList:
        print("here")
        ansList.append(result[0].tolist())
        ansNum.append(j)
        j=j+1
    else:
        index=ansList.index(result[0].tolist())
        ansNum.append(index)
        
    
ansNum=np.array(ansNum)
print(ansList)
ansNum
#np.where(np.isclose(test_Y[0], 1.0))
ansList2=[np.where(test_Y[65]==1.0)[0].tolist()]
if np.where(test_Y[66]==1.0)[0].tolist() not in ansList2:
    ansList2.append(np.where(test_Y[66]==1.0)[0].tolist())
    print('Noo')
else:
    print('Yes')
    
print(ansList2)
%time exps1 = bulk_exp(explainer, res)
%time exps2 = bulk_exp(explainer, res)
plt.imshow(test_X[0,:,:,1])
no_superpixels = 5
def get_imgs_from_exps(exps):
    return np.array([exp.get_image_and_mask(exp.top_labels[0], positive_only=True,
                                            num_features=no_superpixels, hide_rest=True)[0] for exp in exps])
%time imgs1 = get_imgs_from_exps(exps1)
%time imgs2 = get_imgs_from_exps(exps2)
type(imgs1)
calc_identity(imgs1,imgs2)
calc_separability(imgs1)
#SHAP testing
#%time imgs3 = get_imgs_from_exps(exps1[0])
#%time imgs4 = get_imgs_from_exps(exps2[0])
shap_values3 = e.shap_values(t_x[0:100])
shap_values2 = e.shap_values(t_x[0:100])

calc_identity(shap_values3,shap_values2)
calc_separability(shap_values[0])
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[1], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.imshow(foo3)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
!pip install anchor-exp





import anchor
explainer2= anchor.AnchorImage()
explanation = explainer2.explain_instance(foo3,new_predict_fn)
print(test_X[0, :,:,0].shape)
print(test_X[0, :,:,0].shape)
plt.imshow(test_X[0, :,:,0])
multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  (test_X, test_Y), 
                                  epochs = 5, 
                                  callbacks = callbacks_list)
# load the best weights
multi_disease_model.load_weights(weight_path)
pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)
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
sickest_idx = np.argsort(np.sum(test_Y, 1)<1)
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, 
                                                                  test_Y[idx]) 
                             if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, 
                                                                  test_Y[idx], pred_Y[idx]) 
                             if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png')
import shap
import numpy as np

# select a set of background examples to take an expectation over
from scipy import misc
image_path=train_df.iloc[5,12]
background = misc.imread(image_path)




# explain predictions of the model on three images
e = shap.DeepExplainer(multi_disease_model, t_x[0].reshape(1, 128, 128, 1))
# ...or pass tensors directly
e = shap.DeepExplainer((multi_disease_model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_test[1:5])