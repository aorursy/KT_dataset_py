import pandas as pd
import seaborn as sns
import numpy as np
# from bokeh.io import show, output_notebook
# from bokeh.plotting import figure
# from bokeh.models import HoverTool, BoxSelectTool #For enabling tools

# %matplotlib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
plt.style.use("dark_background")
# def plot_acc(history):
#     import matplotlib.pyplot as plt
#     history_dict = history.history
#     acc_values = history_dict['accuracy'] 
#     val_acc_values = history_dict['val_accuracy']
#     acc = history_dict['accuracy']
#     epochs = range(1, len(acc) + 1)
#     plt.plot(epochs, acc, 'bo', label='Training acc', color='green')
#     plt.plot(epochs, val_acc_values, 'b', label='Validation acc', color='red')
#     plt.title('Training and validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()


train_data = pd.read_csv("../input/images-of-gala/Images of Gala/train.csv")
test_data = pd.read_csv("../input/images-of-gala/Images of Gala/test.csv")

!mkdir new_train
!cp -r '../input/images-of-gala/Images of Gala/Train Images' new_train 
sns.countplot(train_data['Class'])
train_data['Class'].value_counts()
def percentage_coutn(vc_obj, title):
    '''
    This function prints the distribution of classes as percentage
    [param]vc_obj: This needs to be the value_counts().items() of a pandas column
    [param]title : Title that is printed like "Training Set, Testing Set"
    
    [example]:-
    percentage_coutn(train_data['Class'].value_counts().items(),"ACTUAL Training Data")
    '''
    le = []
    total_sum = 0
    for i in vc_obj:
        le.append(i)
        total_sum  = total_sum + i[1]
    print(title)
    for i in le:
        print(i[0], "-->", round(i[1]/total_sum,2),"%")
    print("##"*50)
    print("\n")
        

percentage_coutn(train_data['Class'].value_counts().items(),"ACTUAL Training Data")

kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=11)
##Split the data into training and testing

for train_ix, test_ix in kfold.split(train_data['Image'], train_data['Class']):
    #print(train_ix, test_ix)
    #print(type(train_ix))
    X_train, X_test = train_data['Image'].iloc[train_ix], train_data['Image'].iloc[test_ix]
    Y_train, Y_test = train_data['Class'].iloc[train_ix], train_data['Class'].iloc[test_ix]
    #print(Y_train.value_counts())
    #percentage_coutn(Y_train.value_counts().items(),"ACTUAL Training Data")
    #percentage_coutn(Y_test.value_counts().items(),"ACTUAL Training Data")
    print(len(X_train))
    print(len(X_test))
    break
Dataset_Train = pd.concat([X_train, Y_train], axis=1)

Dataset_Test = pd.concat([X_test, Y_test], axis=1)

percentage_coutn(Dataset_Train['Class'].value_counts().items(),"Training Data")
percentage_coutn(Dataset_Test['Class'].value_counts().items(),"Testing Data")
sns.countplot(Dataset_Train['Class'])
sns.countplot(Dataset_Test['Class'])
#import os
#import shutil
#li = []
#for i in train_data[train_data['Class']=='Decorationandsignage']['Image']:
    #print(i)
    
    #i2 = i.split(".")[0]
    #i2 = i2+"__2"
    #i2 = i2+".jpg"
    #bpath = "./new_train/Train Images/"
    #li.append(i2)
    #print(f'cp {bpath+i} {bpath+i2}')
    #os.system("cp {} {} -v", format(i,i2))
    #shutil.copyfile(f'{bpath+i}',f'{bpath+i2}')
    #os.system(f'cp {bpath+i} {bpath+i2} -v')
#df = pd.DataFrame(list(zip(li, li2)), columns =['Image', 'Class'])
# new_data = pd.concat([df, train_data])
# sns.countplot(new_data['Class'])
# new_data['Class'].value_counts()
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rotation_range=10,
#                                    width_shift_range=0.25,
#                                    height_shift_range=0.25,
#                                    shear_range=0.1,
#                                    zoom_range=0.25,
#                                    horizontal_flip=False)

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

# tpath = 'new_train/Train Images/'
# tepath = 'new_train/Train Images/'

# #tepath =  '../input/images-of-gala/Images of Gala/Test Images'
# #test_img = ImageList.from_df(test_data, path=tepath)

# # additional_aug=[*zoom_crop(scale=(0.75,1.25), do_rand=False), 
# #                 brightness(change=(0.1,0.1), 
# #                 contrast(scale=0.5)

# #additional_aug = [brightness(change=(0.5,0.5)), ]#contrast(scale=(0.3,0.3))]
# #additional_aug = [brightness(change=(0.1,0.35)),contrast(scale=(0.3,0.3))]


# TEST = Dataset_Test.drop(['Class'], axis=1)

# test_img = ImageList.from_df(TEST, path=tepath, )

# train_img = (ImageList.from_df(Dataset_Train, path=tpath, )
#         .split_by_rand_pct(0.03)
#         .label_from_df()
#         .add_test(test_img)
#         .transform(get_transforms(flip_vert=False, ), size= 80)
#         #.transform(get_transforms(flip_vert=False, xtra_tfms=additional_aug), size=100)
#         .databunch(path=tpath, bs=64, device= torch.device('cuda:0'))
#         .normalize()
#        )


tpath = 'new_train/Train Images/'
tepath = 'new_train/Train Images/'

#tepath =  '../input/images-of-gala/Images of Gala/Test Images'
#test_img = ImageList.from_df(test_data, path=tepath)
#TEST = Dataset_Test.drop(['Class'], axis=1)
# additional_aug=[*zoom_crop(scale=(0.75,1.25), do_rand=False), 
#                 brightness(change=(0.1,0.1), 
#                 contrast(scale=0.5)

#additional_aug = [brightness(change=(0.5,0.5)), ]#contrast(scale=(0.3,0.3))]
#additional_aug = [brightness(change=(0.1,0.35)),contrast(scale=(0.3,0.3))]


TEST = Dataset_Test.drop(['Class'], axis=1)

test_img = ImageList.from_df(TEST, path=tepath, )



np.random.seed(42)

train_img_80 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img)
        .transform(get_transforms(flip_vert=False, ), size= 80)
        #.transform(get_transforms(flip_vert=False, xtra_tfms=additional_aug), size=100)
        .databunch(path=tpath, bs=128, device= torch.device('cuda:0'))
        .normalize()
       )


tfms = [contrast(scale=(0.9,0.9)), brightness(change=(0.5,0.5))]


train_img_160 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img)
        #.transform(get_transforms(flip_vert=False, ), size= 160)
        .transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size=160)
        .databunch(path=tpath, bs=128, device= torch.device('cuda:0'))
        .normalize()
       )

train_img_224 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img)
        #.transform(get_transforms(flip_vert=False, ), size= 224)
        .transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size=224)
        .databunch(path=tpath, bs=100, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )
percentage_coutn(train_img_160.train_ds.inner_df['Class'].value_counts().items(),"Fast-AI Training")
percentage_coutn(train_img_160.valid_ds.inner_df['Class'].value_counts().items(),"Fast-AI Validation")
Dataset_Train.head(4)
for i in Dataset_Train.head(2)['Image']:

    img = imread('../input/images-of-gala/Images of Gala/Train Images/'+i)
#print(img.shape)
#plt.imshow(img)
#plt.show()

    reszied = resize(img,(224,224,3))
#print(reszied.shape)
    plt.imshow(reszied)
    plt.show()
for i in Dataset_Train.head(2)['Image']:
    def get_ex(): return open_image('new_train/Train Images/'+i)
    
    def plots_f(rows, cols, width, height, **kwargs):
        [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
            rows,cols,figsize=(width,height))[1].flatten())]
    #tfms = [ brightness(change=(0.5,0.5))]#contrast(scale=(0.9,0.9)),
    #tfms = get_transforms(max_rotate=180) 
#     tfms = [crop(size = 224, row_pct=1.0, col_pct= 1.0), 
#            get_transforms(max_rotate=180),
#            contrast(scale=(0.9,0.9))]
    
    #tfms = [perspective_warp(magnitude=(-0.2,0.2))]
    #tfms = [symmetric_warp(magnitude=(-0.2,0.2))] #padding_mode='zeros')
    #tfms = [tilt( direction = 2, magnitude = (0.4, 0.4))]
    tfms = [zoom(scale= 1.0)]
    plots_f(2, 2, 12, 6, size = 224)
#plots_f_1(1, 1, 12, 6, size=224)

#get_ex()
#train_img.show_batch(4, 4)


import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
def get_f1score(learn):
    '''
    prints the f1score of a fastai learn object.
    to use this meathod the cnn_learner needs to be initialized with metrics 
    Percision(), Recall() and in the following order
    metrics=[error_rate, accuracy, Precision(), Recall()]
    '''
    j=1
    for i in learn.recorder.metrics:
    #print(i)
        fn = (i[2] * i[3]) / (i[2] + i[3])
        print("Precision for batch -->",j,"is", (2*fn))
        j = j +1
def get_test_f1(learn):

     preds,y= learn.get_preds(ds_type=DatasetType.Test)
#train_img.classes

     num_cat = { 0:'Attire',  1:'Decorationandsignage',  2:'Food',  3:'misc'}
#sub = Dataset_Test

     sub = pd.DataFrame({'Class': np.argmax(preds,axis=-1)})
#sub['Class'] = np.argmax(preds,axis=-1)
     sub['Class'].replace(num_cat, inplace=True)

     print("F1 score on Test is      ", f1_score(Dataset_Test['Class'], sub['Class'], average='macro'))

     print("Accuracy score on Test is", accuracy_score(Dataset_Test['Class'], sub['Class'], ))

#learn2 = cnn_learner(train_img, models.resnet101, metrics=[error_rate, accuracy],model_dir="/tmp/model/") 
#learn2 = cnn_learner(train_img, models.resnet50, metrics=[error_rate, accuracy,],model_dir="/tmp/model/", )
                     #loss_func=torch.nn.CrossEntropyLoss())
 
                     #, opt_func=optimizer.RAdam(params, lr = 1e-5)) 
#learn2 = cnn_learner(train_img, models.resnet18, metrics=[error_rate, accuracy],model_dir="/tmp/model/") 
L_Res34_80 = cnn_learner(train_img_80, models.resnet34, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )

L_Res34_160 = cnn_learner(train_img_160, models.resnet34, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )

L_Res34_224 = cnn_learner(train_img_224, models.resnet34, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )
L_Res34_80.fit_one_cycle(2,)
L_Res34_80.recorder.plot_losses()
L_Res34_80.unfreeze()
L_Res34_80.lr_find()
#learn2.recorder.plot()
L_Res34_80.recorder.plot(suggestion=True)
#learn2.recorder.min_grad_lr
learn2.recorder.plot_lr(show_moms=True)
#lr = 2.75e-6
L_Res34_80.unfreeze()
L_Res34_80.fit_one_cycle(5, 1e-6 )
L_Res34_80.recorder.plot_losses()
#learn2.unfreeze()
#learn2.fit_one_cycle(30, 2.75e-9)
get_f1score(L_Res34_80)
get_test_f1(L_Res34_80)
L_Res34_160.freeze()
L_Res34_160.fit_one_cycle(2,)
L_Res34_160.unfreeze()
L_Res34_160.lr_find()
#learn2.recorder.plot()
L_Res34_160.recorder.plot(suggestion=True)
#learn2.recorder.min_grad_lr
L_Res34_160.unfreeze()
L_Res34_160.fit_one_cycle(5, 1e-7 )
L_Res34_160.recorder.plot_losses()
L_Res34_160.save("34-160-1")
L_Res34_160.load("34-160-1")
L_Res34_160.unfreeze()
L_Res34_160.fit_one_cycle(5, 1e-6 )
L_Res34_160.recorder.plot_losses()
get_f1score(L_Res34_160)
get_test_f1(L_Res34_160)
L_Res34_224.freeze()
L_Res34_224.fit_one_cycle(2, )
L_Res34_224.unfreeze()
L_Res34_224.lr_find()
#learn2.recorder.plot()
L_Res34_224.recorder.plot(suggestion=True)
L_Res34_224.unfreeze()
L_Res34_224.fit_one_cycle(5, 6.31e-7 )
L_Res34_224.recorder.plot_losses()
get_f1score(L_Res34_224)
get_test_f1(L_Res34_224)
L_Res50_80 = cnn_learner(train_img_80, models.resnet50, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )

L_Res50_160 = cnn_learner(train_img_160, models.resnet50, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )

L_Res50_224 = cnn_learner(train_img_224, models.resnet50, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )
L_Res50_80.freeze()
L_Res50_80.fit_one_cycle(2,)
L_Res50_80.unfreeze()
L_Res50_80.lr_find()
#learn2.recorder.plot()
L_Res50_80.recorder.plot(suggestion=True)
#learn2.recorder.min_grad_lr
L_Res50_80.unfreeze()
L_Res50_80.fit_one_cycle(5, 1e-6)
L_Res50_80.recorder.plot_losses()
get_f1score(L_Res50_80)
get_test_f1(L_Res50_80)
L_Res50_160.freeze()
L_Res50_160.fit_one_cycle(2,)
L_Res50_160.unfreeze()
L_Res50_160.lr_find()
#learn2.recorder.plot()
L_Res50_160.recorder.plot(suggestion=True)
L_Res50_160.unfreeze()
L_Res50_160.fit_one_cycle(5, 1.10e-7)
L_Res50_160.recorder.plot_losses()
L_Res50_160.save('Stage-1')
#L_Res50_160.load('Stage-1')
L_Res50_160.unfreeze()
L_Res50_160.fit_one_cycle(5, 1.10e-5)
L_Res50_160.recorder.plot_losses()
get_f1score(L_Res50_160)
get_test_f1(L_Res50_160)
L_Res50_160.summary()
# L_Res34_160_2 = cnn_learner(train_img_160, models.resnet34, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
#                     )

L_Res50_224 = cnn_learner(train_img_224, models.resnet50, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )
L_Res50_224.load('Stage-1')
L_Res50_224.freeze()
L_Res50_224.fit_one_cycle(2,)
L_Res50_224.unfreeze()
L_Res50_224.lr_find()
#learn2.recorder.plot()
L_Res50_224.recorder.plot(suggestion=True)
L_Res50_224.freeze()
L_Res50_224.fit_one_cycle(5,1e-6)
L_Res50_224.recorder.plot_losses()
L_Res50_224.save('Stage-2')
L_Res50_224.load('Stage-2')
L_Res50_224.freeze()
L_Res50_224.fit_one_cycle(5,1e-5)
L_Res50_224.recorder.plot_losses()
get_f1score(L_Res50_224)
get_test_f1(L_Res50_224)
tepath =  '../input/images-of-gala/Images of Gala/Test Images'
test_img = ImageList.from_df(test_data, path=tepath)
train_img_80.add_test(test_img)
train_img_160.add_test(test_img)
train_img_224.add_test(test_img)
def sub_df(learn, subno):
    from IPython.display import FileLink
    preds, y = learn.get_preds(DatasetType.Test)
# print(learn2.data.c2i)
    num_cat = { 0:'Attire',  1:'Decorationandsignage',  2:'Food',  3:'misc'}
    sub = test_data

    sub['Class'] = np.argmax(preds,axis=-1)
    sub['Class'].replace(num_cat, inplace=True)
    
    pd.DataFrame(sub).to_csv("./"+subno+".csv", index=False)
    FileLink("./"+subno+".csv")
    return sub

Sub_L_Res34_80 = sub_df(L_Res34_80, "L_Res34_80")
Sub_L_Res34_160 = sub_df(L_Res34_160, "L_Res34_160")
Sub_L_Res34_224 = sub_df(L_Res34_224, "L_Res34_224")

Sub_L_Res50_80 = sub_df(L_Res50_80, "L_Res50_80")
Sub_L_Res50_160 = sub_df(L_Res50_160, "L_Res50_160")
Sub_L_Res50_224 = sub_df(L_Res50_224, "L_Res50_224")
sublist = [Sub_L_Res34_80, Sub_L_Res34_160, Sub_L_Res34_224, 
          Sub_L_Res50_80, Sub_L_Res50_160, Sub_L_Res50_224]
l = [[], [], [], [], [], []]
for i,j in enumerate(sublist):
    #print(i)
    l[i].append(j['Class'].values)
len(l[1][0])
import numpy as np
from scipy import stats

a = np.array([#l[0][0],
              l[1][0],
              #l[2][0],
              l[3][0],
              #l[4][0],
              l[5][0],
             
             ])

m = stats.mode(a)
#print(m)
m[0][0]
nsub = test_data
nsub['Class'] = m[0][0]
#     pd.DataFrame(sub).to_csv("./"+subno+".csv", index=False)
#     FileLink("./"+subno+".csv")
#     return sub
from IPython.display import FileLink
pd.DataFrame(nsub).to_csv("./nsub2.csv", index=False)
FileLink("./nsub2.csv")
# learn2.show_results()
# learn2.show_results(DatasetType.Train)

tpath = 'new_train/Train Images/'
tepath = 'new_train/Train Images/'

#tepath =  '../input/images-of-gala/Images of Gala/Test Images'
#test_img = ImageList.from_df(test_data, path=tepath)

# additional_aug=[*zoom_crop(scale=(0.75,1.25), do_rand=False), 
#                 brightness(change=(0.1,0.1), 
#                 contrast(scale=0.5)

#additional_aug = [brightness(change=(0.5,0.5)), ]#contrast(scale=(0.3,0.3))]
#additional_aug = [brightness(change=(0.1,0.35)),contrast(scale=(0.3,0.3))]


TEST = Dataset_Test.drop(['Class'], axis=1)

test_img_2 = ImageList.from_df(TEST, path=tepath, )
tfms = [contrast(scale=(0.9,0.9)), brightness(change=(0.5,0.5))]

train_img_2 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.02)
        .label_from_df()
        .add_test(test_img_2)
        #.transform(get_transforms(flip_vert=False, ), size= 224)
        .transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size=160)
        .databunch(path=tpath, bs=100, device= torch.device('cuda:0'))
        .normalize()
        #.normalize(imagenet_stats)
       )

learn3 = cnn_learner(train_img_2, models.resnet101, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )# opt_func=RAdam)
learn3.freeze()
learn3.fit_one_cycle(2,)
get_f1score(learn3)
get_test_f1(learn3)
# #2 * (precision * recall) / (precision + recall)
# for i in learn3.recorder.metrics:
#     #print(i)
#     fn = (i[2] * i[3]) / (i[2] + i[3])
#     print("Precision is", (2*fn))
learn3.unfreeze()
learn3.lr_find()
#learn2.recorder.plot()
learn3.recorder.plot(suggestion=True)
#learn2.recorder.min_grad_lr
learn3.fit_one_cycle(5, slice(9.12e-5, 1e-5) )
learn3.recorder.plot_losses()
get_f1score(learn3)
get_test_f1(learn3)
learn3.save("Naruto-1")
learn3.load("Naruto-1")
actual_tepath =  '../input/images-of-gala/Images of Gala/Test Images'
actual_test_img = ImageList.from_df(test_data, path=actual_tepath)
train_img_2.add_test(actual_test_img)

res150 = sub_df(learn3, 'l3')

from IPython.display import FileLink
pd.DataFrame(res150).to_csv("./nsub5.csv", index=False)
FileLink("./nsub5.csv")
test_img
tpath = 'new_train/Train Images/'
TEST = Dataset_Test.drop(['Class'], axis=1)
test_img_v = ImageList.from_df(TEST, path=tpath, )
train_img_2.add_test(test_img_v)

learn3.fit_one_cycle(5, slice(9.12e-7, 1e-6) )
learn3.recorder.plot_losses()
#get_f1score(learn3)
get_test_f1(learn3)
actual_tepath =  '../input/images-of-gala/Images of Gala/Test Images'
actual_test_img = ImageList.from_df(test_data, path=actual_tepath)
train_img_2.add_test(actual_test_img)

res150_2 = sub_df(learn3, 'l3')

from IPython.display import FileLink
pd.DataFrame(res150_2).to_csv("./nsub4.csv", index=False)
FileLink("./nsub4.csv")
learn3.save("Matrix-1")
tpath = 'new_train/Train Images/'
tepath = 'new_train/Train Images/'

#tepath =  '../input/images-of-gala/Images of Gala/Test Images'
#test_img = ImageList.from_df(test_data, path=tepath)

# additional_aug=[*zoom_crop(scale=(0.75,1.25), do_rand=False), 
#                 brightness(change=(0.1,0.1), 
#                 contrast(scale=0.5)

#additional_aug = [brightness(change=(0.5,0.5)), ]#contrast(scale=(0.3,0.3))]
#additional_aug = [brightness(change=(0.1,0.35)),contrast(scale=(0.3,0.3))]


TEST = Dataset_Test.drop(['Class'], axis=1)

test_img_2 = ImageList.from_df(TEST, path=tepath, )
tfms = [contrast(scale=(0.9,0.9)), brightness(change=(0.5,0.5))]

train_img_3 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img_2)
        #.transform(get_transforms(flip_vert=False, ), size= 224)
        .transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size = 160)
        .databunch(path=tpath, bs= 64, device= torch.device('cuda:0'))
        #.normalize(imagenet_stats)
        .normalize(imagenet_stats)
       )
learn4 = cnn_learner(train_img_3, models.resnet152, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
learn4.freeze()
learn4.fit_one_cycle(2, )
get_f1score(learn4)
get_test_f1(learn4)
learn4.unfreeze()
learn4.lr_find()
#learn2.recorder.plot()
learn4.recorder.plot(suggestion=True)
#learn4.fit_one_cycle(5, slice(9.12e-7, 1e-6) )
learn4.unfreeze()
learn4.fit_one_cycle(10, 1e-7)
learn4.recorder.plot_losses()
learn4.save('Stage-1')
# {'Food':1,
#            'misc':2,
#            'Attire':3,
#            'Decorationandsignage':4,
    
# }

# {1:0.65660667,
#                 2:1.17682927,
#                 3:0.88453578,
#                 4:2.01312248,
# }

get_f1score(learn4)
get_test_f1(learn4)
from sklearn.utils import class_weight

tepath = 'new_train/Train Images/'
TEST = Dataset_Test.drop(['Class'], axis=1)

test_img_2 = ImageList.from_df(TEST, path=tepath, )
tpath = 'new_train/Train Images/'

tfms = [rand_zoom(scale=(1.,1.5)),rand_crop() ]
train_img_4 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img_2)
        #.transform(get_transforms(flip_vert=False, ), size= 160)
        .transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size = 224)
        .databunch(path=tpath, bs= 64, device= torch.device('cuda:0'))
        #.normalize(imagenet_stats)
        .normalize(imagenet_stats)
       )
weights = [0.88453578, 2.01312248, 0.65660667, 1.17682927  ]
class_weights = torch.FloatTensor(weights).cuda()
C = torch.nn.CrossEntropyLoss(weight=class_weights)
learn5 = cnn_learner(train_img_4, models.resnet152, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    ) # loss_func= C)
learn5.mixup()
learn5.freeze()
learn5.fit_one_cycle(2, )
learn5.recorder.plot_losses()
learn5.unfreeze()
learn5.fit_one_cycle(5, 1e-5)
learn5.recorder.plot_losses()
learn5.save("Stage-1")
learn5.load("Stage-1")
learn5.fit_one_cycle(5, 1e-5)
learn5.recorder.plot_losses()
learn5.save("Stage-2")
learn5.load("Stage-2")
learn5.fit_one_cycle(15, 1e-5)
learn5.recorder.plot_losses()
learn5.save("Stage-3")
learn5.load("Stage-3")
learn5.fit_one_cycle(2, 1e-5)
learn5.recorder.plot_losses()
get_f1score(learn5)
get_test_f1(learn5)
#loss_func


# tpath = 'new_train/Train Images/'

# train_img_5 = (ImageList.from_df(train_data, path=tpath, )
#         .split_by_rand_pct(0.05, seed=1995)
#         .label_from_df()
#         .add_test(test_img_2)
#         .transform(get_transforms(flip_vert=False, ), size= 160)
#         #.transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size = 224)
#         .databunch(path=tpath, bs= 32, device= torch.device('cuda:0'))
#         #.normalize(imagenet_stats)
#         .normalize(imagenet_stats)
#        )
tepath = 'new_train/Train Images/'
TEST = Dataset_Test.drop(['Class'], axis=1)

test_img_2 = ImageList.from_df(TEST, path=tepath, )

tpath = 'new_train/Train Images/'

train_img_5 = (ImageList.from_df(Dataset_Train, path=tpath, )
        .split_by_rand_pct(0.05, seed=1995)
        .label_from_df()
        .add_test(test_img_2)
        .transform(get_transforms(flip_vert=False, ), size= 160)
        #.transform(get_transforms(flip_vert=False, xtra_tfms=tfms), size = 224)
        .databunch(path=tpath, bs= 32, device= torch.device('cuda:0'))
        #.normalize(imagenet_stats)
        .normalize()
       )
learn6 = cnn_learner(train_img_5, models.resnext50_32x4d, metrics=[error_rate, accuracy, Precision(), Recall()],model_dir="/tmp/model/",
                    )
learn6.freeze()
learn6.fit_one_cycle(2)
learn6.unfreeze()
learn6.lr_find()
#learn2.recorder.plot()
learn6.recorder.plot(suggestion=True)
learn6.unfreeze()
learn6.fit_one_cycle( 4, 1.91E-05)
learn6.recorder.plot_losses()

get_f1score(learn6)
get_test_f1(learn6)

actual_tepath =  '../input/images-of-gala/Images of Gala/Test Images'
actual_test_img = ImageList.from_df(test_data, path=actual_tepath)
train_img_5.add_test(actual_test_img)

desnet = sub_df(learn6, 'l3')

from IPython.display import FileLink
pd.DataFrame(desnet).to_csv("./nsub7.csv", index=False)
FileLink("./nsub7.csv")
# sub_test_path =  '../input/images-of-gala/Images of Gala/Test Images'
# sub_test_img = ImageList.from_df(test_data, path=sub_test_path, )

# train_img.add_test(sub_test_img)
# preds, y = learn2.get_preds(DatasetType.Test)
# print(learn2.data.c2i)
# num_cat = { 0:'Attire',  1:'Decorationandsignage',  2:'Food',  3:'misc'}
# sub = test_data

# sub['Class'] = np.argmax(preds,axis=-1)
# sub['Class'].replace(num_cat, inplace=True)
# pd.DataFrame(sub).to_csv("./sub26.csv", index=False)
# from IPython.display import FileLink
# FileLink('./sub26.csv')
interp = ClassificationInterpretation.from_learner(learn3)
interp.plot_confusion_matrix()
interp.most_confused()
interp.plot_top_losses(9,)
# interp = ClassificationInterpretation.from_learner(learn2)
# losses,idxs = interp.top_losses(10)
# for p in train_img.valid_ds.x.items[idxs]:
#     print(p)
# for p in train_img.train_ds.x.items[idxs]:
#     print(p)
# losses,idxs = interp.top_losses()
# top_train_losses = train_img.train_ds.x[idxs]
# top_train_losses
# top_valid_losses

# from fastai.widgets import *
# ds, idxs = DatasetFormatter().from_toplosses(learn2, n_imgs=100,  )

# fd = ImageCleaner(ds, idxs,tpath )


##This returns the path of the image which has the hightest loss

# losses,idxs = interp.top_losses(10)
# #for p in train_img_2.valid_ds.x[idxs]:
# for p in idxs:
#     print(train_img_2.valid_ds.x.items[p])
# from fastai.widgets import *
# ds, idxs = DatasetFormatter().from_toplosses(learn3, n_imgs=6, ds_type=DatasetType.Train )

# fd = ImageCleaner(ds, idxs,tpath )
#Actual path of the image where the image lies
# for p in idxs:
#     print(train_img_2.train_ds.x.items[p])
#     print(open_image(train_img_2.train_ds.x.items[p]))
# preds, y = learn3.get_preds(DatasetType.Test)
# test_idxs = np.where(np.logical_and(preds>=0.4, preds<=0.6))[0]
# for i in test_idxs:
#     print(test_data.loc[i]['Image'])#, valid_data.loc[i]['Class'])
    #print(i)
#(np.abs(preds-0.5))
# #import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import to_categorical
# from keras.preprocessing import image
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator

# from keras.applications import ResNet50

# train_image = []
# for i, x in tqdm(Dataset_Train['Image'].iteritems()):
# #for i in tqdm(range(Dataset_Train.shape[0])):
#     img = image.load_img('../input/images-of-gala/Images of Gala/Train Images/'+Dataset_Train['Image'][i], target_size=(80,80))
#     img = image.img_to_array(img)
#     img = img/255
#     train_image.append(img)
# X = np.array(train_image)

# cat_num = { 'Attire':0,  'Decorationandsignage':1,  'Food':2,  'misc':3}
# Y = Dataset_Train
# Y = Y['Class'].replace(cat_num)
# #Y = Y['Class'].values
# Y = to_categorical(Y)

# test_image = []
# for i, x in tqdm(Dataset_Test['Image'].iteritems()):
# #for i in tqdm(range(Dataset_Train.shape[0])):
#     img = image.load_img('../input/images-of-gala/Images of Gala/Train Images/'+Dataset_Test['Image'][i], target_size=(80,80))
#     img = image.img_to_array(img)
#     img = img/255
#     test_image.append(img)
# X_test = np.array(test_image)

# cat_num = { 'Attire':0,  'Decorationandsignage':1,  'Food':2,  'misc':3}
# Y_test = Dataset_Test
# Y_test = Y_test['Class'].replace(cat_num)
# #Y = Y['Class'].values
# Y_test = to_categorical(Y_test)

# model = Sequential()

# resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet',
#                   input_shape=(80,80,3)))

# model.add(Dense(4, activation='softmax'))

# model.layers[0].trainable = False

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     #featurewise_center=True,
#     #featurewise_std_normalization=True,
#     #rotation_range=20,
#     #width_shift_range=0.2,
#     #height_shift_range=0.2,
#     horizontal_flip=False)

# datagen.fit(X)

# valid = ImageDataGenerator(
#     #featurewise_center=True,
#     #featurewise_std_normalization=True,
#     #rotation_range=20,
#     #width_shift_range=0.2,
#     #height_shift_range=0.2,
#     horizontal_flip=False)

# valid.fit(X_test)

# model.summary()

def plot_acc(history):
    import matplotlib.pyplot as plt
    history_dict = history.history
    acc_values = history_dict['accuracy'] 
    val_acc_values = history_dict['val_accuracy']
    acc = history_dict['accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# his= model.fit_generator(datagen.flow(X, Y, batch_size=32),
#                          validation_data=valid.flow(X_test, Y_test),
#                     steps_per_epoch=len(X) / 32, epochs=10)

# plot_acc(his)
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50
train_datagen = ImageDataGenerator(rescale=1./255,
                                   #shear_range=0.20,
                                   #zoom_range=0.20,
                                   #validation_split=0.20,   
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255,
                                   #shear_range=0.20,
                                   #zoom_range=0.20,
                                   #validation_split=0.20,   
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(dataframe=Dataset_Train,
                                                    directory='new_train/Train Images/',
                                                    x_col='Image',
                                                    y_col='Class',
                                                    #has_ext=True,
                                                    seed=42,
                                                    target_size=(224, 224),
                                                    batch_size=16,
                                                    #subset='training',    
                                                    shuffle=True,
                                                    class_mode='categorical')
valid_generator = valid_datagen.flow_from_dataframe(dataframe=Dataset_Test,
                                                    directory='new_train/Train Images/',
                                                    x_col='Image',
                                                    y_col='Class',
                                                    #has_ext=True,
                                                    seed=42,
                                                    target_size=(224, 224),
                                                    batch_size=16,
                                                    #subset='training',    
                                                    shuffle=True,
                                                    class_mode='categorical')
conv_base = ResNet50(include_top=False, input_shape=(224, 224,3))

#conv_base = ResNet50(include_top=False, pooling='avg', weights='imagenet',)

conv_base.summary()
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base.trainable
conv_base.trainable="False"
his= model.fit_generator(
                         train_generator,
                         validation_data=valid_generator,
                         validation_steps=50,
                         steps_per_epoch=len(Dataset_Train)/32, 
                         epochs=2,
                        )
plot_acc(his)
