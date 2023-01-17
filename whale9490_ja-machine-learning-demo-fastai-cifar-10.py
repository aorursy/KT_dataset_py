%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import numpy as np

import torch

from torch import nn

import fastai

from IPython.display import display

from fastai.vision import *

import matplotlib.pyplot as plt
# GPU使用がONになっていることの確認。

print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
N_EPOCHS = 40

LEARNING_RATE = 0.1

BATCH_SIZE = 256

SHOW_BATCH_FIGSIZE = (10, 10)
# 画像データをダウンロード。

path = untar_data(URLs.CIFAR)
path.ls()
(path/'train').ls()
(path/'train'/'bird').ls()[:10]
def show_image_of_category(category, n=4):

    f, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for i in range(n):

        open_image((path/'train'/category).ls()[i]).show(ax=axes[i], title=category)
show_image_of_category('bird')

show_image_of_category('deer')

show_image_of_category('frog')

show_image_of_category('horse')

show_image_of_category('cat')

show_image_of_category('dog')

show_image_of_category('ship')

show_image_of_category('truck')

show_image_of_category('airplane')

show_image_of_category('automobile')
class Category:

    def __init__(self, name, label):

        self.name = name

        self.label = label



# animal / vehicle の二値分類タスクをやってみるのであれば、以下をアンコメントする。

# CATEGORIES = [

#     Category('bird',       'animal'),

#     Category('deer',       'animal'),

#     Category('frog',       'animal'),

#     Category('horse',      'animal'),

#     Category('cat',        'animal'),

#     Category('dog',        'animal'),

#     Category('ship',       'vehicle'),

#     Category('truck',      'vehicle'),

#     Category('airplane',   'vehicle'),

#     Category('automobile', 'vehicle'),

# ]



CATEGORIES = [

    Category('bird',       'bird'),

    Category('deer',       'deer'),

    Category('frog',       'frog'),

    Category('horse',      'horse'),

    Category('cat',        'cat'),

    Category('dog',        'dog'),

    Category('ship',       'ship'),

    Category('truck',      'truck'),

    Category('airplane',   'airplane'),

    Category('automobile', 'automobile'),

]



N_LABELS = len(set([c.label for c in CATEGORIES]))
dfs = []



for folder_name in ['train', 'test']:

    for cat in CATEGORIES:

        file_names = os.listdir(path/folder_name/cat.name)

        _df = pd.DataFrame()

        _df['name'] = f'{folder_name}/{cat.name}/' + pd.Series(file_names)

        _df['category'] = cat.name

        _df['label'] = cat.label

        _df['is_valid'] = folder_name == 'test'

        dfs.append(_df)



all_data_df = pd.concat(dfs).reset_index(drop=True)

all_data_df
def get_data_df(all_data_df, bird, deer, frog, horse, cat, dog, ship, truck, airplane, automobile):

    category_n_samples_map = {

        'bird': bird,

        'deer': deer,

        'frog': frog,

        'horse': horse,

        'cat': cat,

        'dog': dog,

        'ship': ship,

        'truck': truck,

        'airplane': airplane,

        'automobile': automobile,

    }

    

    train_df = (all_data_df

     .query('not is_valid')

     .groupby('category')

     .apply(lambda x: x.sample(n=category_n_samples_map[x.name], random_state=42)))



    val_df = all_data_df.query('is_valid').sample(frac=1, random_state=42)  # CIFAR-10のtest imageはカテゴリごとに1000ずつ。



    print(f'トレーニングデータ数={len(train_df)}')

    print(f'バリデーションデータ数={len(val_df)}')



    data_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    return data_df
def show_results(learn):

#     learn.recorder.plot_losses()

    learn.recorder.plot_metrics()

    learn.show_results(figsize=SHOW_BATCH_FIGSIZE, rows=3)
def get_val_predictions(learn, data_df):

    id_class_map = {i: c for i, c in enumerate(learn.data.classes)}

    

    pred, gt_class = learn.get_preds(ds_type=DatasetType.Valid)

    pred_class = np.argmax(pred, axis=1)

    

    val_df = data_df.query('is_valid').copy()

    val_df['pred'] = pd.Series(pred_class.numpy()).map(id_class_map).to_numpy()

    

    return val_df
def show_accuracy(df):

    print('--- 正解率 ---')

    

    _correct_filter = df['label'] == df['pred']

    

    # 全体

    n_correct = _correct_filter.sum()

    n_all = len(df)

    print(f'{"all":10}: {n_correct / n_all * 100:>4.1f}% ({n_correct} / {n_all})')

    

    # カテゴリ毎

    for category in CATEGORIES:

        _category_filter = df['category'] == category.name



        n_all = _category_filter.sum()

        n_correct = (_category_filter & _correct_filter).sum()

        

        top_preds = df.loc[_category_filter, 'pred'].value_counts()[:3] / n_all

        top_preds_texts = [f'{_cat:10}: {_ratio * 100:>4.1f}%' for _cat, _ratio in top_preds.items()]

        print(f'{category.name:10}: {n_correct / n_all * 100:>4.1f}% ({top_preds_texts[0]}, {top_preds_texts[1]}, {top_preds_texts[2]})')
# CIFAR-10ではトレーニング用画像として種類毎に5000画像ずつ用意されているが、今回はデモ用ということで2000画像ずつだけ用いることにした。

# ※ バリデーション用画像は1000画像ずつ用意されていて、これはすべて使った。

data_df_1 = get_data_df(all_data_df,

                        bird=2000, deer=2000, frog=2000, horse=2000, cat=2000, dog=2000,

                        ship=2000, truck=2000, airplane=2000, automobile=2000)

display(data_df_1)



data_1 = (ImageList.from_df(data_df_1, path=path, convert_mode='RGB')

          .split_from_df(col='is_valid')

          .label_from_df(cols='label')

          .databunch(bs=BATCH_SIZE)

          .normalize())

data_1.show_batch(figsize=SHOW_BATCH_FIGSIZE, rows=3, hide_axis=False)
model_1 = nn.Sequential(

    conv_layer(3, 8, stride=2),   # 16

    conv_layer(8, 16, stride=2),  # 8

    conv_layer(16, 32, stride=2), # 4

    conv_layer(32, 16, stride=2), # 2

    conv_layer(16, N_LABELS, stride=2), # 1

    Flatten()      # remove (1,1) grid

)



# metrics=accuracy を指定することで、学習が進むにつれて精度(正解率)がどう変化するかを追うことができる。

# loss_func(損失関数)は機械学習において非常に重要な概念だが、今回は説明しない。

learn_1 = Learner(data_1, model_1, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn_1.fit_one_cycle(N_EPOCHS, max_lr=LEARNING_RATE)
show_results(learn_1)
val_df_1 = get_val_predictions(learn_1, data_df_1)

show_accuracy(val_df_1)
data_df_2 = get_data_df(all_data_df,

                        bird=2000, deer=2000, frog=2000, horse=2000, cat=2000, dog=2000,

                        ship=2000, truck=2000, airplane=2000, automobile=2000)

display(data_df_2)



data_2 = (ImageList.from_df(data_df_2, path=path, convert_mode='RGB')

          .split_from_df(col='is_valid')

          .label_from_df(cols='label')

          .databunch(bs=BATCH_SIZE)

          .normalize())

data_2.show_batch(figsize=SHOW_BATCH_FIGSIZE, rows=3, hide_axis=False)
# ResNetっぽいモデルを作成する。

model_2 = nn.Sequential(

    conv_layer(3, 16, stride=2),

    res_block(16),

    res_block(16),

    conv_layer(16, 32, stride=2),

    res_block(32),

    res_block(32),

    conv_layer(32, 64, stride=2),

    res_block(64),

    res_block(64),

    conv_layer(64, 128, stride=2),

    res_block(128),

    res_block(128),

    conv_layer(128, 256, stride=2),

    Flatten(),

    nn.Dropout(p=0.2),

    nn.Linear(256, N_LABELS),

)



learn_2 = Learner(data_2, model_2, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn_2.fit_one_cycle(N_EPOCHS, max_lr=LEARNING_RATE)
show_results(learn_2)
val_df_2 = get_val_predictions(learn_2, data_df_2)

show_accuracy(val_df_2)
tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.05, max_lighting=0.2, max_warp=None)
def show_transform_example(tfms, rows, cols, width, height, image_path, **kwargs):

    img = open_image(image_path)

    [img.apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
open_image(f'{path}/train/dog/7270_dog.png').show()
show_transform_example(tfms, 4, 4, 10, 10, f'{path}/train/dog/7270_dog.png', size=224)
data_df_3 = get_data_df(all_data_df,

                        bird=2000, deer=2000, frog=2000, horse=2000, cat=2000, dog=2000,

                        ship=2000, truck=2000, airplane=2000, automobile=2000)

display(data_df_3)



data_3 = (ImageList.from_df(data_df_3, path=path, convert_mode='RGB')

          .split_from_df(col='is_valid')

          .label_from_df(cols='label')

          .transform(tfms)  # ここで、augmentationを反映する。

          .databunch(bs=BATCH_SIZE)

          .normalize())

data_3.show_batch(figsize=SHOW_BATCH_FIGSIZE, rows=3, hide_axis=False)
model_3 = nn.Sequential(

    conv_layer(3, 16, stride=2),

    res_block(16),

    res_block(16),

    conv_layer(16, 32, stride=2),

    res_block(32),

    res_block(32),

    conv_layer(32, 64, stride=2),

    res_block(64),

    res_block(64),

    conv_layer(64, 128, stride=2),

    res_block(128),

    res_block(128),

    conv_layer(128, 256, stride=2),

    Flatten(),

    nn.Dropout(p=0.2),

    nn.Linear(256, N_LABELS),

)



learn_3 = Learner(data_3, model_3, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn_3.fit_one_cycle(N_EPOCHS, max_lr=LEARNING_RATE)
show_results(learn_3)
val_df_3 = get_val_predictions(learn_3, data_df_3)

show_accuracy(val_df_3)