import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os, gc, random
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm, tqdm_notebook
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121 as ModelPredict
# from tensorflow.keras.applications.densenet import preprocess_input, DenseNet169 as ModelPredict
# from tensorflow.keras.applications.densenet import preprocess_input, DenseNet201 as ModelPredict
# from tensorflow.keras.applications.nasnet import preprocess_input, NASNetLarge as ModelPredict
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2 as ModelPredict
# from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19 as ModelPredict
# from tensorflow.keras.applications.resnet import preprocess_input, ResNet50 as ModelPredict
# Data access
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
# GCS_DS_PATH = './'

img_size = 256
batch_size = 16
SEED = 2020

df_train = pd.read_csv(GCS_DS_PATH + '/train.csv')
df_test = pd.read_csv(GCS_DS_PATH + '/test.csv')

train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)

seed_everything(SEED)
def resize_image(img):
    old_size = img.shape[:2]
    if old_size[1] == img_size:
        return img
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1],new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0,0,0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def load_image(path, img_id):
    path = os.path.join(path,img_id+'.jpg')
    img = cv2.imread(path)
    new_img = resize_image(img)
    new_img = preprocess_input(new_img)
    return new_img

def get_model_feature():
    inp = Input((img_size, img_size, 3))
    backbone = ModelPredict(input_tensor=inp, include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:,:,0])(x)
    return Model(inp, out)

def create_feature(model_feature, df, img_path, out="features.csv"):
    img_ids = df.image_name.values
    n_batches = len(img_ids)//batch_size + 1
    features = {}
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_ids = img_ids[start:end]
        batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
        for i, img_id in enumerate(batch_ids):
            batch_images[i] = load_image(img_path, img_id)
        batch_preds = model_feature.predict(batch_images)
        for i, img_id in enumerate(batch_ids):
            features[img_id] = batch_preds[i]

    feats = pd.DataFrame.from_dict(features, orient='index')
    feats.to_csv(out)
    return feats
# train_feats = pd.read_csv('/kaggle/input/siimisic-256x256/train_img_features_dense121.csv')
# test_feats = pd.read_csv('/kaggle/input/siimisic-256x256/test_img_features_dense121.csv')

mf = get_model_feature()
create_feature(mf, df_train, train_img_path, out='train_img_features.csv')
create_feature(mf, df_test, test_img_path, out='test_img_features.csv')
train_feats = pd.read_csv('train_img_features.csv')
test_feats = pd.read_csv('test_img_features.csv')
train_feats.set_index(train_feats.columns[0], inplace=True)
test_feats.set_index(test_feats.columns[0], inplace=True)

#Combine the image and tabular data
df_train_full = pd.merge(df_train, train_feats, how='inner', left_on='image_name', right_index=True)
df_test_full = pd.merge(df_test, test_feats, how='inner', left_on='image_name', right_index=True)

#Drop the unwanted columns
train = df_train_full.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis=1)
test = df_test_full.drop(['image_name','patient_id'],axis=1)

#Label Encode categorical features
train.sex.fillna('NaN',inplace=True)
test.sex.fillna('NaN',inplace=True)
train.anatom_site_general_challenge.fillna('NaN',inplace=True)
test.anatom_site_general_challenge.fillna('NaN',inplace=True)
le_sex = LabelEncoder()
le_site = LabelEncoder()
train.sex = le_sex.fit_transform(train.sex)
test.sex = le_sex.transform(test.sex)
train.anatom_site_general_challenge = le_site.fit_transform(train.anatom_site_general_challenge)
test.anatom_site_general_challenge = le_site.transform(test.anatom_site_general_challenge)
train['age_approx'] = train['age_approx'].fillna(0)
test['age_approx'] = test['age_approx'].fillna(0)
folds = StratifiedKFold(n_splits=5, shuffle=True)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
features = [f for f in train.columns if f != 'target']
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['target'])):
    train_X, train_y = train[features].iloc[train_idx], train['target'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['target'].iloc[valid_idx]
    clf = LGBMClassifier(
        #device='gpu',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        colsample_bytree=0.9,
        num_leaves=50
    )
    print('*****Fold: {}*****'.format(n_fold))
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)],
            eval_metric= 'auc', verbose= 20, early_stopping_rounds= 20)

    oof_preds[valid_idx] = clf.predict_proba(valid_X, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    
display_importances(feature_importance_df)
submission = pd.DataFrame({
    "image_name": df_test.image_name, 
    "target": sub_preds
})
submission.to_csv('submission.csv', index=False)