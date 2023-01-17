import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# ---------------------------------------------------------------------
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score,classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from keras.preprocessing import image as ik
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import keras.backend as K
from keras.layers import AveragePooling2D, MaxPooling2D, Input
from keras.models import Model
# ---------------------------------------------------------------------

from scipy.spatial import distance as dist

import cv2
import math
import shutil
import os
import shutil
from tqdm import tqdm



from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score

train = pd.read_csv("../input/melanoma-zefir/train2.csv")
test = pd.read_csv("../input/melanoma-zefir/test2.csv")
train[locing_end].head()
patient_id = LabelEncoder()
sex = LabelEncoder()
anatom_site_general_challenge = LabelEncoder()
patient_id.fit(train["patient_id"].unique())
sex.fit(train["sex"].unique())
anatom_site_general_challenge.fit(train["anatom_site_general_challenge"].unique())
def label_e(dataframe):
    dataframe["patient_id"] = patient_id.transform(dataframe["patient_id"])
    dataframe["sex"] = sex.transform(dataframe["sex"])
    dataframe["anatom_site_general_challenge"] = anatom_site_general_challenge.transform(dataframe["anatom_site_general_challenge"])
patient_id2 = LabelEncoder()
patient_id2.fit(test["patient_id"].unique())
def label_e2(dataframe):
    dataframe["patient_id"] = patient_id2.transform(dataframe["patient_id"])
    dataframe["sex"] = sex.transform(dataframe["sex"])
#     dataframe["diagnosis"] = diagnosis.transform(dataframe["diagnosis"])
    dataframe["anatom_site_general_challenge"] = anatom_site_general_challenge.transform(dataframe["anatom_site_general_challenge"])
train_c = train.copy()
label_e(train_c)
test_c = test.copy()
label_e2(test_c)
SHAPE=256
SHAPE_RESIZE=190
NUM_LENGTH=10
NUM_SEGMENTATION = 9
COUNT_COLOR = (SHAPE_RESIZE//NUM_SEGMENTATION)
locing = ["l"+str(i) for i in range(NUM_LENGTH)]
colors_table = ["Color"+str(canal)+str(znach) for znach in range(COUNT_COLOR*COUNT_COLOR) for canal in range(3)]
matric_color = np.array([[i+j for i in range(COUNT_COLOR)] for j in range(0,COUNT_COLOR*COUNT_COLOR-1,COUNT_COLOR)])
num_segmen = math.ceil(COUNT_COLOR/2)
rows = (COUNT_COLOR//2)
def get_number_color(start,end):
    for j in [start,end]:
        end = np.append(end,matric_color[j])
        end = np.append(end,matric_color[:,j])
    return np.unique(end)
def delete_elemen(massiv,arr):
    mas = massiv
    for i in arr:
         mas = np.setdiff1d(mas,i)
    
    return mas
mass = [get_number_color(i,j) for i,j in zip(range(num_segmen),reversed(range(rows,COUNT_COLOR)))]
mass = [delete_elemen(mass[i],mass[:i]) for i in reversed(range(0,len(mass)))] 
new_features2 = np.array(["diff_0_1_segment","diff_0_2_segment","diff_0_3_segment","diff_0_4_segment","diff_0_5_segment",
                         "diff_0_6_segment","diff_0_7_segment","diff_0_8_segment","diff_0_9_segment","diff_0_10_segment",
                         "diff_1_2_segment","diff_1_3_segment","diff_1_4_segment","diff_1_5_segment","diff_1_6_segment",
                         "diff_1_7_segment","diff_1_8_segment","diff_1_9_segment","diff_1_10_segment",
                         "diff_2_3_segment","diff_2_4_segment",
                         "diff_2_5_segment","diff_2_6_segment","diff_2_7_segment","diff_2_8_segment","diff_2_9_segment",
                         "diff_2_10_segment",
                         "diff_3_4_segment","diff_3_5_segment","diff_3_6_segment","diff_3_7_segment",
                         "diff_3_8_segment","diff_3_9_segment","diff_3_10_segment","diff_4_5_segment","diff_4_6_segment",
                         "diff_4_7_segment","diff_4_8_segment","diff_4_9_segment","diff_4_10_segment",
                         "diff_5_6_segment","diff_5_7_segment","diff_5_8_segment","diff_5_9_segment","diff_5_10_segment",
                         "diff_6_7_segment","diff_6_8_segment","diff_6_9_segment","diff_6_10_segment",
                         "diff_7_8_segment","diff_7_9_segment","diff_7_10_segment",
                         "diff_8_9_segment","diff_8_10_segment","diff_9_10_segment"
                         ]).tolist()
features_blue_segment = ["diff_blue_segment_"+str(i) for i in range(num_segmen)]
features_red_segment = ["diff_red_segment_"+str(i) for i in range(num_segmen)]
features_green_segment = ["diff_green_segment_"+str(i) for i in range(num_segmen)]
new_features_locing = np.array(["mean_locing","std_locing"]) 
blue_in_segment = ["mean_blue_seegment", "std_blue_seegment"]
mean_in_color = ["mean_in_color","std_color","mean_in_color_red",
                 "mean_in_color_green","mean_in_color_blue",
                "std_in_color_red",
                 "std_in_color_green","std_in_color_blue",
                ]
red_in_segment = ["mean_red_seegment", "std_red_seegment"]
new_features = np.array([])
new_features_std = np.array([])
canal_name = ["red","green","blue"]
for i, mas in enumerate(mass):
    for canal in range(3):
        name = "Color_segment_"+str(i)+"_mean_canal_"+canal_name[canal]
        name2 = "Color_segment_"+str(i)+"_std_canal_"+canal_name[canal]
        new_features = np.append(new_features,name)
        new_features_std = np.append(new_features_std,name2)
locing_end = np.hstack((locing,["width","height","split","globuli","sizeveil","sizeglobuli","anatom_site_general_challenge","age_approx","patient_id","sex"]))
locing_end = np.hstack((locing_end,new_features))
locing_end = np.hstack((locing_end,new_features2))
locing_end = np.hstack((locing_end,new_features_locing))
locing_end = np.hstack((locing_end,features_red_segment))
locing_end = np.hstack((locing_end,red_in_segment))
locing_end = np.hstack((locing_end,blue_in_segment))
locing_end = np.hstack((locing_end,features_green_segment))
locing_end = np.hstack((locing_end,features_blue_segment))
locing_end = np.hstack((locing_end,mean_in_color))
locing_end = np.hstack((locing_end,new_features_std[3:]))
locing_end
train_split = 0
train_val_split = 0

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)
for train_index, test_index in split.split(train_c,train_c["target"]):
    train_split = train_c.loc[train_index].copy()
    train_val_split = train_c.loc[test_index].copy()
train_x = train_split[locing_end]
train_y = train_split["target"]
val_x = train_val_split[locing_end]
val_y = train_val_split["target"]
std_scaller = StandardScaler()
std_scaller.fit(train_x)
train_x_std_scalled = std_scaller.transform(train_x)
val_x_std_scalled = std_scaller.transform(val_x)


test_x = test_c[locing_end]
test_x_std_scalled = std_scaller.transform(test_x)
CW = class_weight.compute_class_weight('balanced',np.unique(train_y),train_y)
clases = [0,1]
class_weights = dict(zip(clases,CW))
class_weights

select = SelectFromModel(RandomForestClassifier(n_estimators=59, max_depth=44, min_samples_split=9,  min_samples_leaf=19, class_weight=class_weights))
# select = SelectFromModel(RandomForestClassifier(n_estimators=300, max_depth=44, min_samples_split=9,  min_samples_leaf=12, class_weight=class_weights))

select.fit(train_x_std_scalled,train_y)
SELECT_X = select.transform(train_x_std_scalled)
VAL_X = select.transform(val_x_std_scalled)
m_depth = SELECT_X.shape[1]
m_depth
tree2 = RandomForestClassifier(n_estimators=300, max_depth=150, min_samples_split=9,  min_samples_leaf=18, class_weight=class_weights)
tree2.fit(SELECT_X,train_y)
print("ROC AUC: ",make_scorer(roc_auc_score, needs_proba=True)(tree2, VAL_X, val_y))
test_pred = tree2.predict_proba(TEST_X)
prediction = pd.DataFrame(test_pred,columns=["t","target"])
test["target"] = prediction["target"]
submission = test[["image_name","target"]]
submission.to_csv("submit.csv", index=False, line_terminator="\n")