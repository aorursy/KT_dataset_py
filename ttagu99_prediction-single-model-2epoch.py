# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import os
def apk(actual, predicted, k):
    if actual not in predicted:  # 정답이 제출한 값에 없으면 score는 0
        return 0.0
    for i in range(k):
        if actual in  predicted[:i+1]:  # 제출한 값의 범위를 첫번째, 첫번째 ~ 두번째, *** 첫번째 ~ N번째 까지 에 정답이 있다면.
            return 1.0 / len(predicted[:i+1])  # (1.0 / 비교 범위 길이) 가 score임
import os
data_path = "/kaggle/input/quickdraw-doodle-recognition/" 
print(os.listdir(data_path))
import pandas as pd
sub_df = pd.read_csv(data_path+'sample_submission.csv')
print("test data 수:",len(sub_df))
sub_df.head()
train_file_path = data_path + 'train_raw/'
eiffel_df = pd.read_csv(train_file_path + 'The Eiffel Tower.csv')
eiffel_df.head()
train_simple_file_path = data_path + 'train_simplified/'
eiffel_simple_df = pd.read_csv(train_simple_file_path + 'The Eiffel Tower.csv')
eiffel_simple_df.head()
import json
raw_images = [json.loads(draw) for draw in eiffel_df.head()['drawing'].values]
simple_images = [json.loads(draw) for draw in eiffel_simple_df.head()['drawing'].values]
import matplotlib.pyplot as plt
for index in range(3):
    f, (ax1, ax2) = plt.subplots(ncols=2,nrows=1,figsize=(8,4))
    for x,y,t in raw_images[index]:
        ax1.plot(x, y, marker='.')
    for x,y in simple_images[index]:
        ax2.plot(x, y, marker='.')
    ax1.set_title('raw drawing')
    ax2.set_title('simplified drawing')    
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax1.legend(range(len(raw_images[index])))
    ax2.legend(range(len(simple_images[index])))
    plt.show()  
print("======== 첫번째 raw drawing의 첫 획 Data 중 5개 Point 정보 =========")
print("x좌표: ", json.loads(eiffel_df['drawing'][0])[0][0][:5])
print("y좌표: ", json.loads(eiffel_df['drawing'][0])[0][1][:5])
print("msec: ", json.loads(eiffel_df['drawing'][0])[0][2][:5])

print("======== 첫번째 Simplified drawing의 첫 획 Data 중 5개 Point 정보 =========")
print("x좌표: ", json.loads(eiffel_simple_df['drawing'][0])[0][0][:5])
print("y좌표: ", json.loads(eiffel_simple_df['drawing'][0])[0][1][:5])
train_csvs= os.listdir(train_file_path)
print("train_raw 폴더 내 파일 수:", len(train_csvs))
print(train_csvs[:5])

file_size = 0
label_names = []

for csv_file in train_csvs:
    file_size += os.path.getsize(train_file_path + csv_file) # data file들의 용량을 계산
    label_names.append(csv_file.replace('.csv','')) 
print("파일 크기 : ", file_size//(1024*1024*1024) ,"GB")

label_names = sorted(label_names,key=lambda x : str.lower(x+'.csv')) # at kaggle notebook 
test_raw_df = pd.read_csv(data_path+"test_raw.csv")
test_raw_df.head()
print(test_raw_df.shape)
test_raw_df.shape[0]%len(label_names) 
import cv2
import numpy as np
def draw_raw_cv2(raw_strokes, size=128, lw=6, last_drop_r=0.0, second_strokes = None):  
    ofs = lw*2 # 완성된 이미지에 테두리 공백 
    limit_ett = 20*1000 # 최대 시간 20초
    npstrokes = [] 
    mminfo={"xmin":float('inf'),"ymin":float('inf'), "xmax":float('-inf'),"ymax":float('-inf')} 
    
    # strokes drop augmentation
    drop_num = int(np.random.random()*last_drop_r *len(raw_strokes))
    if drop_num>0:
        raw_strokes = raw_strokes[:-drop_num]
    
    # mixup augmentation
    if second_strokes is not None:
        first_ett = raw_strokes[-1][-1][-1]
        end_fist_st_len = len(raw_strokes)
        raw_strokes.extend(second_strokes)
        
    for t, stroke in enumerate(raw_strokes):
        npstroke = np.array(stroke)
        #print(npstroke.shape)
        npstrokes.append(npstroke)        
        mminfo["xmin"] = min(mminfo["xmin"], min(npstroke[0]))
        mminfo["xmax"] = max(mminfo["xmax"], max(npstroke[0]))
        mminfo["ymin"] = min(mminfo["ymin"], min(npstroke[1]))
        mminfo["ymax"] = max(mminfo["ymax"], max(npstroke[1]))
        
    ett=npstrokes[-1][-1][-1] # 얼마나 빨리 완료하는가 20초 이하  
    
    nimg = np.zeros((size,size,3),dtype=float)
    # print(mminfo) # min 좌표에 음수가 있는 경우도 있음.   
    org_width = mminfo["xmax"] - mminfo["xmin"] 
    org_height = mminfo["ymax"] - mminfo["ymin"]
    ratio = max(org_width,org_height) / (size-ofs*2)
    if ratio == 0 :
        print('ratio 0 case ? null data ? log for debugging',mminfo)
        return nimg
    pre_st_t = 0 
    for t, stroke in enumerate(npstrokes):
        stroke[0] = (stroke[0] - mminfo["xmin"])/ratio + ofs
        stroke[1] = (stroke[1] - mminfo["ymin"])/ratio + ofs
        inertia_x = 0
        inertia_y = 0
        if second_strokes is not None and t == end_fist_st_len:
            pre_st_t = 0
        for i in range(len(stroke[0]) - 1): # 각 stroke의 Point loop, 마지막 좌표 전까지
            color = min((1.0 - 0.95*float(t)/len(npstrokes)),1.0) # 획 순에 대한 color
            sx = int(stroke[0][i])
            sy = int(stroke[1][i])
            st = stroke[2][i]
            ex = int(stroke[0][i + 1])
            ey = int(stroke[1][i + 1])
            et = stroke[2][i+1]
            
            color_v = min((((sx-ex)**2+(sy-ey)**2)**0.5 / (abs(et-st)+1) *5), 1.0) ## like 속력, 비 정상 data 가 있음. et-st 가 음수인경우 et-st가 0인 경우
            if i==0:
                color_a = 0
            else:
                color_a = min((((inertia_x-ex)**2+(inertia_y-ey)**2)**0.5 / (abs(et-st)+1) *5), 1.0) ## 획의 변화량, like 가속력, 첫점은 255            
            nimg = cv2.line(nimg, (sx, sy), (ex, ey), (color,color_v,color_a), lw)
            # print(color_v,color_a)
            if i==0:
                color_inter = min((float(st-pre_st_t)*10/limit_ett),1.0)
                if t == 0 or (second_strokes is not None and t == end_fist_st_len):
                    color_inter = 1.0 # 첫 stroke의 첫번째 점 표시
                nimg = cv2.circle(nimg, (sx, sy), lw, (0.0,0.0,color_inter), -1) ##interval time
                
            if i==len(stroke[0])-2 and t == len(raw_strokes) -1: #마지막 획에 마지막 점
                color_end = (float(ett)/(limit_ett))
                nimg = cv2.circle(nimg, (ex, ey), lw, (0.0,color_end,0.0), -1) ##end time

            if second_strokes is not None and i==len(stroke[0])-2 and t == end_fist_st_len -1: #마지막 획에 마지막 점
                color_end = (float(first_ett)/(limit_ett))
                nimg = cv2.circle(nimg, (ex, ey), lw, (0.0,color_end,0.0), -1) ##end time
                
            inertia_x = ex + (ex-sx)
            inertia_y = ey + (ey-sy)
            pre_st_t=et     
            
    return nimg

import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.utils import to_categorical      
class DoodelGenerator(tf.keras.utils.Sequence):
    def __init__(self, df_files, input_shape, batchsize,label_num=340, lw=3, state='Train', last_drop_r=0.0, mixup_r = 0.0):
        self.df_files = df_files
        self.file_sel = 0 # 파일 list 중 현재 fit하는데 사용할 파일 index
        self.batchsize = batchsize
        self.input_shape = input_shape
        self.label_num = label_num
        self.lw = lw
        self.state = state
        self.last_drop_r = last_drop_r
        self.mixup_r = mixup_r
        self.on_epoch_end()
        self.len = -(-len(self.df)//self.batchsize) 

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        batch_idx = self.idx[index*self.batchsize:(index+1)*self.batchsize] # batch size 만큼 index를 뽑음
        h,w,ch = self.input_shape
        X = np.zeros((len(batch_idx), h,w,ch)) #batch
        y = np.zeros((len(batch_idx), self.label_num))
        df = self.df.loc[batch_idx]
        mixup_num = int(self.batchsize*self.mixup_r)
        mixup_df = self.df.loc[np.random.randint(0,len(self.df),size=mixup_num)]
        mixup_strokes=[]
        mixup_labels=[]
        for raw_strokes, label in mixup_df.values:
            mixup_strokes.append(json.loads(raw_strokes))
            mixup_labels.append(label)
            
        for i in range(self.batchsize):
            raw_strokes = json.loads(df.drawing.values[i])
            if i < len(mixup_strokes):
                X[i, :, :, ] = draw_raw_cv2(raw_strokes, size=h, lw=self.lw
                                        , last_drop_r = self.last_drop_r,second_strokes=mixup_strokes[i])
                if self.state != 'Test':
                    ysm_mix = self.smooth_labels(to_categorical(mixup_labels[i], num_classes=self.label_num))
                    ysm_org = self.smooth_labels(to_categorical(df.y.values[i], num_classes=self.label_num))
                    y[i, :] = (ysm_mix*0.5) + (ysm_org*0.5)
            else:
                X[i, :, :, ] = draw_raw_cv2(raw_strokes, size=h, lw=self.lw
                                            , last_drop_r = self.last_drop_r)
            
                if self.state != 'Test':
                    y[i, :] = to_categorical(df.y.values[i], num_classes=self.label_num)
            
        if self.state != 'Test':
            return X,y
        else:
            return X
    
    def get_cur_df(self): # 현재 로딩되어 있는 파일을 반환하는 함수, holdout set 평가시 사용
        return self.df 
    
    def smooth_labels(self,labels, factor=0.1): # mix up augmentation 사용시 사용
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels
    
    def on_epoch_end(self):
        self.df = pd.read_csv(self.df_files[self.file_sel])
        print('current step file : ', self.df_files[self.file_sel], 'state:', self.state, 'df_len:', self.df.shape[0])
        self.idx = np.tile(np.arange(len(self.df)),2) # train file size가 flexible함으로 idx 배열을 연장
        if self.state == 'Train':
            np.random.shuffle(self.idx)        
        self.file_sel = (self.file_sel+1)%len(self.df_files) # next csv file roll
    
hold_out_set= 'train_k99'
!pip install efficientnet
import efficientnet.tfkeras as efn
def build_model(backbone= efn.EfficientNetB0, input_shape = (128,128,3), use_imagenet = 'imagenet'):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet,include_top= False)
    x = base_model.output
    x = tf.keras.layers.GlobalAvgPool2D(name='gap')(x)
    predictions = tf.keras.layers.Dense(len(label_names), activation='softmax', name='prediction')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model
def mapk(actual, predicted, k=3): # 학습 후 hold out set 을 평가 하는데 사용할 함수
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions): # submission을 위해 top3 category로 변환할 함수
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def map_at3(y_true, y_pred): # train 과정 중에 평가를 위한 함수
    map3 = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)*0.5
    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)*0.17
    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)*0.33    
    return map3
recipes = []
recipes.append({'backbone':efn.EfficientNetB7, "batch_size":30, 'name':'Efb7','val_sel':7,'input_shape':(128,128,3),'lw':2})
ext_data_path = '/kaggle/input/doodle-model/'
import random
for i, recipe in enumerate(recipes):
    model_name = recipe["name"] + '_val_' + str(recipe["val_sel"])  + '_ep2'
    best_save_model_file = model_name + '.h5' 
    print('best_save_model_file path : ',best_save_model_file)

    model = build_model(backbone= recipe['backbone'], input_shape = recipe['input_shape'], use_imagenet = None)
    model.load_weights(ext_data_path + best_save_model_file)
    test_datagen = DoodelGenerator([data_path+"test_raw.csv"], input_shape=recipe['input_shape'], lw=recipe['lw']
                                   , batchsize=recipe['batch_size'],state='Test')
    test_predictions = model.predict(test_datagen, verbose=1)
    top3 = preds2catids(test_predictions)
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(label_names)}
    top3cats = top3.replace(id2cat) 
    sub_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = sub_df[['key_id', 'word']]
    submission.to_csv('submission_'+model_name+'.csv', index=False)
from collections import Counter,OrderedDict
from operator import itemgetter

def balancing_predictions(test_prob, factor = 0.1, minfactor = 0.001, patient = 5, permit_cnt=332, max_search=10000, label_num=340):
    maxk = float('inf')
    s_cnt = np.zeros(label_num)
    for i in range(max_search):
        ctop1 = Counter(np.argmax(test_prob,axis=1))
        ctop1 = sorted(ctop1.items(), key=itemgetter(1), reverse=True)
        if maxk > ctop1[0][1]:
            maxk = ctop1[0][1]
        else:
            s_cnt[ctop1[0][0]]+=1
            if np.max(s_cnt)>patient:
                if factor< minfactor:
                    print('stop min factor')
                    break
                s_cnt=np.zeros(label_num)
                factor*=0.99
                print('reduce factor: ', factor, ', current max category num: ', ctop1[0][1])

        if ctop1[0][1] <= permit_cnt:
            print('idx: ',ctop1[0][0] ,', num: ', ctop1[0][1]) 
            break
        test_prob[:,ctop1[0][0]] *= (1.0-factor)
        
    return test_prob
bal_test_prob = balancing_predictions(test_predictions)
bal_top3 = preds2catids(bal_test_prob)
bal_top3cats = bal_top3.replace(id2cat) 
sub_df['word'] = bal_top3cats['a'] + ' ' + bal_top3cats['b'] + ' ' + bal_top3cats['c']
bal_submission = sub_df[['key_id', 'word']]
bal_submission.to_csv('submission_bal_'+model_name+'.csv', index=False)
