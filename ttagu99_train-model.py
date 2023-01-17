def apk(actual, predicted, k):
    if actual not in predicted:  # 정답이 제출한 값에 없으면 score는 0
        return 0.0
    for i in range(k):
        if actual in  predicted[:i+1]:  # 제출한 값이 K번째 안에 정답
            return 1.0 / len(predicted[:i+1])  # score
actual ='A'
predicted = ['A','B','C']
apk(actual,predicted, 3)
predicted = ['B','A','C']
apk(actual,predicted, 3)
predicted = ['B','C','A']
apk(actual,predicted, 3)
predicted = ['B','C','D']
apk(actual,predicted, 3)
import os
data_path = "/kaggle/input/quickdraw-doodle-recognition/" # data set을 다운로드한 경로
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
# kaggle hdd 용량이 작음(5GB)으로 local에서 가능
from tqdm import tqdm
divide_shuffles = 100 # data를 분할 할 file 수
raw_shuffle_data_path = data_path + 'shuffle_raw_gzs/'
try:
    os.mkdir(raw_shuffle_data_path)
    for y, csv_file in enumerate(tqdm(train_csvs)):
        df = pd.read_csv(train_file_path+csv_file, usecols= ['drawing','key_id'])
        df['y'] = y # label
        df['cv'] = (df.key_id//10000) % divide_shuffles # keyid로 data 나누기.
        for k in range(divide_shuffles):
            filename = raw_shuffle_data_path +f'train_k{k}.csv'
            chunk = df[df.cv == k] # 0~99 까지 cv에 번호로 select
            chunk = chunk.drop(['key_id','cv'], axis=1)
            if y == 0:
                chunk.to_csv(filename, index=False) # 처음이면 파일을 만들고
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False) # add   
    for k in tqdm(range(divide_shuffles)):
        filename = raw_shuffle_data_path +f'train_k{k}.csv'
        df = pd.read_csv(filename) # 아직 까지는 label 순서대로 data가 만들어 져 있음.
        # label 별 파일의 내용을 shuffle 하기 위해 랜덤 값 추가
        df['rnd'] = np.random.rand(len(df)) 
        # 추가된 랜덤값으로 정렬하여 순서를 shuffle함
        df = df.sort_values(by='rnd').drop('rnd', axis=1) 
        # ssd 용량이 부족하지 않다면 compression은 빼는게 빠름
        df.to_csv(filename.replace('.csv','.gz'),compression='gzip', index=False) 
        os.remove(filename)
except:
    print("shuffled train data 준비는 한번만 실행합니다.")
    pass

test_raw_df = pd.read_csv(data_path+"test_raw.csv")
test_raw_df.head()
print(test_raw_df.shape)
test_raw_df.shape[0]%len(label_names) 
import cv2
import numpy as np
def draw_raw_cv2(raw_strokes,size=128,lw=6,last_drop_r=0.0,second_strokes = None):  
    ofs = lw*2 # 완성된 이미지에 테두리 공백 
    limit_ett = 20*1000 # 최대 시간 20초
    npstrokes = [] 
    mminfo={"xmin":float('inf'),"ymin":float('inf')
        , "xmax":float('-inf'),"ymax":float('-inf')} 
    
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
            ## like 속력, 비 정상 data 가 있음. et-st 가 음수인경우 et-st가 0인 경우
            color_v = min((((sx-ex)**2+(sy-ey)**2)**0.5 / (abs(et-st)+1) *5), 1.0) 
            if i==0:
                color_a = 0
            else: ## 획의 변화량, like 가속력, 첫점은 255   
                color_a = min((((inertia_x-ex)**2 +
                 (inertia_y-ey)**2)**0.5 / (abs(et-st)+1) *5), 1.0)          
            nimg = cv2.line(nimg, (sx, sy), (ex, ey), (color,color_v,color_a), lw)
            # print(color_v,color_a)
            if i==0:
                color_inter = min((float(st-pre_st_t)*10/limit_ett),1.0)
                if t == 0 or (second_strokes is not None 
                              and t == end_fist_st_len):
                    color_inter = 1.0 # 첫 stroke의 첫번째 점 표시
                # interval time
                nimg = cv2.circle(nimg, (sx, sy), lw
                                  , (0.0,0.0,color_inter), -1)
            # 마지막 획에 마지막 점    
            if i==len(stroke[0])-2 and t == len(raw_strokes) -1: 
                color_end = (float(ett)/(limit_ett)) # end time
                nimg = cv2.circle(nimg, (ex, ey), lw
                                  , (0.0,color_end,0.0), -1) 
            # mix up augmentation 마지막 획에 마지막 점
            if second_strokes is not None \
            and i==len(stroke[0])-2 and t == end_fist_st_len -1: 
                color_end = (float(first_ett)/(limit_ett)) # end time
                nimg = cv2.circle(nimg, (ex, ey), lw, (0.0,color_end,0.0), -1) 
                
            inertia_x = ex + (ex-sx)
            inertia_y = ey + (ey-sy)
            pre_st_t=et         
    return nimg
train_file0_df = pd.read_csv(raw_shuffle_data_path+'train_k0.gz')
draw_test = draw_raw_cv2(json.loads(train_file0_df.loc[0].drawing),  size=128, lw=2)
fig,ax = plt.subplots(figsize=(5,5))
ax.set_title(label_names[train_file0_df.loc[0].y])
ax.imshow(draw_test)
plt.show()

#  check last drop augmentation
draw_test_drop=draw_raw_cv2(json.loads(train_file0_df.loc[0].drawing),size=128
                              ,lw=2,last_drop_r=0.3)
draw_test_drop_mix=draw_raw_cv2(json.loads(train_file0_df.loc[0].drawing)
                            ,size=128,lw=2,last_drop_r=0.3 
                ,second_strokes=json.loads(train_file0_df.loc[100].drawing))
fig,ax = plt.subplots(ncols=3,figsize=(15,5))
ax[0].set_title(label_names[train_file0_df.loc[0].y] + ' ▶ orginal')
ax[0].imshow(draw_test)
ax[1].set_title(label_names[train_file0_df.loc[0].y] + ' ▶ drop_aug')
ax[1].imshow(draw_test_drop)
ax[2].set_title(label_names[train_file0_df.loc[0].y] + ' ▶ mix ' + 
                label_names[train_file0_df.loc[100].y])
ax[2].imshow(draw_test_drop_mix)
plt.show()
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.utils import to_categorical      
class DoodelGenerator(tf.keras.utils.Sequence):
    def __init__(self,df_files,input_shape,batchsize,label_num=340
                 ,lw=3,state='Train',last_drop_r=0.0,mixup_r = 0.0):
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
        # batch size 만큼 index를 뽑음
        batch_idx = self.idx[index*self.batchsize:(index+1)*self.batchsize] 
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
                ,last_drop_r = self.last_drop_r,second_strokes=mixup_strokes[i])
                if self.state != 'Test':
                    ysm_mix = self.smooth_labels(to_categorical(mixup_labels[i]
                                                , num_classes=self.label_num))
                    ysm_org = self.smooth_labels(to_categorical(df.y.values[i]
                                                , num_classes=self.label_num))
                    y[i, :] = (ysm_mix*0.5) + (ysm_org*0.5)
            else:
                X[i, :, :, ] = draw_raw_cv2(raw_strokes, size=h, lw=self.lw
                                        , last_drop_r = self.last_drop_r)
            
                if self.state != 'Test':
                    y[i, :] = to_categorical(df.y.values[i]
                                             ,num_classes=self.label_num)
            
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
        print('current step file : ', self.df_files[self.file_sel]
              ,'state:', self.state, 'df_len:', self.df.shape[0])
        self.idx = np.tile(np.arange(len(self.df)),2) # train file size가 flexible함으로 idx 배열을 연장
        if self.state == 'Train':
            np.random.shuffle(self.idx)        
        self.file_sel = (self.file_sel+1)%len(self.df_files) # next csv file roll
    
# tensorflow ver 2.1 의 bug, model.fit 에서 tf.keras.utils.Sequnce의 on_epoch_end 를 호출하지 않음
# 별도의 callback에서 호출하도록 추가, 추후 버그 수정버전으로 release 되면 삭제해도 되는 부분
class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, callback): # train generator의 on_epoch_end 콜백 인자로 받음
        self.callback = callback
        
    def on_epoch_end(self, epoch, logs=None): # 단순 callback을 호출 해줌
        self.callback()

df_files = [raw_shuffle_data_path +f'train_k{k}.gz' for k in range(divide_shuffles)]
print("학습을 위해 준비된 파일 수 : ", len(df_files))
input_shape = (128,128,3)
gen_data_check = DoodelGenerator(df_files, input_shape=input_shape
            ,batchsize=25,state='DataCheck',lw=2,last_drop_r=0.2,mixup_r= 0.1)
import matplotlib.pyplot as plt
xx, y = gen_data_check.__getitem__(0)
fig, axs = plt.subplots(5, 5, figsize=(10,10))
labels = np.argmax(y,axis=1) # generator에서 출력한 labels
for i in range(25):
    axs[i//5][i%5].imshow(xx[i]) 
    axs[i//5][i%5].axis('off')
    axs[i//5][i%5].set_title(label_names[labels[i]])
plt.show()
train_vals = df_files[:-1]
hold_out_set= df_files[-1:]
print('hold_out_set:',len(hold_out_set), 'train_val_set:',len(train_vals))

R_EPOCHS = 1 # 2 실제 Epoch, 1Epoch 만 학습함
EPOCHS = R_EPOCHS * (len(train_vals)-1)
print('Real Epochs:',R_EPOCHS, 'Divide Virtual Epoch:', EPOCHS)
!pip install efficientnet
import efficientnet.tfkeras as efn
def build_model(backbone= efn.EfficientNetB0
                ,input_shape = (128,128,3),use_imagenet = 'imagenet'):
    base_model = backbone(input_shape=input_shape
                          ,weights=use_imagenet,include_top= False)
    x = base_model.output
    x = tf.keras.layers.GlobalAvgPool2D(name='gap')(x)
    predictions = tf.keras.layers.Dense(len(label_names),activation='softmax'
                    ,name='prediction')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model
def mapk(actual, predicted, k=3): # 학습 후 hold out set 을 평가 하는데 사용할 함수
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions): # submission을 위해 top3 category로 변환할 함수
    return pd.DataFrame(np.argsort(-predictions,axis=1)[:,:3]
                        ,columns=['a','b','c'])
    
def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    
def map_at3(y_true, y_pred): # train 과정 중에 평가를 위한 함수
    map3 = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)*0.5
    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)*0.17
    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)*0.33    
    return map3

recipes = []
recipes.append({'backbone':efn.EfficientNetB0,"batch_size":100
        ,'name':'Efb0','val_sel':0,'input_shape':(128,128,3),'lw':2})
recipes.append({'backbone':efn.EfficientNetB4, "batch_size":40
        ,'name':'Efb1','val_sel':4,'input_shape':(128,128,3),'lw':2})
recipes.append({'backbone':efn.EfficientNetB5, "batch_size":40
        ,'name':'Efb2','val_sel':5,'input_shape':(128,128,3),'lw':2})
recipes.append({'backbone':efn.EfficientNetB7, "batch_size":28
        ,'name':'Efb7','val_sel':7,'input_shape':(128,128,3),'lw':2})
def make_sub(model, test_datagen, holdout_datagen, model_name):
   # ho_prob save
    ho  = model.predict(holdout_datagen,verbose=1) 
    ho_df = holdout_datagen.get_cur_df()
    ho = ho[:len(ho_df)]
    top3 = preds2catids(ho)
    ho_map3 = mapk(ho_df.y, np.array(top3)) # Hold Out 점수 계산
    np.save(data_path+'output/ho_prob_'+model_name+str(ho_map3)+'.npy',ho)

    # test_prob save
    test_predictions = model.predict(test_datagen, verbose=1)
    test_predictions = test_predictions[:len(test_raw_df)]
    top3 = preds2catids(test_predictions)
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(label_names)}
    top3cats = top3.replace(id2cat)
    np.save(data_path+'output/test_prob_' 
            +model_name+str(ho_map3)+'.npy',test_predictions)    

    # single model test submission    
    sub_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = sub_df[['key_id', 'word']]
    submission.to_csv(data_path+'output/submission_'  
                      +model_name+str(ho_map3)+'.csv', index=False)
import random

strategy = tf.distribute.MirroredStrategy(  
                tf.config.experimental.list_logical_devices('GPU')) 
gpus= strategy.num_replicas_in_sync  # 가용 gpu 수 
print('use gpus:',gpus)

# val map_at3 score 가 virtual epoch 10회 연속 증가 하지 않으면, LR 1/2 감소
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_map_at3', factor=0.5
            ,verbose=1, patience=10,cooldown=5, min_lr=0.00001,min_delta=0.00001) 

for i, recipe in enumerate(recipes):
    model_name = recipe["name"] + '_val_' + str(recipe["val_sel"]) 
    best_save_model_file = model_name + '.h5' 
    print('best_save_model_file path : ',best_save_model_file)
    check_point=tf.keras.callbacks.ModelCheckpoint(monitor='val_map_at3',verbose=1
        ,filepath=data_path + best_save_model_file,save_best_only=True,mode='max') 

    valid_file = train_vals[recipe["val_sel"]]
    train_files = list(np.delete(train_vals, recipe["val_sel"]))    
    # list 를 Shuffle, 중단후  동일 파일 재학습 하여 Overfitting 되는 문제 완화
    random.shuffle(train_files) 
    # Train Set만 augmentation 적용
    train_datagen = DoodelGenerator(train_files,input_shape=recipe['input_shape']
                ,lw=recipe['lw'], last_drop_r=0.2 , mixup_r=0.1
                , batchsize=recipe['batch_size']*gpus,state='Train') 
    valid_datagen = DoodelGenerator([valid_file],input_shape=recipe['input_shape']
                ,lw=recipe['lw'],batchsize=recipe['batch_size']*gpus,state='Valid')
    

    with strategy.scope():    # 가용 gpu 수 모두 사용
        model = build_model(backbone= recipe['backbone']
        ,input_shape=recipe['input_shape'], use_imagenet = 'imagenet')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001*gpus)
        ,loss='categorical_crossentropy'
        ,metrics=[tf.keras.metrics.categorical_accuracy, top_3_accuracy, map_at3]) 
    # 동일 모델이름의 사전 학습된 weights가 있다면 로딩 후 재시작   
    if os.path.exists(data_path + best_save_model_file): 
        print('restart train : ', data_path + best_save_model_file)
        model.load_weights(data_path + best_save_model_file)
    hist = model.fit(train_datagen
        ,validation_data=valid_datagen,epochs= EPOCHS,verbose=1
        ,callbacks =[reduce_lr, check_point, OnEpochEnd(train_datagen.on_epoch_end)])
    # 학습 완료 후 Best Score Model 로딩
    model.load_weights(data_path + best_save_model_file) 
    test_datagen = DoodelGenerator([data_path+"test_raw.csv"]
                            ,input_shape=recipe['input_shape'],lw=recipe['lw']
                            , batchsize=recipe['batch_size']*gpus,state='Test')
    holdout_datagen = DoodelGenerator(hold_out_set,input_shape=recipe['input_shape']
                            ,lw=recipe['lw']
                            , batchsize=recipe['batch_size']*gpus,state='Holdout')
    # hold out & test set, probability 파일 저장
    make_sub(model, test_datagen, holdout_datagen, model_name) 
import pandas as pd
import numpy as np
output_path = data_path+'output/'
outputs = os.listdir(output_path)
 # hold out probability
hold_out_probs = [output_path+f for f in outputs if f.find('ho_prob') >= 0 ]
# test probability
test_out_probs = [output_path+f for f in outputs if f.find('test_prob') >= 0 ] 

ho_df = pd.read_csv(hold_out_set[0])
ho_s = []
for prob_path in hold_out_probs:
    ho = np.load(prob_path)
    ho = ho[:len(ho_df)]
    ho_s.append(ho)
targets = ho_df.y.to_numpy() # hold out target
from scipy.optimize import OptimizeResult
def map3_loss(weights, predictions, targets):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    top3 = np.argsort(-final_prediction, axis=1)[:, :3]
    ho_map3 = mapk(targets, np.array(top3))    
    print(weights, ho_map3)
    return 1 - ho_map3

def custom_minimizer(fun,x0,args=(),stepsize=0.1,maxiter=100,callback=None):
    bestx = x0
    besty = fun(x0,*args) # loss func
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter: 
        improved = False
        niter += 1
        for dim in range(np.size(x0)):
            # 각 prob에 해당하는 weight를 step size만큼 +,- 방향 이동한 곳의 
            # loss 계산하고, 가장 줄어드는 방향으로 업데이트
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]: 
                testx = np.copy(bestx)
                testx[dim] = s
                minx = min(testx)
                if minx<0: 
                    testx-=minx  
                testx/=sum(testx)
                testy = fun(testx, *args) 
                funcalls += 1
                if testy < besty: 
                    besty = testy
                    bestx = testx
                    improved = True
            if callback is not None:
                callback(bestx)
    return OptimizeResult(fun=besty, x=bestx, nit=niter, success=(niter > 1)) 

starting_values = [0.25]*len(ho_s) # 균등 분할 부터 Start
res = custom_minimizer(map3_loss,starting_values, args=(ho_s, targets))
print('Best Ensamble Score: ', 1-res['fun'])
print('Best Weights:', res['x'])

def ens_sub(ens_prob, file_name='ens_sub.csv'):
    top3 = preds2catids(ens_prob)
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(label_names)}
    top3cats = top3.replace(id2cat) 
    sub_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = sub_df[['key_id', 'word']]
    submission.to_csv(data_path+'output/'+ file_name, index=False)
    
ens_prob = np.zeros((len(sub_df),len(label_names)))
for i, prob_path in enumerate(test_out_probs):
    ens_prob += (np.load(prob_path) * res['x'][i])
    
ens_sub(ens_prob)

from collections import Counter,OrderedDict
from operator import itemgetter
def balancing_predictions(test_prob,factor=0.1,minfactor=0.001,patient=5
                          ,permit_cnt=332,max_search=10000,label_num=340):
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
                print('reduce factor: ',factor,', current max category num: '
                ,ctop1[0][1])

        if ctop1[0][1] <= permit_cnt:
            print('idx: ',ctop1[0][0] ,', num: ', ctop1[0][1]) 
            break
        test_prob[:,ctop1[0][0]] *= (1.0-factor)
    return test_prob

bal_ens_prob = balancing_predictions(ens_prob)
ens_sub(bal_ens_prob,'bal_ens_sub.csv')
