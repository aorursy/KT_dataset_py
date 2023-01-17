!wget -nc https://github.com/weiyunchen/code1/raw/master/data.xlsx

!pip install plotly_express
### 导入模块

import datetime

from datetime import datetime

from math import sqrt

import numpy as np

import pandas as pd

from numpy import concatenate

import folium

import lightgbm as lgb

from folium.plugins import HeatMap

from keras.models import Sequential

from keras import backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.layers.embeddings import Embedding

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from scipy import stats

from scipy.stats import norm, skew 

import plotly_express as px

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.figure_factory as ff



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

#显示所有列

pd.set_option('display.max_columns', None)

#显示所有行

pd.set_option('display.max_rows', None)

#设置value的显示长度为50

pd.set_option('max_colwidth',50)
# 载入数据

df = pd.read_excel('data.xlsx',encoding='gbk',sheetname=4)

df0 = pd.read_excel('data.xlsx',encoding='gbk',sheetname=0)

df1 = pd.read_excel('data.xlsx',encoding='gbk',sheetname=1)

df2 = pd.read_excel('data.xlsx',encoding='gbk',sheetname=2)

df3 = pd.read_excel('data.xlsx',encoding='gbk',sheetname=3)

# 数据合并

dff = pd.concat([df0,df1,df2,df3])

# 数据预处理

dff["tube_name"] = dff["city"] + dff["tube_name"]

dff.drop(labels=['city'],axis=1,inplace=True)

# 数据合并

data = pd.merge(df,dff,on="tube_name")

                

data.rename(columns={'租金中位数（元/月）':'area_price_level', '居住便利度':'convenience', 'tube_distance（m）':'tube_distance'},inplace = True)
data['city_cn']=data['city']

data.city[data.city=='上海'] = 'shanghai'

data.city[data.city=='广州'] = 'guangzhou'

data.city[data.city=='北京'] = 'beijing'

data.city[data.city=='深圳'] = 'shenzhen'
# 对类别特征编码

lb=LabelEncoder()

data['city_encode'] = lb.fit_transform(data['city'].values)

data['floor_encode'] = lb.fit_transform(data['floor'].values)

data['towards_encode'] = lb.fit_transform(data['towards'].values)
data.head(3)
px.scatter(data,x="price_area",y="rent_room",color='city_cn',size="convenience",size_max=6)
px.scatter(data[(data['city']=='guangzhou')|(data['city']=='shenzhen')],x="price_area",y="rent_room",color='tube_name',size="convenience",size_max=6)
test=data[(data.rent_room<=1000)&(data.price_area>=70)]



drop1=[x for i,x in enumerate(data.index) if (data['rent_room'].iloc[i]<=1000)&(data['price_area'].iloc[i]>=60)]



data=data.drop(drop1,axis=0) 
p = sns.pairplot(pd.DataFrame(list(zip(data['rent_room'], data['city'], data['room_no'], data['tube_distance'], data['convenience'])), 

                        columns=['rent_room','city', 'room_no', 'tube_distance', 'convenience']), hue='city', palette="Set2")
plt.figure(figsize=(6,6))

sns.heatmap(data.corr(),linewidths=0.1,linecolor='black',square=True,cmap='summer')
number_of_room_in_tube = data['tube_name'].value_counts().sort_values(ascending=True)



dt = [go.Pie(

        labels = number_of_room_in_tube.index,

        values = number_of_room_in_tube.values,

        hoverinfo = 'label+value'

    

)]



plotly.offline.iplot(dt, filename='active_category')
shanghai=data[data.city=='shanghai']

beijing=data[data.city=='beijing']

shenzhen=data[data.city=='shenzhen']

guangzhou=data[data.city=='guangzhou']



guangdong=data[(data.city=='guangzhou')|(data.city=='shenzhen')]

guangdong = guangdong.groupby('tube_name').first()



print(len(shanghai))

print(len(beijing))

print(len(shenzhen))

print(len(guangzhou))

print(len(guangdong))
sh_map = folium.Map(location=[31.216199,121.469405],

                        zoom_start=12,

                   tiles="cartodbpositron")





for i in range(275):

    lat = shanghai['latitude'].iloc[i] 

    long = shanghai['longitude'].iloc[i] 

    radius = shanghai['rent_room'].iloc[i]/550 



    if shanghai['rent_room'].iloc[i] > 4500:

        color = "#008080"  # 蓝色为高价房

    elif shanghai['rent_room'].iloc[i] < 3000:

        color = "#9BCD9B"  # 灰色为低价房

    else:

        color = "#9C9C9C"  #绿色为平价房

    

    popup_text = """城市 : {}<br>

                楼层 : {}<br>

                租金 : {}<br>

                朝向 : {}<br>

                面积 : {}<br>

                房间数量 : {}<br>

                居住便利度 : {}<br>

                街区 : {}<br>"""

    popup_text = popup_text.format(shanghai['city_cn'].iloc[i] ,

                               shanghai['floor'].iloc[i] ,

                               shanghai['rent_room'].iloc[i] ,

                               shanghai['towards'].iloc[i] ,

                               shanghai['price_area'].iloc[i] ,

                               shanghai['room_no'].iloc[i],

                               shanghai['convenience'].iloc[i],

                               shanghai['tube_name'].iloc[i]

                               )

    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(sh_map)



sh_map
gz_map = folium.Map(location=[23.133180,113.319719],

                        zoom_start=13,

                   tiles="CartoDB dark_matter")





for i in range(819):

    lat = guangzhou['latitude'].iloc[i] 

    long = guangzhou['longitude'].iloc[i] 

    radius = guangzhou['rent_room'].iloc[i]/490 



    if guangzhou['rent_room'].iloc[i] > 3000:

        color = "#008080"  # 蓝色为高价房

    elif guangzhou['rent_room'].iloc[i] < 2000:

        color = "#CD950C"  # 棕色为低价房

    else:

        color = "#9C9C9C"  #灰色为平价房

    

    popup_text = """城市 : {}<br>

                楼层 : {}<br>

                租金 : {}<br>

                朝向 : {}<br>

                面积 : {}<br>

                房间数量 : {}<br>

                居住便利度 : {}<br>

                街区 : {}<br>"""

    popup_text = popup_text.format(guangzhou['city_cn'].iloc[i] ,

                               guangzhou['floor'].iloc[i] ,

                               guangzhou['rent_room'].iloc[i] ,

                               guangzhou['towards'].iloc[i] ,

                               guangzhou['price_area'].iloc[i] ,

                               guangzhou['room_no'].iloc[i],

                               guangzhou['convenience'].iloc[i],

                               guangzhou['tube_name'].iloc[i]

                               )

    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(gz_map)



gz_map
sz_map = folium.Map(location=[22.540045,114.069211],

                        zoom_start=13,

                   tiles="cartodbpositron")





for i in range(215):

    lat = shenzhen['latitude'].iloc[i] 

    long = shenzhen['longitude'].iloc[i] 

    radius = shenzhen['rent_room'].iloc[i]/500 



    if shanghai['rent_room'].iloc[i] > 4500:

        color = "#008080"  # 蓝色为高价房

    elif shanghai['rent_room'].iloc[i] < 3000:

        color = "#9BCD9B"  # 灰色为低价房

    else:

        color = "#9C9C9C"  #绿色为平价房

    

    popup_text = """城市 : {}<br>

                楼层 : {}<br>

                租金 : {}<br>

                朝向 : {}<br>

                面积 : {}<br>

                房间数量 : {}<br>

                居住便利度 : {}<br>

                街区 : {}<br>"""

    popup_text = popup_text.format(shenzhen['city_cn'].iloc[i] ,

                               shenzhen['floor'].iloc[i] ,

                               shenzhen['rent_room'].iloc[i] ,

                               shenzhen['towards'].iloc[i] ,

                               shenzhen['price_area'].iloc[i] ,

                               shenzhen['room_no'].iloc[i],

                               shenzhen['convenience'].iloc[i],

                               shenzhen['tube_name'].iloc[i]

                               )

    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(sz_map)



sz_map
num = 20



lat = np.array(guangdong["latitude"][0:num])                       

lon = np.array(guangdong["longitude"][0:num])                        

rent_room = np.array(guangdong["area_price_level"][0:num],dtype=float)    





data1 = [[lat[i],lon[i],rent_room[i]] for i in range(num)]    



map_osm = folium.Map(location=[22.7,113.5],zoom_start=10)   

HeatMap(data1).add_to(map_osm) 



folium.TileLayer('cartodbpositron').add_to(map_osm)



map_osm
sns.distplot(data['rent_room'] , fit=norm);



(mu, sigma) = norm.fit(data['rent_room'])



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('rent_room distribution')



#Get QQ-plot

fig = plt.figure()

res = stats.probplot(data['rent_room'], plot=plt)

plt.show()

data.head()
col=['price_area', 'tube_distance','convenience', 'floor_encode', 'latitude', 'longitude','towards_encode']



X=data[col]

y=data['rent_room']



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
# 导入到lightgbm矩阵

lgb_train = lgb.Dataset(X_train, y_train, feature_name=col, categorical_feature=['floor_encode'])

lgb_test = lgb.Dataset(X_test, y_test, feature_name=col, categorical_feature=['floor_encode'], reference=lgb_train)



# 设置参数

params = {'nthread': 4,  # 进程数

              'objective': 'regression',

              'learning_rate':0.001,

              #'num_leaves': 1024, 

              #'max_depth': 10, 

              'feature_fraction': 0.7,  # 样本列采样

              'lambda_l1':0.001,  # L1 正则化

              'lambda_l2': 0,  # L2 正则化

              'bagging_seed': 100,  # 随机种子

              }

params['metric'] = ['rmse']



evals_result = {}  #记录训练结果



print('开始训练...')

gbm_start=datetime.now() 

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=3000,

                valid_sets=[lgb_train, lgb_test],

                evals_result=evals_result,

                verbose_eval=10)

gbm_end=datetime.now() 

print('spendt time :'+str((gbm_end-gbm_start).seconds)+'(s)')
ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()
ax = lgb.plot_importance(gbm, max_num_features=7)

plt.show()
df_counts = data.groupby(['rent_room', 'city_cn']).size().reset_index(name='counts')

px.scatter(df_counts,x="rent_room",y="counts",color='city_cn',size="counts",size_max=20)
data['convenience'].describe()
predict_lgb=gbm.predict(data[col]) 



data['rent_room_lgb']=predict_lgb
plt.figure(figsize=(12,6))

sns.regplot(data['price_area'],data['rent_room'],color='pink',label = 'true', marker = '+')

sns.regplot(data['price_area'],data['rent_room_lgb'],color='teal', label = 'lgb', marker = 'x')

plt.legend()

plt.xlabel('price_area')

plt.ylabel('rent_room')

plt.show()
predict=gbm.predict(test[col]) 



test['rent_room_pre']=predict
plt.figure(figsize=(12,6))

sns.regplot(test['price_area'],test['rent_room_pre'],color='teal', label = 'rent_lgb', marker = 'x')

sns.regplot(test['price_area'],test['rent_room'],color='orange',label = 'rent_label', marker = '+')

plt.legend()

plt.xlabel('price_area')

plt.ylabel('rent_room')

plt.show()
col=['price_area', 'tube_distance','convenience', 'latitude', 'longitude']



X=data[col].values

y=data['rent_room'].values



X = X[:,0:5]

Y = y[:,]
# 随机拆分训练集与测试集



train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)

 

# 全连接神经网络

model = Sequential()

input = X.shape[1]

# 隐藏层256

model.add(Dense(256, input_shape=(input,)))

model.add(Activation('relu'))

#Dropout层用于防止过拟合

model.add(Dropout(0.2))



# 隐藏层128

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.2))



# 隐藏层64

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.2))



# 回归问题输出层不需要激活函数

model.add(Dense(1))

# 用 ADAM 优化算法以及优化的最小均方误差损失函数

model.compile(loss='mean_squared_error', optimizer=Adam())

# early stoppping

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

# 训练

history = model.fit(train_X, train_y, epochs=80, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[early_stopping])

# loss曲线

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
# 预测

yhat = model.predict(test_X)

# 预测y逆标准化

inv_yhat0 = concatenate((test_X, yhat), axis=1)

inv_yhat = inv_yhat0[:,-1]

# 原始y逆标准化

test_y = test_y.reshape((len(test_y), 1))

inv_y0 = concatenate((test_X,test_y), axis=1)

inv_y = inv_y0[:,-1]

# 计算 RMSE

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y)

plt.plot(inv_yhat)

plt.show()
predict_nn=model.predict(data[col]) 



data['rent_room_pre_nn']=predict_nn
plt.figure(figsize=(12,6))

sns.regplot(data['price_area'],data['rent_room'],color='pink', label = 'true', marker = 'o')

sns.regplot(data['price_area'],data['rent_room_pre_nn'],color='orange',label = 'nn', marker = 'x')

sns.regplot(data['price_area'],data['rent_room_lgb'],color='teal',label = 'lgb',marker = '+')

plt.legend()

plt.xlabel('price_area')

plt.ylabel('rent_room')

plt.show()
def build_embedding_network():

 

    inputs = []

    embeddings = []

    

    

    input_cate_feature_1 = Input(shape=(1,))

    embedding = Embedding(4, 2, input_length=1)(input_cate_feature_1)

    embedding = Reshape(target_shape=(2,))(embedding)

    inputs.append(input_cate_feature_1)

    embeddings.append(embedding)

    

    input_cate_feature_2 = Input(shape=(1,))

    embedding = Embedding(39, 6, input_length=1)(input_cate_feature_2)

    embedding = Reshape(target_shape=(6,))(embedding)

    inputs.append(input_cate_feature_2)

    embeddings.append(embedding)

    

    input_numeric = Input(shape=(5,))

    embedding_numeric = Dense(256)(input_numeric) 

    inputs.append(input_numeric)

    embeddings.append(embedding_numeric)

 

    x = Concatenate()(embeddings)

   

    x = Dense(128, activation='relu')(x)

    x = Dropout(.15)(x)

    x = Dense(64, activation='relu')(x)

    x = Dropout(.15)(x)

    

    output = Dense(1, activation='relu')(x)

    

    model = Model(inputs, output)

 

    model.compile(loss='mean_squared_error', optimizer=Adam())

    print(model.summary())

    

    return model
col=['floor_encode', 'towards_encode', 'price_area', 'tube_distance','convenience', 'latitude', 'longitude']

cate1=['floor_encode']

cate2=['towards_encode']

cont=['price_area', 'tube_distance','convenience', 'latitude', 'longitude']



X=data[col].values

y=data['rent_room'].values



train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)



# 调整train数据集

tr_cate_feature_1 = train_X[:,0:1]

tr_cate_feature_2 = train_X[:,1:2]

tr_contious_feature = train_X[:,2:7]



tr_X = []

tr_X.append(tr_cate_feature_1)

tr_X.append(tr_cate_feature_2)

tr_X.append(tr_contious_feature)



tr_label = train_y[:,]

tr_Y = []

tr_Y.append(tr_label)



# 调整test数据集

te_cate_feature_1 = test_X[:,0:1]

te_cate_feature_2 = test_X[:,1:2]

te_contious_feature = test_X[:,2:7]



te_X = []

te_X.append(te_cate_feature_1)

te_X.append(te_cate_feature_2)

te_X.append(te_contious_feature)



te_label = test_y[:,]

te_Y = []

te_Y.append(te_label)
import tensorflow as tf

import random as rn

 

#random seeds for stochastic parts of neural network 

np.random.seed(10)

from tensorflow import set_random_seed

set_random_seed(15)

 

from keras.models import Model

from keras.layers import Input, Dense, Concatenate, Reshape, Dropout

from keras.layers.embeddings import Embedding
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

embedding_nn = build_embedding_network()

history = embedding_nn.fit(tr_X, tr_Y, epochs=200, batch_size=20, validation_data=(te_X, te_Y), verbose=2, shuffle=False, callbacks=[early_stopping])
# 预测

yhat = embedding_nn.predict(te_X)

# 预测y逆标准化



inv_yhat0 = concatenate((test_X, yhat), axis=1)

inv_yhat = inv_yhat0[:,-1]

# 原始y逆标准化

test_y = test_y.reshape((len(test_y), 1))

inv_y0 = concatenate((test_X,test_y), axis=1)

inv_y = inv_y0[:,-1]

# 计算 RMSE

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y)

plt.plot(inv_yhat)

plt.show()
col=['floor_encode', 'towards_encode', 'price_area', 'tube_distance','convenience', 'latitude', 'longitude']

cate1=['floor_encode']

cate2=['towards_encode']

cont=['price_area', 'tube_distance','convenience', 'latitude', 'longitude']



cate_feature_1 = data[cate1].values

cate_feature_2 = data[cate2].values

contious_feature = data[cont].values



TX = []

TX.append(cate_feature_1)

TX.append(cate_feature_2)

TX.append(contious_feature)



TY = data['rent_room'].values



preds_emnn = embedding_nn.predict(TX)[:,0]

data['rent_room_emnn'] = preds_emnn
plt.figure(figsize=(12,6))

sns.regplot(data['price_area'],data['rent_room'],color='pink', label = 'true', marker = 'o')

sns.regplot(data['price_area'],data['rent_room_lgb'],color='gray',label = 'lgb',marker = '+')

sns.regplot(data['price_area'],data['rent_room_pre_nn'],color='orange',label = 'nn', marker = 'x')

sns.regplot(data['price_area'],data['rent_room_emnn'],color='teal',label = 'nn_em',marker = '*')

plt.legend()

plt.xlabel('price_area')

plt.ylabel('rent_room')

plt.show()