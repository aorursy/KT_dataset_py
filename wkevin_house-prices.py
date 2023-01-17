%matplotlib inline
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
# 注意本机练习时最好与kaggle保持版本一致，以防水土不服。
import sys
print(sys.version)
print(np.__version__)
print(pd.__version__)
print(sp.__version__)
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_allX = pd.concat([df_train.loc[:,'MSSubClass':'SaleCondition'],
                   df_test.loc[:,'MSSubClass':'SaleCondition']])
df_allX = df_allX.reset_index(drop=True)
print(df_train.shape,df_test.shape,df_allX.shape) # df_allX 少了 Id 和 SalePrice 两列
df_train.describe()
# 数值量特征
feats_numeric  = df_allX.dtypes[df_allX.dtypes != "object"].index.values
#feats_numeric = [attr for attr in df_allX.columns if df_allX.dtypes[attr] != 'object']

# 字符量特征
feats_object = df_allX.dtypes[df_allX.dtypes == "object"].index.values
#feats_object = [attr for attr in df_allX.columns if df_allX.dtypes[attr] == 'object']
#feats_object = df_train.select_dtypes(include = ["object"]).columns

print(feats_numeric.shape,feats_object.shape)
# 离散的数值量，需要人工甄别
feats_numeric_discrete  = ['MSSubClass','OverallQual','OverallCond'] # 户型、整体质量打分、整体条件打分 —— 文档中明确定义的类型量
feats_numeric_discrete += ['TotRmsAbvGrd','KitchenAbvGr','BedroomAbvGr','GarageCars','Fireplaces'] # 房间数量
feats_numeric_discrete += ['FullBath','HalfBath','BsmtHalfBath','BsmtFullBath'] # 外国人这么爱洗澡？搞这么多浴室
feats_numeric_discrete += ['MoSold','YrSold'] # 年、月，这些不看成离散的应该也行

# 连续型特征
feats_continu = feats_numeric.copy()
# 离散型特征
feats_discrete = feats_object.copy()

for f in feats_numeric_discrete:
    feats_continu = np.delete(feats_continu,np.where(feats_continu == f))
    feats_discrete = np.append(feats_discrete,f)

print(feats_continu.shape,feats_discrete.shape)
def plotfeats(frame,feats,kind,cols=4):
    """批量绘图函数。
    
    Parameters
    ----------
    frame : pandas.DataFrame
        待绘图的数据
    
    feats : list 或 numpy.array
        待绘图的列名称
        
    kind : str
        绘图格式：'hist'-直方图；'scatter'-散点图；'hs'-直方图和散点图隔行交替；'box'-箱线图，每个feat一幅图；'boxp'-Price做纵轴，feat做横轴的箱线图。
        
    cols : int
        每行绘制几幅图
    
    Returns
    -------
    None
    """
    rows = int(np.ceil((len(feats))/cols))
    if rows==1 and len(feats)<cols:
        cols = len(feats)
    #print("输入%d个特征，分%d行、%d列绘图" % (len(feats), rows, cols))
    if kind == 'hs': #hs:hist and scatter
        fig, axes = plt.subplots(nrows=rows*2,ncols=cols,figsize=(cols*5,rows*10))
    else:
        fig, axes = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*5,rows*5))
        if rows==1 and cols==1:
            axes = np.array([axes])
        axes = axes.reshape(rows,cols) # 当 rows=1 时，axes.shape:(cols,)，需要reshape一下
    i=0
    for f in feats:
        #print(int(i/cols),i%cols)
        if kind == 'hist':
            #frame.hist(f,bins=100,ax=axes[int(i/cols),i%cols])
            frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols),i%cols])
        elif kind == 'scatter':
            frame.plot.scatter(x=f,y='SalePrice',ylim=(0,800000), ax=axes[int(i/cols),i%cols])
        elif kind == 'hs':
            frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols)*2,i%cols])
            frame.plot.scatter(x=f,y='SalePrice',ylim=(0,800000), ax=axes[int(i/cols)*2+1,i%cols])
        elif kind == 'box':
            frame.plot.box(y=f,ax=axes[int(i/cols),i%cols])
        elif kind == 'boxp':
            sns.boxplot(x=f,y='SalePrice', data=frame, ax=axes[int(i/cols),i%cols])
        i += 1
    plt.show()
plotfeats(df_train,feats_continu,kind='scatter',cols=6)
plotfeats(df_train,feats_numeric_discrete,kind='scatter',cols=6)
# SalePrice 的偏离度
df_train.skew()['SalePrice']
#df_train.plot(kind='hist',y='SalePrice',bins=100)
df_train['SalePrice'].plot(kind='hist',y='SalePrice',bins=100) # 为了和下面的图做对比才使用这行的
#sns.distplot(df_train['SalePrice'], fit='norm');
#plt.hist(df_train['SalePrice'],bins=100)
stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'].apply(lambda x: np.log1p(x)).plot(kind='hist',y='SalePrice',bins=100)
# 计算各列自己的偏离度
skewed = df_allX[feats_numeric].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
#skewed = df_allX[feats_numeric].skew().sort_values(ascending=False)
skewed[:10]
# 用直方图和散点图（SalePrice之间）对比展示偏离度
plotfeats(df_train,skewed[:6].index,kind='hs',cols=6)
df_train.kurt()['SalePrice']
# 计算各列自己的峰度
kurted = df_allX[feats_numeric].kurt().sort_values(ascending=False)
kurted[:10]
# 用直方图和散点图（SalePrice之间）对比展示峰度
plotfeats(df_train,kurted[:6].index,kind='hs',cols=6)
plotfeats(df_train,feats_numeric,kind='box',cols=6)
# 由于没有标准化，比例尺差异巨大，此处的绘图不具参考意义，待标准化后的数据才可以

plt.figure(figsize=(16,10))

plt.subplot(121)
sns.boxplot(data=df_allX[feats_continu],orient="h")

plt.subplot(122)
sns.boxplot(data=df_allX[feats_discrete],orient="h")
plotfeats(df_train, ['OverallQual'], kind='boxp', cols=6)
plotfeats(df_train, feats_numeric_discrete, kind='boxp', cols=6)
plotfeats(df_train, feats_object, kind='boxp', cols=6)
a = np.random.random(size=(1000,))
b = np.random.random(size=1000,)
f,p = stats.f_oneway(a, b)
print(f,p)
a = np.random.randn(1000,)
b = np.random.randn(1000,)
f,p = stats.f_oneway(a, b)
print(f,p)
a = np.random.randint(1,10,size=1000,)
b = np.random.randint(1,10,size=1000,)
f,p = stats.f_oneway(a, b)
print(f,p)
a = np.random.randint(1,10,size=1000,)
b = np.random.randint(5,15,size=1000,)
f,p = stats.f_oneway(a, b)
print(f,p)
a = np.random.binomial(5,0.2,size=1000)
b = np.random.randn(1000,)
f,p = stats.f_oneway(a, b)
print(f,p)
# stats.f_oneway() 的入参是分好组的多个array 
# 本例将2列数据(自变量X、因变量Y)的dataframe转换为分组数据
def anovaXY(data):
    samples = []
    X = data.columns[0]
    Y = data.columns[1]
    for level in data[X].unique():
        if (type(level) == float): # np.NaN 的特殊处理
            s = data[data[X].isnull()][Y].values
        else:
            s = data[data[X] == level][Y].values
        samples.append(s)
    f,p = stats.f_oneway(*samples) # 也能用指针？
    return (f,p)
df = pd.DataFrame(columns=('feature','f','p','logp'))
df['feature'] = feats_discrete
for fe in feats_discrete:
    data = pd.concat([df_train[fe],df_train['SalePrice']],axis=1)
    f,p = anovaXY(data)
    df.loc[df[df.feature==fe].index,'f'] = f
    df.loc[df[df.feature==fe].index,'p'] = p
    df.loc[df[df.feature==fe].index,'logp'] = 1000 if (p==0) else np.log(1./p)

# OverallQual 的 p=0，说明房价和整体评价紧密相关
plt.figure(figsize=(10,4))
sns.barplot(data=df.sort_values('p'), x='feature', y='logp')
plt.xticks(rotation=90)
def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features] # 此处用的是 Series.corr() 
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.2*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')    
    plt.show()
spearman(df_train, np.delete(df_train.columns.values,-1))
corr_pearson = df_train.corr(method='pearson')
corr_spearman = df_train.corr(method='spearman')
# corrmat 是 38*38的矩阵：所以只是 numeric 的 feature 才会参与计算
corr_pearson.shape
corr_spearman.shape
# 如果不设置figsize，会出现部分数据不显示的情况
# 就是说要手工计算充分的空间给sns.heatmap() —— 这都是啥bug啊
plt.figure(figsize=(20, 20))
plt.subplot(211)
sns.heatmap(corr_pearson, vmax=.8, square=True);
plt.subplot(212)
sns.heatmap(corr_spearman, vmax=.8, square=True);
feats_d = corr_pearson.nlargest(8,'SalePrice').index
feats_d
sns.pairplot(df_train[feats_d],size=2.5)
feats_del = ['YrSold','MoSold']
df_allX.shape
df_allX.drop(feats_del, axis=1, inplace=True)  # 快意泯恩仇 ：）
# 同步修正一下4个特征名称集
for f in feats_del:
    feats_numeric  = np.delete(feats_numeric,  np.where(feats_numeric  == f))
    feats_object   = np.delete(feats_object,   np.where(feats_object   == f))
    feats_continu  = np.delete(feats_continu,  np.where(feats_continu  == f))
    feats_discrete = np.delete(feats_discrete, np.where(feats_discrete == f))
df_allX.shape
# 经过前面偏离度分析，可以观察得出下面几个feature存在离群点
feats_away = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','1stFlrSF','GrLivArea','TotalBsmtSF']
plotfeats(df_train,feats_away,kind='scatter')
ids = []
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2][['Id','GrLivArea','SalePrice']]
# 1299 是 df_train 中的 Id 列， 1298 对应 df_allX 中的 index
ids.append(1299)
ids.append(524)
df_train.sort_values(by = 'TotalBsmtSF', ascending = False)[:2][['Id','TotalBsmtSF','SalePrice']]
# 1299 又出现了，看来提供这个数据的同学是不是恶作剧啊？
df_train.sort_values(by = '1stFlrSF', ascending = False)[:2][['Id','1stFlrSF','SalePrice']]
df_train.sort_values(by = 'BsmtFinSF1', ascending = False)[:2][['Id','BsmtFinSF1','SalePrice']]
# 全部指向 1299，此行数据必须除之
df_train.sort_values(by = 'LotArea', ascending = False)[:3][['Id','LotArea','SalePrice']]
ids.append(314)
ids.append(335)
ids.append(250)
df_train.sort_values(by = 'LotFrontage', ascending = False)[:3][['Id','LotFrontage','SalePrice']]
ids.append(1299)
ids.append(935)
np.unique(ids)
print(df_train.shape,df_test.shape,df_allX.shape)
for id in np.unique(ids):
    df_train = df_train.drop(df_train[df_train.Id==id].index)
    df_allX = df_allX.drop(df_allX[df_allX.index==(id-1)].index)
print(df_train.shape,df_test.shape,df_allX.shape)
# Python 中 NaN 的类型：
print(type(None),type(np.NaN))
def NaNRatio(frame,feats):    
    """
    查找并统计 numpy.NaN 的值, feats 可以是数值型 or 字符型特征
    """
    na_count = frame[feats].isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(frame)
    na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
    return na_data[na_data['count']>0]
def NARatio(frame,feats):
    """
    查找并统计字符串 NA 的值
    """
    nadict={}
    for c in feats:        
        # 方法1：
        # frame[f][frame[f]=='NA'] —— 问题是这种方法只能比较 object 列，numeric列会报错        
        # 方法2：
        for r in frame.index:
            if 'NA'==frame.loc[r,c]:
                if 0==nadict.get(c,0):
                    nadict[c]=[]
                nadict[c].append(r)
    return nadict
def transNaNtoNumber(frame, column, method, val=0):
    """
    将 numpy.NaN 转为指定的数字
    """
    if method == 'mean':
        frame[column] = frame[column].fillna(round(frame[column].mean()))
    elif method == 'min':
        frame[column] = frame[column].fillna(round(frame[column].min()))
    elif method == 'max':
        frame[column] = frame[column].fillna(round(frame[column].max()))
    elif method == 'special':
        frame[column] = frame[column].fillna(val).round()
    else:
        return
    return frame
def transNaNtoNA(frame, feature):
    """
    将 numpy.NaN 转为字符串 NA
    """
    # frame[feature][df[feature].isnull()] = 'NA' # 这么写有warnning
    frame.loc[frame[feature].isnull(),feature] = 'NA'
def transNAtoNumber(frame,feat,val=0):
    """
    将字符串 NA 替换为指定数值（默认0）
    """
    for r in frame[frame[feat]=='NA'].index:
        frame.loc[r,feat] = val
    return frame
# 查看数值型特征的 NA 值的数量和比例
pd.concat([NaNRatio(df_train,feats_numeric),NaNRatio(df_test,feats_numeric)],axis=1,sort=True)
#LotFrontage：到街道的距离：取平均值
df_allX = transNaNtoNumber(df_allX,'LotFrontage','mean') 

#GarageYrBlt：车库的建造年份：因为没有车库才没有年份，所以不能取平均值，暂取最小值
df_allX = transNaNtoNumber(df_allX,'GarageYrBlt','min')

#MasVnrArea：砌墙面的面积，因为没有砌墙才导致为0，取最小值替代
df_allX = transNaNtoNumber(df_allX,'MasVnrArea','special',0)

#其他：比例较小，统一用mean替代
df_allX = df_allX.fillna(df_allX.mean())
# df_allX 中的数值型特征中已没有 np.NaN
pd.concat([NaNRatio(df_allX,feats_numeric)],axis=1)
pd.concat([NaNRatio(df_train,feats_object),NaNRatio(df_test,feats_object)],axis=1,sort=True)
for c in feats_object:
    transNaNtoNA(df_allX,c)
# df_allX 中的字符型特征中已没有 np.NaN
NaNRatio(df_allX,feats_object)
df_allX[feats_numeric] = df_allX[feats_numeric].apply(lambda x:(x-x.mean())/(x.std()))
plt.figure(figsize=(16,10))

plt.subplot(121)
sns.boxplot(data=df_allX[feats_continu],orient="h")

plt.subplot(122)
sns.boxplot(data=df_allX[feats_discrete],orient="h")
df_allX.shape
#df_allX = pd.get_dummies(df_allX[feats_object], dummy_na=True)
df_allX.shape
def encode(frame, feature, targetfeature='SalePrice'):
    ordering = pd.DataFrame()
    # 找出指定特征的水平值，并做临时df的索引
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    # 按各水平分组，并求每组房价的均值
    ordering['price_mean'] = frame[[feature, targetfeature]].groupby(feature).mean()[targetfeature]
    # 排序并为order列赋值1、2、3、……
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    return ordering
encode(df_train,'BsmtCond') # numpy.NaN 的处理不是所希望的
dfc = df_train.copy()
transNaNtoNA(dfc,'BsmtCond') # 把 NaN 转为 'NA'
encode(dfc,'BsmtCond') # 字符串 NA 能够正确处理
# 转前留证
df_allX.loc[20:30,'Alley'] 
dfc = df_train.copy()

for fb in feats_object:
    print("\r\n-----\r\n",fb,end=':')
    transNaNtoNA(dfc,fb)
    for attr_v, score in encode(dfc,fb).items():
        print(attr_v,score,end='\t')
        df_allX.loc[df_allX[fb] == attr_v, fb] = score        
# 转后验证
df_allX.loc[20:30,'Alley'] 
# 检查一遍是否还有 numpy.NaN
NaNRatio(df_allX,df_allX.columns.values)
# 检查一遍是否还有 'NA'
stillNA = NARatio(df_allX,df_allX.columns.values)
stillNA
df_allX.loc[1914:1916,'MSZoning'] #果然有
df_allX[['MSZoning','OverallQual']][df_allX['MSZoning']=='NA']
dftemp = df_allX.copy()
for sn in stillNA.keys():
    dftemp  = transNAtoNumber(dftemp,sn)
    df_allX = transNAtoNumber(df_allX,sn,dftemp[sn].mean())
df_allX.loc[1914:1916,'MSZoning']
NARatio(df_allX,df_allX.columns.values)
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train),1,float('inf'))
    return np.sqrt( 2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(y_train))).asscalar()/num_train)
import mxnet
print(mxnet.__version__)
from mxnet import nd, autograd, gluon
num_train = df_train.shape[0]
X_train = nd.array(df_allX[:num_train])
X_test  = nd.array(df_allX[num_train:].values)
y_train = nd.array(df_train.SalePrice.values)
square_loss = gluon.loss.L2Loss()
def get_net(units=128, dropout=0.1):
    net = gluon.nn.Sequential()
    with net.name_scope():  
        if units != 0:
            net.add(gluon.nn.Dense(units, activation='relu'))
        if dropout != 0:
            net.add(gluon.nn.Dropout(dropout))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
def train(net, X_train, y_train, X_test, y_test, epochs, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label) 
            loss.backward()
            trainer.step(batch_size)

        # 训练用 L2Loss，画图和返回用 RMSE Loss
        train_loss.append(get_rmse_log(net, X_train, y_train))
        if X_test is not None:
            test_loss.append(get_rmse_log(net, X_test, y_test))

        
    # 返回的是 epochs 个过程 loss
    if X_test is not None:
        return train_loss, test_loss
    else:
        return train_loss
import torch
print(torch.__version__)
from torch import nn, autograd as ag, optim
# 待补充
import tensorflow as tf
print(tf.__version__)
# 待补充
# 待补充
def k_fold_cross_valid(k, epochs, X_train, y_train, learning_rate, weight_decay, units=128, dropout=0.1, savejpg=False):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    train_loss_std_sum = 0.0
    test_loss_std_sum = 0.0

    cols = k
    rows = int(np.ceil(k/cols))
    fig, axes = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*5,rows*5))
        
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        
        net = get_net(units=units, dropout=dropout)
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test, 
            epochs, learning_rate, weight_decay)        
        print("%d-fold \tTrain loss:%f \tTest loss: %f" % (test_i+1, train_loss[-1], test_loss[-1]))
        
        axes[test_i%cols].plot(train_loss, label='train')
        axes[test_i%cols].plot(test_loss, label='test')
        
        train_loss_sum += np.mean(train_loss[-10:])
        test_loss_sum += np.mean(test_loss[-10:])
        
        train_loss_std_sum += np.std(train_loss[10:])
        test_loss_std_sum  += np.std(test_loss[10:])
    
    print("%d-fold Avg: train loss: %f, Avg test loss: %f, Avg train lost std: %f, Avg test lost std: %f" % 
          (k, train_loss_sum/k, test_loss_sum/k, train_loss_std_sum/k, test_loss_std_sum/k))

    if savejpg:
        #plt.savefig("~/house-prices/%d-%d-%.3f-%d-%d-%.3f.jpg" %(k,epochs,learning_rate,weight_decay,units,dropout))
        plt.close()
    else:
        plt.show()
        
    return train_loss_sum / k, test_loss_sum / k, train_loss_std_sum / k, test_loss_std_sum /k
# 下面先根据经验赋值一组数据，验证上面模型和算法的可行性

k=5
epochs=50
learning_rate=5
weight_decay=0
units=0
dropout=0

train_avg_loss, test_avg_loss, train_avg_loss_std, test_avg_loss_std = k_fold_cross_valid(
    k, epochs, X_train, y_train, learning_rate, weight_decay, units, dropout, savejpg=False)
# 排列组合
def expand(mulcoldf, sigcoldf):
    r = pd.DataFrame(columns=np.append(mulcoldf.columns.values, sigcoldf.columns.values))
    for x in sigcoldf.values:
        s = mulcoldf.copy()
        s[sigcoldf.columns[0]] = x[0]
        r = pd.concat([r,s])
    return r

# k,epochs,learning_rate,weight_decay,units,dropout
def get_params(k=[5],epochs=[50],learning_rate=[5,0.5],weight_decay=[0],units=[0,128],dropout=[0,0.01]):
    p = pd.DataFrame()
    p = expand(pd.DataFrame({'k':k}), pd.DataFrame({'epochs':epochs}))
    p = expand(p, pd.DataFrame({'learning_rate':learning_rate}))
    p = expand(p, pd.DataFrame({'weight_decay':weight_decay}))
    p = expand(p, pd.DataFrame({'units':units}))
    p = expand(p, pd.DataFrame({'dropout':dropout}))
    return p.reset_index(drop=True)
params = get_params(learning_rate=[0.1,0.5,1,2,3,4,5], weight_decay=[1,10,100,130,150,500], units=[64,128,256])
params[-5:]
# 自己炼丹时注意删除，我这里主要为了提交 kaggle 方便
params = params[:8]
dfrult = pd.DataFrame(columns=('k','epochs','learning_rate','weight_decay','units','dropout','train_avg_loss','test_avg_loss','train_avg_loss_std','test_avg_loss_std'))
i = 0
for param in params.values:
    print("%s %d" % ("="*80,i))
    i += 1
    k,epochs,learning_rate,weight_decay,units,dropout = param.tolist()    
    print("k-fold=%d,epochs=%d,learning_rate=%f,weight_decay=%f,units=%d,dropout=%d" % (k,epochs,learning_rate,weight_decay,units,dropout))
    
    train_avg_loss, test_avg_loss, train_avg_loss_std, test_avg_loss_std = k_fold_cross_valid(k, epochs, X_train, y_train, learning_rate, weight_decay, units, dropout, savejpg=True)
    
    temp = pd.DataFrame([[k,epochs,learning_rate,weight_decay,units,dropout,train_avg_loss,test_avg_loss,train_avg_loss_std,test_avg_loss_std]],
                columns=['k','epochs','learning_rate','weight_decay','units','dropout','train_avg_loss','test_avg_loss','train_avg_loss_std','test_avg_loss_std'])
    dfrult = pd.concat([dfrult, temp])
df = dfrult.copy()
df[:5]
df['diff'] = df['test_avg_loss'] - df['train_avg_loss']
df['sum'] = df['test_avg_loss'] + df['train_avg_loss']
df = df.sort_values('sum').reset_index(drop=True)
df[:5]
# 标杆
df.loc[df['learning_rate']==0.1].loc[df['weight_decay']==130].loc[df['units']==128].loc[df['dropout']==0.01]
df[df['diff']<0.03][:5]
def learn(epochs, X_train, y_train, test, learning_rate, weight_decay, units, dropout):
    net = get_net(units=units, dropout=dropout)
    train_loss = train(net, X_train, y_train, None, None, epochs, learning_rate, weight_decay)
    plt.plot(train_loss)
    plt.show()
    print("train loss last 10 data avg: %f" % np.mean(train_loss[-10:]))
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
# 选出那粒丹药
#k,epochs,learning_rate,weight_decay,units,dropout = (5,50,0.1,130,128,0.01)
k,epochs,learning_rate,weight_decay,units,dropout = df.iloc[0][0:6]
print("k-fold=%d,epochs=%d,learning_rate=%f,weight_decay=%f,units=%d,dropout=%d" % (k,epochs,learning_rate,weight_decay,units,dropout))

learn(epochs, X_train, y_train, df_test, learning_rate, weight_decay, units, dropout)