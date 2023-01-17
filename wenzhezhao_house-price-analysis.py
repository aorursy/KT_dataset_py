import os



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.style as style

import matplotlib.gridspec as gridspec

import seaborn as sns

from mpl_toolkits import mplot3d

from scipy import stats

import statsmodels.api as sm

%matplotlib inline



import lightgbm as lgb

from sklearn import preprocessing,linear_model,metrics,ensemble

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor



from scipy.stats import pearsonr



import descartes

import geopandas as gpd

from shapely.geometry import Point,Polygon



pd.set_option('max_columns',100)

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path



style.use('fivethirtyeight')
print(os.listdir('../input/houseprice/HousePrice'))
SH=pd.read_excel("../input/houseprice/HousePrice/Shanghai(7w).xlsx")
print(SH.shape)## (73009,11)

SH = SH.drop(SH[SH['行政区'].isnull()].index)### delete rows containing NaN values 

#SH = SH.loc[:,SH.columns!='编号']



print(SH.shape)## (72906,11)
### 把基本属性一列拆分成多列

SH_ordinary=SH[~SH['基本属性'].str.contains('别墅类型')]

attribute_ordinary = SH_ordinary['基本属性'].str.split('/',expand=True) 

COLUMNS_ordinary ={0:'房屋户型',1:'所在楼层',2:'建筑面积',3:'户型结构',

          4:'套内面积',5:'建筑类型',6:'房屋朝向',7:'建筑结构',

          8:'装修情况',9:'梯户比例',10:'配备电梯',11:'产权年限'}

attribute_ordinary = attribute_ordinary.rename(COLUMNS_ordinary,axis=1).drop([12,13,14],axis=1)

for i in range(attribute_ordinary.shape[1]):

    attribute_ordinary.iloc[:,i]=attribute_ordinary.iloc[:,i].str.strip().str[5:].str.strip()

    

SH_bieshu=SH[SH['基本属性'].str.contains('别墅类型')]

attribute_bieshu = SH_bieshu['基本属性'].str.split('/',expand=True) 

COLUMNS_bieshu={0:'房屋户型',1:'所在楼层',2:'建筑面积',3:'套内面积',

                4:'房屋朝向',5:'建筑结构',6:'装修情况',7:'别墅类型',

                8:'产权年限'}

attribute_bieshu = attribute_bieshu.rename(COLUMNS_bieshu,axis=1).drop([9],axis=1)

for i in range(attribute_bieshu.shape[1]):

    attribute_bieshu.iloc[:,i]=attribute_bieshu.iloc[:,i].str.strip().str[5:].str.strip()



attribute=pd.concat([attribute_ordinary,attribute_bieshu],axis=0)
### 把交易属性一列拆分成多列

def all_find(s, ch):

    return [i for i, ltr in enumerate(s) if ltr == ch]



wrong_index=all_find(SH[SH.index==47651]['交易属性'].iloc[0],'/')[-2] ### 47651 has redundant '/'

SH.loc[47651,'交易属性']=SH.loc[47651,'交易属性'][0:wrong_index]+SH.loc[47651,'交易属性'][wrong_index+1:]

transaction = SH['交易属性'].str.split('/',expand=True) 

COLUMNS ={0:'挂牌时间',1:'交易权属',2:'上次交易',3:'房屋用途',

          4:'房屋年限',5:'产权所属',6:'抵押信息',7:'房本备件'}

transaction = transaction.rename(COLUMNS,axis=1)



for i in range(transaction.shape[1]):

    if i==6:

        transaction.iloc[:,i]=transaction.iloc[:,i].str.strip().str[4:].str.strip()

    else:

        transaction.iloc[:,i]=transaction.iloc[:,i].str.strip().str[5:].str.strip()
### 整合以上

SH_aggregate = pd.concat([SH,attribute,transaction],axis=1)



SH_aggregate.drop('基本属性',axis=1,inplace =True)

SH_aggregate.drop('交易属性',axis=1,inplace =True)



del SH,attribute,transaction,SH_bieshu,SH_ordinary
### 整理总价，首付，经纬度（化为经度、纬度），所在楼层（化为所处楼段。楼总高度），建筑面积，行政区，建筑类型（与后面的板塔类型重复，删掉）。

total_price=np.array(SH_aggregate['建筑面积'].str[:-1].astype('float'))*np.array(SH_aggregate['单价'])/1e4

SH_aggregate['总价']=np.round(total_price,1)



SH_aggregate['首付']=SH_aggregate['首付'].map(lambda x:x[2:-1]).astype(float)



SH_aggregate['经度']=SH_aggregate['经纬度'].str.split(',').str[0].astype(float)

SH_aggregate['纬度']=SH_aggregate['经纬度'].str.split(',').str[1].astype(float)



SH_aggregate['所处楼段']=SH_aggregate['所在楼层'].str.split('(').str[0]

SH_aggregate['楼总高度']=SH_aggregate['所在楼层'].str.split('(').str[1].str[1:-2].astype('int')



SH_aggregate['建筑面积']=SH_aggregate['建筑面积'].map(lambda x:x[:-1]).astype(float)



SH_aggregate.drop('建筑类型',axis=1,inplace =True) ### same as '板塔类型', so drop



SH_aggregate.loc[SH_aggregate['行政区']=='闸北','行政区']='静安'



del total_price
### 整理建成时间（化为建成时间、板塔类型），建成时间（化为建成年数）

buildtime_banta = SH_aggregate['建成时间'].str.split('/',expand=True) 

buildtime_banta.rename({0:'建成时间',1:'板塔类型'},axis=1,inplace =True)



SH_aggregate.drop('建成时间',axis=1,inplace =True)



SH_aggregate=pd.concat([SH_aggregate,buildtime_banta],axis=1)



SH_aggregate['建成年数']=2020-SH_aggregate[SH_aggregate['建成时间']!='未知年建']['建成时间'].str[:4].astype('int')



### fill buildYr NaN value with community meidan 

community_med = SH_aggregate.groupby(['小区名称'])['建成年数'].median()



NaN_buildYr = SH_aggregate[SH_aggregate['建成年数'].isnull()][['小区名称','建成年数']]

NaN_buildYr['小区名称'].replace(community_med,inplace=True)

cols={'小区名称':'填充建成年数'}

NaN_buildYr.rename(cols,axis=1,inplace=True)



SH_aggregate.loc[SH_aggregate['建成年数'].isnull(),'建成年数'] = NaN_buildYr['填充建成年数']



### fill buildYr NaN value with plate meidan 

plate_med = SH_aggregate.groupby(['板块'])['建成年数'].median()



NaN_buildYr = SH_aggregate[SH_aggregate['建成年数'].isnull()][['板块','建成年数']]

NaN_buildYr['板块'].replace(plate_med,inplace=True)

cols={'板块':'填充建成年数'}

NaN_buildYr.rename(cols,axis=1,inplace=True)



SH_aggregate.loc[SH_aggregate['建成年数'].isnull(),'建成年数'] = NaN_buildYr['填充建成年数']



### fill buildYr NaN value with all data meidan

SH_aggregate.loc[SH_aggregate['建成年数'].isnull(),'建成年数'] = SH_aggregate['建成年数'].median() ##　19 Yr



del buildtime_banta,community_med,NaN_buildYr,plate_med 
### 整理梯户比例（化为梯户比），配备电梯，户型结构，挂牌时间（化为挂牌年份和挂牌天数），抵押信息

### transform '梯户比例' to ratio

ladder=SH_aggregate['梯户比例'].str.split('梯').str[0]

ladder_dict = {'一':1,'两':2,'三':3,'四':4,

               '五':5,'六':6,'八':8,'十八':18}

ladder.replace(ladder_dict,inplace=True)



family=SH_aggregate['梯户比例'].str.split('梯').str[-1].str[:-1]

family_dict={'两':2,'四':4,'三':3,'六':6,'八':8,'五':5,'七':7,'一':1,'十':10,'十二':12,'九':9,'二十四':24,'十一':11,

             '二十':20,'十六':16,'十四':14,'十五':15,'十八':18,'十三':13,'二十一':21,'二十二':22,'十七':17,'十九':19,

             '二十七':27,'二十八':28,'三十七':37,'三十一':31,'二十三':23,'三十':30,'二十九':29,'二十六':26,'三十三':33,

             '二十五':25,'三十二':32,'五十':50,'三十五':35,'三十四':34,'三十六':36,'四十':40,'三十八':38,'七十七':77,

             '八十':80,'五十二':52,'四十三':43,'九十六':96,'四十九':49,'四十七':47,'四十一':41,'四十六':46,'四十五':45,

             '六十五':65,'一百零七':107,'七十九':79,'六十九':69,'四十八':48,'七十五':75}

family.replace(family_dict,inplace=True)



SH_aggregate['梯户比']=ladder/family



del ladder,family



tihubi_null_index = SH_aggregate[SH_aggregate['梯户比'].isnull()].index

SH_aggregate.loc[tihubi_null_index,'梯户比']=1



dianti_null_index = SH_aggregate[SH_aggregate['配备电梯'].isnull()].index

SH_aggregate.loc[dianti_null_index,'配备电梯']='有'



bieshu_null_index = SH_aggregate[SH_aggregate['户型结构'].isnull()].index

SH_aggregate.loc[dianti_null_index,'户型结构']='别墅结构'



SH_aggregate['挂牌年份']=SH_aggregate['挂牌时间'].str[:4].astype(int)

SH_aggregate['挂牌时间']=SH_aggregate['挂牌时间'].astype('datetime64[D]')

current_t = pd.to_datetime('2020-01-01')

SH_aggregate['挂牌天数']=(current_t-SH_aggregate['挂牌时间']).astype(str).str.split('d').str[0].astype(int)



ydy_index = SH_aggregate[SH_aggregate['抵押信息'].str.contains('有抵押')].index

wdy_index = SH_aggregate[SH_aggregate['抵押信息'].str.contains('无抵押')].index



SH_aggregate.loc[ydy_index,'抵押信息']='有抵押'

SH_aggregate.loc[wdy_index,'抵押信息']='无抵押'



del tihubi_null_index,dianti_null_index,bieshu_null_index,ydy_index,wdy_index 
### 整理房屋户型（化为卧室、客厅、厨房、卫生间。房间总数），构造新列：小区均价，（距市中心）距离。

# SH_aggregate[SH_aggregate['房屋户型'].apply(lambda x:len(x))==9].index  [23496]

SH_aggregate['卧室']=SH_aggregate['房屋户型'].str[0].astype(int)

SH_aggregate['客厅']=SH_aggregate['房屋户型'].str[2].astype(int)

SH_aggregate['厨房']=SH_aggregate['房屋户型'].str[4].astype(int)

SH_aggregate['卫生间']=SH_aggregate['房屋户型'].str[6].astype(int)

SH_aggregate.loc[23496,'卫生间']=10

SH_aggregate['房间总数']=SH_aggregate['卧室']+SH_aggregate['客厅']+SH_aggregate['厨房']+SH_aggregate['卫生间']



SH_aggregate.drop('房屋户型',axis=1,inplace =True)



SH_aggregate['小区均价']=SH_aggregate['小区名称']

communityMed = SH_aggregate.groupby(['小区名称'])['单价'].median()

SH_aggregate['小区均价'].replace(communityMed,inplace=True)



SH_central=[121.5,31.2]

SH_aggregate['距离']=np.sqrt((SH_aggregate['经度']-SH_central[0])**2+(SH_aggregate['纬度']-SH_central[1])**2)
SH_aggregate.drop(['编号','梯户比例','别墅类型'],axis=1,inplace = True)
SH_aggregate.shape ## (72906, 43)
SH_aggregate.info()
### 从SH_aggregate中提取需要的变量，为SH_sub,并重命名为英文名。

SH_sub=SH_aggregate[['单价','总价','首付','经度','纬度','行政区','板块','建筑面积','所处楼段','楼总高度','小区均价',

                     '户型结构','建筑结构','装修情况','梯户比','配备电梯','产权年限','挂牌天数','交易权属','挂牌年份','距离',

                     '房屋用途','房屋年限','产权所属','抵押信息','房本备件','板塔类型','卧室','客厅','厨房','卫生间','房间总数','建成年数']]

English_columns={'单价':'price','总价':'totalPrice','首付':'downPay','经度':'Lng','纬度':'Lat','行政区':'district','板块':'plate',

                 '建筑面积':'square','所处楼段':'buildingSec','楼总高度':'totalHeight','户型结构':'houseType',

                 '建筑结构':'buildingStructure','装修情况':'renovationCondition','梯户比':'ladderRatio','配备电梯':'elevator',

                 '产权年限':'propertyRight','挂牌天数':'hangonDays','交易权属':'tradeOwnership','房屋用途':'houseUsage',

                 '房屋年限':'fiveYearsProperty','距离':'distance','产权所属':'publicNonpublic','抵押信息':'pledgeInfo',

                 '房本备件':'uploadPic','板塔类型':'buildingType','挂牌年份':'hangonYr','卧室':'livingRoom','客厅':'drawingRoom',

                 '厨房':'kitchen','卫生间':'bathRoom','房间总数':'totalRoom','建成年数':'buildYr','小区均价':'communityAverage'}

SH_sub=SH_sub.rename(English_columns,axis=1)
SH_sub.shape
SH_sub.info()
numerics = ['int64', 'float64']

categorical_columns = []

numerical_columns = []

features = SH_sub.columns.values.tolist()

for col in features:

    if SH_sub[col].dtype in numerics: 

        numerical_columns.append(col)

        continue

    categorical_columns.append(col)

# Encoding categorical features

for col in categorical_columns:

    if col in SH_sub.columns:

        le = LabelEncoder()

        le.fit(list(SH_sub.loc[:,col].astype(str).values))

        SH_sub.loc[:,col] = le.transform(list(SH_sub.loc[:,col].astype(str).values))



mask = np.zeros_like(SH_sub.corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(18, 15))

plt.title('SH_sub Numerical Features Pearson Correlation Matrix',fontsize=25)



sns.heatmap(SH_sub.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})

plt.show()
SH_sub.describe()
def plotting_3_chart(df, feature):

    ## Importing seaborn, matplotlab and scipy modules. 

    # style.use('fivethirtyeight')



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    

    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Shanghai house price Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1,fit=stats.norm)





    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('Shanghai house price QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Shanghai house price Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plotting_3_chart(SH_sub, 'price')
def plotting_2_chart(df, feature):

    ## Importing seaborn, matplotlab and scipy modules. 

    # style.use('fivethirtyeight')



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    

    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Shanghai log house price Histogram')

    ## plot the histogram. 

    sns.distplot(np.log1p(df.loc[:,feature]), norm_hist=True, ax = ax1,fit=stats.norm)





    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('Shanghai log house price QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(np.log1p(df.loc[:,feature]), plot = ax2)



#     ## Customizing the Box Plot. 

#     ax3 = fig.add_subplot(grid[:, 2])

#     ## Set title. 

#     ax3.set_title('Shanghai house price Box Plot')

#     ## Plotting the box plot. 

#     sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plotting_2_chart(SH_sub, 'price')
y = np.array(SH_sub.price)

plt.subplot(131)

plt.plot(range(len(y)),y,'.');plt.ylabel('price');plt.xlabel('index');

plt.subplot(132)

sns.boxplot(y=SH_sub.price)



plt.title('Shanghai house price distribution')

plt.show()
price_district=pd.DataFrame(SH_sub[['district','price']])

district_dict ={0:'Jiading',1:'Fengxian',2:'Baoshan',3:'Chongming',4:'Xuhui',5:'Putuo',6:'Yangpu',

                7:'Songjiang',8:'Pudong',9:'Hongkou',10:'Jinshan',11:'Changning',12:'Minhang',

                13:'Qingpu',14:'Jingan',15:'Huangpu'}

price_district['district'].replace(district_dict,inplace=True)



fig = plt.figure(figsize=(10,10))

##ax = plt.axes(projection="3d")

ax = plt.axes()



y_points = price_district['price']

x_points = price_district['district']



ax.scatter(x_points, y_points, c=y_points, cmap='hsv')



ax.set_xlabel('district')

ax.set_ylabel('price')

fig.autofmt_xdate()



plt.title('Shanghai house price according to district')

plt.show()
os.listdir('../input/sh-shapfile')
sh_map=gpd.read_file('../input/sh-shapfile/shang_dis_merged.shp')

sh_dict = {'崇明县':'Chongming','宝山区':'Baoshan','嘉定区':'Jiading','浦东新区':'Pudong','杨浦区':'Yangpu','闸北区':'Jingan',

           '虹口区':'Hongkou','普陀区':'Putuo','青浦区':'Qingpu','闵行区':'Minhang','长宁区':'Changning','黄浦区':'Huangpu',

           '徐汇区':'Xuhui','松江区':'Songjiang','奉贤区':'Fengxian','金山区':'Jinshan'} 

sh_map['Name'].replace(sh_dict,inplace=True)



f, ax = plt.subplots(figsize=(10, 10))



sh_map.plot(ax=ax,edgecolor='black',column = 'Name' ,categorical=True, markersize=100, legend=True, cmap='tab20')

plt.title('Shanghai district map')

plt.show()
geometry = [Point(xy) for xy in zip(SH_sub['Lng'],SH_sub['Lat'])]

geometry[:3]



crs={'init':'epsg:4326'}

geo_SH = gpd.GeoDataFrame(SH_sub,crs=crs,geometry=geometry)

SH_sub.drop('geometry',axis=1,inplace=True)



geo_SH=geo_SH[geo_SH['Lng']>=120]### Longtitude outliers of map

#geo_SH.head()
fig,ax = plt.subplots(figsize=(15,15))

sh_map.plot(ax=ax,edgecolor='black',column = 'Name' ,categorical=True, markersize=100, legend=True, cmap='tab20',alpha=0.5)

geo_SH.plot(ax=ax,markersize = 5,alpha=0.3)



plt.title('House distribution of Shanghai district map',fontsize=25)### discover four districts does not include

plt.show()
fig,ax = plt.subplots(figsize=(15,15))

sh_map.plot(ax=ax,edgecolor='black',column = 'Name' ,categorical=True, markersize=100, legend=True, cmap='tab20',alpha=0.5)

geo_SH[geo_SH['price']<5e4].plot(ax=ax,markersize = 5,color='green',alpha=0.3,label='0-5w')

geo_SH[(geo_SH['price']>=5e4)&(geo_SH['price']<10e4)].plot(ax=ax,markersize = 5,color='blue',alpha=0.3,label='5-10w')

geo_SH[(geo_SH['price']>=10e4)&(geo_SH['price']<15e4)].plot(ax=ax,markersize = 5,color='yellow',alpha=0.3,label='10-15w')

geo_SH[geo_SH['price']>15e4].plot(ax=ax,markersize = 5,color='red',alpha=0.3,label='15w+')



leg = plt.legend(loc=1, title="Price")

ax.add_artist(leg)



plt.title('House price of Shanghai district map',fontsize=25)

plt.show()
fig,ax = plt.subplots(figsize=(15,15))

sh_map.plot(ax=ax,edgecolor='black',column = 'Name' ,categorical=True, markersize=100, legend=True, cmap='tab20',alpha=0.5)

geo_SH[geo_SH['buildYr']<50].plot(ax=ax,markersize = 5,color='green',alpha=0.3,label='< 50 Yr')

geo_SH[geo_SH['buildYr']>=50].plot(ax=ax,markersize = 5,color='red',alpha=0.3,label='>= 50 Yr')



leg = plt.legend(loc=1, title="buildYear")

ax.add_artist(leg)



plt.title('House build year of Shanghai district map',fontsize=25)

plt.show()
fig,ax = plt.subplots(figsize=(15,15))

sh_map.plot(ax=ax,edgecolor='black',column = 'Name' ,categorical=True, markersize=100, legend=True, cmap='tab20',alpha=0.5)

geo_SH[geo_SH['houseType']!=0].plot(ax=ax,markersize = 5,color='green',alpha=0.3,label='No Bieshu')

geo_SH[geo_SH['houseType']==0].plot(ax=ax,markersize = 5,color='red',alpha=0.3,label='Bieshu')



leg = plt.legend(loc=1, title="Bieshu type")

ax.add_artist(leg)



plt.title('House type of Shanghai district map',fontsize=25)

plt.show()
del geo_SH
SH_sub['hangonYr'].value_counts() ## 2002:3   2003:1   2008:1   2009:1
SH_sub.groupby(['hangonYr'])['price'].mean()
SH_sub['hangonDays'].value_counts().index.sort_values()
plt.subplots(figsize=(12,9))

plt.scatter(SH_sub.groupby(['hangonDays'])['price'].mean().index,SH_sub.groupby(['hangonDays'])['price'].mean().values,s=2)

plt.title('Shanghai mean house price variation of hangonDays 483-3395')

plt.show()### shape like exponential
plt.subplots(figsize=(12,9))

sns.distplot(SH_sub['price'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(SH_sub['price'])



# plot with the distribution

plt.title('Shanghai house price distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(SH_sub['price'], plot=plt)

plt.show()
#we use log function which is in numpy

SH_sub['logPrice'] = np.log1p(SH_sub['price']) ### log1p(x): log(1+x)



#Check again for more normal distribution



plt.subplots(figsize=(12,9))

sns.distplot(SH_sub['logPrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(SH_sub['logPrice'])



# plot with the distribution

plt.title('Shanghai log house price distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(SH_sub['logPrice'], plot=plt)

plt.show()
y = SH_sub['logPrice']

# X = BJ_sub[['Lng','Lat','district','buildingSec','totalHeight','buildingType','buildingStructure',

#             'renovationCondition','ladderRatio','elevator','subway','fiveYearsProperty','totalRoom',

#            'buildYr','tradeDays']].values

X = SH_sub[['Lng','Lat','district','plate','buildingSec','totalHeight','buildingType','buildingStructure','houseType',

            'renovationCondition','ladderRatio','elevator','fiveYearsProperty','propertyRight','tradeOwnership',

           'buildYr','houseUsage','livingRoom','drawingRoom','kitchen','bathRoom','square','hangonYr','pledgeInfo',

            'publicNonpublic','uploadPic','distance','communityAverage']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
lm = linear_model.LinearRegression()

#Fit the model

lm.fit(X_train, y_train)



model_lin=sm.OLS(y_train,sm.add_constant(X_train))

result_lin=model_lin.fit()

result_lin.summary()
print("Test accuracy --> ", lm.score(X_test, y_test)*100)

print("Train accuracy --> ", lm.score(X_train, y_train)*100)
predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test.values, predictions))

print('MSE:', metrics.mean_squared_error(y_test.values, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test.values, predictions)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)



plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.plot([10, 12], [10, 12], 'k-', color = 'r')

plt.title('Shanghai house price linear model prediction versus testset real value')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(predictions, label = 'predict',alpha=0.7)

plt.title('Shanghai house price linear model prediction on test set')

plt.legend()

plt.show()
dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)

dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
print("Test accuracy --> ", dtreg.score(X_test, y_test)*100)

print("Train accuracy --> ", dtreg.score(X_train, y_train)*100)
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.title('Shanghai house price decision tree prediction versus testset real value')

plt.plot([9.5, 12], [9.5, 12], 'k-', color = 'r')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(dtr_pred, label = 'predict',alpha=0.7)

plt.title('Shanghai house price decision tree prediction on test set')

plt.legend()

plt.show()
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.1, n_estimators=500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(X_train,y_train)



lgb_pred = model_lgb.predict(X_test)

lgb_pred = lgb_pred.reshape(-1,1)
print("Test accuracy --> ", model_lgb.score(X_test, y_test)*100)

print("Train accuracy --> ", model_lgb.score(X_train, y_train)*100)
print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))

print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,lgb_pred, c='orange')

plt.plot([9.75, 11.75], [9.75, 11.75], 'k-', color = 'r')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.title('Shanghai house price lightGBM prediction versus testset real value')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(lgb_pred, label = 'predict',alpha=0.7)

plt.title('Shanghai house price lightGBM prediction on test set')

plt.legend()

plt.show()
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model_lgb,ax = axes,height = 0.5)

plt.title('Shanghai feature importance')

plt.show()
merge_pred=(0.2)*predictions+(0.3)*dtr_pred+(0.5)*lgb_pred
print('MAE:', metrics.mean_absolute_error(y_test, merge_pred))

print('MSE:', metrics.mean_squared_error(y_test, merge_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, merge_pred)))
BJ=pd.read_csv("../input/houseprice/HousePrice/Beijing(31w).csv",dtype={"id": object,"livingRoom ": object,

                                                                             "drawingRoom ":object,"bathRoom":object})
print(BJ.shape)

BJ=BJ.drop(BJ[BJ['livingRoom']=='#NAME?'].index)### 删除含有问题字符的列

print(BJ.shape)
### construct totalRoom, change tradeTime to tradeYr, tradeDays, change floor to buildingSec, totalHeight.

BJ = BJ.astype({"livingRoom": int, "drawingRoom": int, 'bathRoom':int})

BJ['totalRoom']=BJ['livingRoom']+BJ['drawingRoom']+BJ['kitchen']+BJ['bathRoom']



BJ['tradeYr']=BJ['tradeTime'].str[:4].astype(int)

BJ['tradeTime']=BJ['tradeTime'].astype('datetime64[D]')

current_t = pd.to_datetime('2020-01-01')

BJ['tradeDays']=(current_t-BJ['tradeTime']).astype(str).str.split('d').str[0].astype(int)



BJ['buildingSec']=BJ['floor'].str.split(' ').str[0]

BJ['totalHeight']=BJ['floor'].str.split(' ').str[1].astype(int)
### construct buildYr and fill NaN

BJ['buildYr']=2020-BJ[BJ['constructionTime']!='未知']['constructionTime'].str[:4].astype('int')

### fill buildYr NaN value with community median 

community_med = BJ.groupby(['Cid'])['buildYr'].median()



NaN_buildYr = BJ[BJ['buildYr'].isnull()][['Cid','buildYr']]

NaN_buildYr['Cid'].replace(community_med,inplace=True)

cols={'Cid':'fillBuildYr'}

NaN_buildYr.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['buildYr'].isnull(),'buildYr'] = NaN_buildYr['fillBuildYr']

 

### fill buildYr NaN value with district median 

district_med = BJ.groupby(['district'])['buildYr'].median()



NaN_buildYr = BJ[BJ['buildYr'].isnull()][['district','buildYr']]

NaN_buildYr['district'].replace(district_med,inplace=True)

cols={'district':'fillBuildYr'}

NaN_buildYr.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['buildYr'].isnull(),'buildYr'] = NaN_buildYr['fillBuildYr']
### buildingType

### fill buildingType NaN value with community mode 

community_mod = BJ.groupby(['Cid'])['buildingType'].agg(lambda x:x.value_counts().index[0] if x.value_counts().size>0 else np.NaN)



NaN_buildingType = BJ[BJ['buildingType'].isnull()][['Cid','buildingType']]

NaN_buildingType['Cid'].replace(community_mod,inplace=True)

cols={'Cid':'fillbuildingType'}

NaN_buildingType.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['buildingType'].isnull(),'buildingType'] = NaN_buildingType['fillbuildingType']

### fill buildingType NaN value with district mode 

district_mod = BJ.groupby(['district'])['buildingType'].agg(lambda x:x.value_counts().index[0])



NaN_buildingType = BJ[BJ['buildingType'].isnull()][['district','buildingType']]

NaN_buildingType['district'].replace(district_mod,inplace=True)

cols={'district':'fillbuildingType'}

NaN_buildingType.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['buildingType'].isnull(),'buildingType'] = NaN_buildingType['fillbuildingType']
### handle ladderRatio outliers

#### BJ[(BJ['Cid']==1111027379227)&(BJ['totalHeight']==13)]['ladderRatio']

#### from above, house with same community and same total height, ladderRatio is 0.5. So change outlier into 0.5.

BJ.loc[BJ['ladderRatio']==1.000940e+07,'ladderRatio']=0.5

# BJ[(BJ['Cid']==1111027382213)&(BJ['totalHeight']==18)&(BJ['totalRoom']==8)]['ladderRatio'].value_counts()

BJ.loc[18367,'ladderRatio']=0.5
### construct distance

BJ_central=[116.38,39.9]

# BJ['Hdist']=np.sqrt((BJ['Lng']-BJ_central[0])**2)

# BJ['Vdist']=np.sqrt((BJ['Lat']-BJ_central[1])**2)

BJ['distance']=np.sqrt((BJ['Lng']-BJ_central[0])**2+(BJ['Lat']-BJ_central[1])**2)
#### abandon price outliers

BJ=BJ[BJ['price']>=8000] ### lianjia website could find: lowest trade price 通天苑 449RMB/m2
### fill communityAverage

### fill communityAverage NaN value with communityAverage median 

community_med = BJ.groupby(['Cid'])['communityAverage'].median()



NaN_coA = BJ[BJ['communityAverage'].isnull()][['Cid','communityAverage']]

NaN_coA['Cid'].replace(community_med,inplace=True)

cols={'Cid':'fillcoA'}

NaN_coA.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['communityAverage'].isnull(),'communityAverage'] = NaN_coA['fillcoA']



### fill communityAverage NaN value with price median 

price_med = BJ.groupby(['Cid'])['price'].median()



NaN_coA = BJ[BJ['communityAverage'].isnull()][['Cid','communityAverage']]

NaN_coA['Cid'].replace(price_med,inplace=True)

cols={'Cid':'fillcoA'}

NaN_coA.rename(cols,axis=1,inplace=True)



BJ.loc[BJ['communityAverage'].isnull(),'communityAverage'] = NaN_coA['fillcoA'] 

BJ.drop('DOM',axis=1,inplace=True)
BJ.shape
BJ.info()
BJ_sub=BJ[['price','totalPrice','Lng','Lat','district','square','buildingSec','totalHeight','buildingType',

           'buildingStructure','renovationCondition','ladderRatio','elevator','followers','tradeDays','subway',

           'communityAverage','fiveYearsProperty','livingRoom','drawingRoom','kitchen','bathRoom',

           'totalRoom','buildYr','tradeYr','distance']]
BJ_sub.info()
col='buildingSec' ### only one object column

le = LabelEncoder()

le.fit(list(BJ_sub.loc[:,col].astype(str).values))

BJ_sub.loc[:,col] = le.transform(list(BJ_sub.loc[:,col].astype(str).values))



mask = np.zeros_like(BJ_sub.corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(18, 15))

plt.title('BJ_sub Numerical Features Pearson Correlation Matrix',fontsize=25)



sns.heatmap(BJ_sub.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})

plt.show()
def plotting_3_chart(df, feature):

    ## Importing seaborn, matplotlab and scipy modules. 

    # style.use('fivethirtyeight')



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    

    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Beijing house price Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1,fit=stats.norm)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('Beijing house price QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Beijing house price Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plotting_3_chart(BJ_sub, 'price')
def plotting_2_chart(df, feature):

    ## Importing seaborn, matplotlab and scipy modules. 

    # style.use('fivethirtyeight')



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    

    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Beijing log house price Histogram')

    ## plot the histogram. 

    sns.distplot(np.log1p(df.loc[:,feature]), norm_hist=True, ax = ax1,fit=stats.norm)





    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('Beijing log house price QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(np.log1p(df.loc[:,feature]), plot = ax2)



#     ## Customizing the Box Plot. 

#     ax3 = fig.add_subplot(grid[:, 2])

#     ## Set title. 

#     ax3.set_title('Shanghai house price Box Plot')

#     ## Plotting the box plot. 

#     sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plotting_2_chart(BJ_sub, 'price')
os.listdir('../input/beijing/北京市行政区划矢量文件')
bj_map=gpd.read_file('../input/beijing/北京市行政区划矢量文件/bj.shp')

bj_dict = {'大兴区':'Daxing','门头沟区':'Mentougou','海淀区':'Haidian','密云区':'Miyun','延庆区':'Yanqing',

           '怀柔区':'Huairou','平谷区':'Pinggu','石景山区':'Shijingshan','丰台区':'Fengtai','东城区':'Dongcheng',

           '顺义区':'Shunyi','房山区':'Fangshan','朝阳区':'Chaoyang','西城区':'Xicheng','昌平区':'Changping',

           '通州区':'Tongzhou'} 

bj_map['district'].replace(bj_dict,inplace=True)



f, ax = plt.subplots(figsize=(10, 10))



bj_map.plot(ax=ax,edgecolor='black',column = 'district' ,categorical=True, markersize=100, legend=True, cmap='tab20')



# geo_BJ[geo_BJ['district']==6].plot(ax=ax,markersize = 5,alpha=0.3)

plt.title('Beijing district map')

plt.show()
geometry = [Point(xy) for xy in zip(BJ_sub['Lng'],BJ_sub['Lat'])]

geometry[:3]



crs={'init':'epsg:4326'}

geo_BJ = gpd.GeoDataFrame(BJ_sub,crs=crs,geometry=geometry)

BJ_sub.drop('geometry',axis=1,inplace=True)



#geo_BJ.head()
fig,ax = plt.subplots(figsize=(15,15))

bj_map.plot(ax=ax,edgecolor='black',column = 'district' ,categorical=True, markersize=100, legend=True, cmap='tab20',alpha=0.65)

geo_BJ.plot(ax=ax,markersize = 5,alpha=0.3)



plt.title('House distribution of Beijing district map',fontsize=25)### discover four districts does not include

plt.show()
folder = Path("../input/bj-shape")



gdf = pd.concat([

    gpd.read_file(shp)

    for shp in folder.glob("*.shp") 

]).pipe(gpd.GeoDataFrame)
f, ax = plt.subplots(figsize=(15, 15))

gdf.plot(ax=ax,edgecolor='black',cmap='tab20')



geo_BJ[geo_BJ['price']<5e4].plot(ax=ax,markersize = 5,alpha=0.3,color='blue',label='0-5w')

geo_BJ[(geo_BJ['price']>=5e4)&(geo_BJ['price']<10e4)].plot(ax=ax,markersize = 5,alpha=0.3,color='yellow',label='5-10w')

geo_BJ[geo_BJ['price']>=10e4].plot(ax=ax,markersize = 5,alpha=0.3,color='red',label='10w+')



leg = plt.legend(loc=1, title="Price")

ax.add_artist(leg)



plt.title('House price distribution of Beijing subway map',fontsize=25)

plt.show()
plt.subplots(figsize=(12,9))

sns.distplot(BJ_sub['price'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(BJ_sub['price'])



# plot with the distribution

plt.title('Beijing house price distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(BJ_sub['price'], plot=plt)

plt.show()
#we use log function which is in numpy

BJ_sub['logPrice'] = np.log1p(BJ_sub['price']) ### log1p(x): log(1+x)



#Check again for more normal distribution



plt.subplots(figsize=(12,9))

sns.distplot(BJ_sub['logPrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(BJ_sub['logPrice'])



# plot with the distribution

plt.title('Beijing log house price distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(BJ_sub['logPrice'], plot=plt)

plt.show()
y = BJ_sub['logPrice']

# X = BJ_sub[['Lng','Lat','district','buildingSec','totalHeight','buildingType','buildingStructure',

#             'renovationCondition','ladderRatio','elevator','subway','fiveYearsProperty','totalRoom',

#            'buildYr','tradeDays']].values

X = BJ_sub[['Lng','Lat','district','buildingSec','totalHeight','buildingType','buildingStructure','square',

            'renovationCondition','ladderRatio','elevator','subway','fiveYearsProperty','followers',

           'buildYr','communityAverage','livingRoom','drawingRoom','kitchen','bathRoom','tradeYr',

            'distance']]
# Split data into train and test formate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
X_train.shape
#Train the model

lm = linear_model.LinearRegression()

#Fit the model

lm.fit(X_train, y_train)



model_lin=sm.OLS(y_train,sm.add_constant(X_train))

result_lin=model_lin.fit()

result_lin.summary()
print("Test accuracy --> ", lm.score(X_test, y_test)*100)

print("Train accuracy --> ", lm.score(X_train, y_train)*100)
predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test.values, predictions))

print('MSE:', metrics.mean_squared_error(y_test.values, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test.values, predictions)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)



plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.plot([9.5, 11.75], [9.5, 11.75], 'k-', color = 'r')

plt.title('Beijing house price linear model prediction versus testset real value')

plt.show()
BJ_sub['tradeYr'].value_counts() ## 2002:3   2003:1   2008:1   2009:1
focus_Yr=BJ_sub['tradeYr'].value_counts().index.sort_values()[4:] 

Yr_volume=BJ_sub['tradeYr'].value_counts()[focus_Yr].values

sns.barplot(x=focus_Yr,y=Yr_volume,palette="Blues_d")

plt.xlabel('Year')

plt.ylabel('Volume')



plt.title('Trading volume of Year 2010-2018') ### predict year of 2018

plt.show()

Yr_volume
BJ_sub.groupby(['tradeYr'])['price'].mean()[4:] ### but data of 2010,2018 is too little 
BJ_sub.groupby(['tradeDays'])['price'].mean().index
plt.subplots(figsize=(12,9))

plt.scatter(BJ_sub.groupby(['tradeDays'])['price'].mean().index,BJ_sub.groupby(['tradeDays'])['price'].mean().values,s=2)

plt.title('Beijing mean house price variation of tradeDays 703-6423')

plt.show()### shape like exponential
plt.plot(BJ_sub.groupby(['tradeYr'])['price'].mean()[4:-1])

plt.title('Beijing mean house price variation of Year 2011-2017')

plt.show()### shape like exponential
plt.plot(np.log1p(BJ_sub.groupby(['tradeYr'])['price'].mean())[4:-1])

plt.title('Beijing mean log house price variation of Year 2011-2017')

plt.show()
### use data apart from Year 2018 as training data,

### use data of Year 2018 as test data.

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

X_train = X[X['tradeYr']!=2018]

X_test = X[X['tradeYr']==2018]

y_train = y[X['tradeYr']!=2018]

y_test = y[X['tradeYr']==2018]
lm = linear_model.LinearRegression()

#Fit the model

lm.fit(X_train, y_train)



model_lin=sm.OLS(y_train,sm.add_constant(X_train))

result_lin=model_lin.fit()

result_lin.summary()
print("Test accuracy --> ", lm.score(X_test, y_test)*100)

print("Train accuracy --> ", lm.score(X_train, y_train)*100)
predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)



plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)



plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.title('Beijing house price linear model prediction versus testset real value')

plt.plot([10, 12], [10, 12], 'k-', color = 'r')



plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(predictions, label = 'predict',alpha=0.7)

plt.title('Beijing house price linear model prediction on test set')

plt.legend()

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test.values, predictions))

print('MSE:', metrics.mean_squared_error(y_test.values, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test.values, predictions)))
np.mean(np.expm1(predictions)) ### prediction of Year 2018
pre2010=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2010])))

pre2011=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2011])))

pre2012=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2012])))

pre2013=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2013])))

pre2014=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2014])))

pre2015=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2015])))

pre2016=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2016])))

pre2017=np.mean(np.expm1(lm.predict(X_train[X_train['tradeYr']==2017])))

pre2018=np.mean(np.expm1(predictions))

print('linear model predict of Year 2010: ',pre2010)

print('linear model predict of Year 2011: ',pre2011)

print('linear model predict of Year 2012: ',pre2012)

print('linear model predict of Year 2013: ',pre2013)

print('linear model predict of Year 2014: ',pre2014)

print('linear model predict of Year 2015: ',pre2015)

print('linear model predict of Year 2016: ',pre2016)

print('linear model predict of Year 2017: ',pre2017)

print('linear model predict of Year 2018: ',pre2018)
plt.figure(figsize=(10,8))

pre10to18=pd.Series([pre2010,pre2011,pre2012,pre2013,pre2014,pre2015,pre2016,pre2017,pre2018],

                    index=[2010,2011,2012,2013,2014,2015,2016,2017,2018])

plt.plot(pre10to18,color='red',label='predict')

plt.scatter(BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].index,BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].values,label='real')

plt.title('Linear model predict Beijing mean house price variation of Year 2011-2018')

plt.legend()

plt.show()
## error of data 2018 但是这个2018年数据集是有偏的，觉得这个平方误差并没有太大的预测价值。

np.mean(np.expm1(y_test)-np.mean(np.expm1(predictions)))**2
result_lin.params['tradeYr'] ### log(Pn+1/Pn) = log(Pn+1) - log(Pn) = 0.1719
## 采用上述斜率来直接拟合2010-2017北京平均房价的图像，最后得出2018年的图像 但我觉得不是经得起推敲的方法，不进行下去了。
X_train = X[X['tradeYr']!=2018]

X_test = X[X['tradeYr']==2018]

y_train = y[X['tradeYr']!=2018]

y_test = y[X['tradeYr']==2018]
dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)

dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
print("Test accuracy --> ", dtreg.score(X_test, y_test)*100)

print("Train accuracy --> ", dtreg.score(X_train, y_train)*100)
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.title('Beijing house price decision tree prediction versus testset real value')

plt.plot([10, 12], [10, 12], 'k-', color = 'r')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(dtr_pred, label = 'predict',alpha=0.7)

plt.title('Beijing house price decision tree prediction on test set')

plt.legend()

plt.show()
np.mean(np.expm1(dtr_pred))
pre2010=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2010])))

pre2011=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2011])))

pre2012=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2012])))

pre2013=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2013])))

pre2014=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2014])))

pre2015=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2015])))

pre2016=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2016])))

pre2017=np.mean(np.expm1(dtreg.predict(X_train[X_train['tradeYr']==2017])))

pre2018=np.mean(np.expm1(dtr_pred))

print('linear model predict of Year 2010: ',pre2010)

print('linear model predict of Year 2011: ',pre2011)

print('linear model predict of Year 2012: ',pre2012)

print('linear model predict of Year 2013: ',pre2013)

print('linear model predict of Year 2014: ',pre2014)

print('linear model predict of Year 2015: ',pre2015)

print('linear model predict of Year 2016: ',pre2016)

print('linear model predict of Year 2017: ',pre2017)

print('linear model predict of Year 2018: ',pre2018)
plt.figure(figsize=(10,8))

pre10to18=pd.Series([pre2010,pre2011,pre2012,pre2013,pre2014,pre2015,pre2016,pre2017,pre2018],

                    index=[2010,2011,2012,2013,2014,2015,2016,2017,2018])

plt.plot(pre10to18,color='red',label='predict')

plt.scatter(BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].index,BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].values,label='real')

plt.title('Decision tree predict Beijing mean house price variation of Year 2011-2018')

plt.legend()

plt.show()
X_train = X[X['tradeYr']!=2018]

X_test = X[X['tradeYr']==2018]

y_train = y[X['tradeYr']!=2018]

y_test = y[X['tradeYr']==2018]
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.1, n_estimators=500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(X_train,y_train)

lgb_pred = model_lgb.predict(X_test)

lgb_pred = lgb_pred.reshape(-1,1)
print("Test accuracy --> ", model_lgb.score(X_test, y_test)*100)

print("Train accuracy --> ", model_lgb.score(X_train, y_train)*100)
print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))

print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,lgb_pred, c='orange')

plt.plot([10.5, 11.75], [10.5, 11.75], 'k-', color = 'r')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.title('Beijing house price lightGBM prediction versus testset real value')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test.values,label ='Test',alpha=0.7)

plt.plot(lgb_pred, label = 'predict',alpha=0.7)

plt.title('Beijing house price lightGBM prediction on test set')

plt.legend()

plt.show()
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model_lgb,ax = axes,height = 0.5)

plt.title('Beijing feature importance')

plt.show()
np.mean(np.expm1(lgb_pred))
pre2010=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2010])))

pre2011=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2011])))

pre2012=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2012])))

pre2013=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2013])))

pre2014=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2014])))

pre2015=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2015])))

pre2016=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2016])))

pre2017=np.mean(np.expm1(model_lgb.predict(X_train[X_train['tradeYr']==2017])))

pre2018=np.mean(np.expm1(lgb_pred))

print('linear model predict of Year 2010: ',pre2010)

print('linear model predict of Year 2011: ',pre2011)

print('linear model predict of Year 2012: ',pre2012)

print('linear model predict of Year 2013: ',pre2013)

print('linear model predict of Year 2014: ',pre2014)

print('linear model predict of Year 2015: ',pre2015)

print('linear model predict of Year 2016: ',pre2016)

print('linear model predict of Year 2017: ',pre2017)

print('linear model predict of Year 2018: ',pre2018)
plt.figure(figsize=(10,8))

pre10to18=pd.Series([pre2010,pre2011,pre2012,pre2013,pre2014,pre2015,pre2016,pre2017,pre2018],

                    index=[2010,2011,2012,2013,2014,2015,2016,2017,2018])

plt.plot(pre10to18,color='red',label='predict')

plt.scatter(BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].index,BJ_sub.groupby(['tradeYr'])['price'].mean()[4:].values,label='real')

plt.title('LightGBM predict Beijing mean house price variation of Year 2011-2018')

plt.legend()

plt.show()
merge_pred=(0.2)*predictions+(0.3)*dtr_pred+(0.5)*lgb_pred
print('MAE:', metrics.mean_absolute_error(y_test, merge_pred))

print('MSE:', metrics.mean_squared_error(y_test, merge_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, merge_pred)))