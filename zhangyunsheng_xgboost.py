import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns 



sns.set() # 使用初始网格状绘图纸

myfont = matplotlib.font_manager.FontProperties(fname="../input/fontssimhei/simhei.ttf")

plt.rcParams['font.family'] = ['Times New Roman']
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
#查看训练集销售量的前5个和后5个样本

sales_train.head(5).append(sales_train.tail(5))
#训练集数据形状

print(sales_train.shape)
#测试集数据（待预测）查看

test.head(5).append(test.tail(5))
fig = plt.figure(figsize=(20,10)) # 画布大小设置

plt.subplots_adjust(hspace=.4) # 调整子图的位置距离，hspace（垂直间距）= 0.4



#查看shop_id的分布

plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=3)  # 建立3*3的网格区域，当前位置为0行0列，行方向占1个单位，列方向占据3个单位

sales_train['shop_id'].value_counts(normalize=True).plot(kind='bar', color='orangered') # 使用柱状图

plt.title('商店ID的分布状态（图1）',fontproperties=myfont)

plt.xlabel('商店ID',fontproperties=myfont)

plt.ylabel('标准化出现次数',fontproperties=myfont)



#查看item_id的分布

plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=1)  # 建立3*3的网格区域，当前位置为1行0列，行方向占1个单位，列方向占据1个单位

sales_train['item_id'].plot(kind='hist', color='deepskyblue')  # 使用直方图，颜色=蓝色

plt.title('产品ID的分布状态（图2）',fontproperties=myfont)

plt.xlabel('产品ID',fontproperties=myfont)

plt.ylabel('出现次数',fontproperties=myfont)



#查看item_price的分布

plt.subplot2grid((3,3), (1,1), rowspan=1, colspan=1)  # 建立3*3的网格区域，当前位置为1行1列，行方向占1个单位，列方向占据1个单位

sales_train['item_price'].plot(kind='hist', color='darkorange')  # 使用直方图

plt.title('产品价格的分布状态（图3）',fontproperties=myfont)

plt.xlabel('产品价格',fontproperties=myfont)

plt.ylabel('出现次数',fontproperties=myfont)



#查看item_cnt_day的分布

plt.subplot2grid((3,3), (1,2), rowspan=1, colspan=1)  # 建立3*3的网格区域，当前位置为1行2列，行方向占1个单位，列方向占据1个单位

sales_train['item_cnt_day'].plot(kind='hist', color='cornflowerblue')  # 使用直方图

plt.title('产品销售量的分布状态（图4）',fontproperties=myfont)

plt.xlabel('产品销售量',fontproperties=myfont)

plt.ylabel('出现次数',fontproperties=myfont)



#查看date_block_num的分布

plt.subplot2grid((3,3), (2,0), rowspan=1, colspan=3)  # 建立3*3的网格区域，当前位置为2行0列，行方向占1个单位，列方向占据3个单位

sales_train['date_block_num'].value_counts(normalize=True).plot(kind='bar', color='darkseagreen') # 使用柱状图

plt.title('月份数与销售记录的分布状态（图5）',fontproperties=myfont)

plt.xlabel('月份数',fontproperties=myfont)

plt.ylabel('标准化销售记录的出现次数',fontproperties=myfont)



plt.show()
#查看排序最大的前五个商品价格

sales_train['item_price'].sort_values(ascending=False)[:5]
#进一步检查该三十多万的售价商品

#在数据中找到对应整体信息

sales_train[sales_train['item_price'] == 307980]
#在商品介绍中找到对应信息

items[items['item_id'] == 6066]
#这是一种俄罗斯的防病毒产品，销售给了522人。我们需要进一步查看数据集中还有无关于此商品的记录

sales_train[sales_train['item_id'] == 6066]
#显然，只有一条记录是关于此商品名的，故可以当作一个异常值删除

sales_train = sales_train[sales_train['item_price'] < 300000]
#检查极小值端的异常情况

sales_train['item_price'].sort_values(ascending=True)[:5]
#极小端存在销售价为负的情况，在数据集中进一步检查

sales_train[sales_train['item_price'] == -1]
#查看该商品的对应信息

sales_train[sales_train['item_id'] == 2973]
#查看该商品对应的价格信息

price_info = sales_train[sales_train['item_id'] == 2973]['item_price']

price_info.describe()
#可见此商品的这个价格是不合理的，而且平均售价在2000以上，故应该删去或填充其他值

#考虑到不同商店同一个商品的售价是不同的，因此应该用其对应的32号商店中此商品的价格中位数代替

price_median = sales_train[(sales_train['shop_id'] == 32) & (sales_train['item_id'] == 2973) & (sales_train['date_block_num'] == 4) & (sales_train['item_price'] > 0)].item_price.median()

sales_train.loc[sales_train['item_price'] < 0, 'item_price'] = price_median
#查看排序最大的前五个商品销售数量

sales_train['item_cnt_day'].sort_values(ascending=False)[:5]
#找到最大销售量2169对应的数据信息

sales_train[sales_train['item_cnt_day'] == 2169]
#10月份的一天里，11373号产品在12号店就卖出了2169次。

#检查一下这个商品的对应信息。

items[items['item_id'] == 11373]
#借助翻译，这是一种和俄罗斯运输公司“Boxberry”有关的货物商品。

#继续查看此商品在其他商店的销售情况

sales_train[sales_train['item_id'] == 11373]
#查看该商品对应的销售量信息

sale_num = sales_train[sales_train['item_id'] == 11373]['item_cnt_day']

sale_num.describe()
#可见11373这个商品通常卖得很少，销售量几乎在个位数（75％=8件）。

#因此对于销售次数2169可以被认为是一个异常值，应当去除。

sales_train = sales_train[sales_train['item_cnt_day'] < 2000]
#此外注意到还有一个商品销售次数达到1000次，保险起见还需要看一下这个商品

#找到销售量1000对应的数据信息

sales_train[sales_train['item_cnt_day'] == 1000]
#检查一下这个商品的对应信息。

items[items['item_id'] == 20949]
#借助翻译，这是一种迈克品牌的白色小包。

#继续查看此商品在其他商店的销售情况

sales_train[sales_train['item_id'] == 20949]
#查看该商品对应的销售量信息

sale_num = sales_train[sales_train['item_id'] == 20949]['item_cnt_day']

sale_num.describe()
#同样地，对于20949这个商品通常卖得很少，销售量几乎在个位数（75％=7件），并且出乎意料地出现了销售量负值

#故对于销售次数1000可以被认为是一个异常值，应当去除。

sales_train = sales_train[sales_train['item_cnt_day'] < 1000]
#上面的信息提醒我们还要查验查看排序最小的哪些商品销售数量

sales_train['item_cnt_day'].sort_values(ascending=True)[:10]
#似乎许多商品都存在销售负值，这可能代表这些商品不仅没有销售反而是进货，因此我们不处理这方面的问题。

#异常值处理暂时到这里结束。
fig = plt.figure(figsize=(20,10))  # 画布

plt.subplots_adjust(hspace=.4)  # 子图间距



#查看shop_id的分布

plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=3)

test['shop_id'].value_counts(normalize=True).plot(kind='bar', color='darkviolet')

plt.title('测试集商店ID的分布状态（图6）',fontproperties=myfont)

plt.xlabel('商店ID',fontproperties=myfont)

plt.ylabel('标准化商店ID的出现次数',fontproperties=myfont)



#查看item_id的分布

plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=1)

test['item_id'].plot(kind='hist', color='sienna')

plt.title('测试集商品ID的分布状态（图7）',fontproperties=myfont)

plt.xlabel('商品ID',fontproperties=myfont)

plt.ylabel('标准化商品ID的出现次数',fontproperties=myfont)



plt.show()
#返回训练集和测试集中的商店ID唯一值的数目

shops_train = sales_train['shop_id'].nunique()

shops_test = test['shop_id'].nunique()

print('训练集中的商店ID有 {} 个 '.format(shops_train))

print('测试集中的商店ID有 {} 个 '.format(shops_test))
#虽然训练集的ID多于测试集，但这也无法保证训练集包含了出现在测试集中所有的商店。

#因此,需要检验测试集ID是否为训练集ID的一个子集。

def is_subset(set0,set1):

    if set0.issubset(set1):

        print ("二者是子集的包含关系") 

    else:

        print ("二者不是子集的包含关系")



shops_train_set = set(sales_train['shop_id'].unique())

shops_test_set = set(test['shop_id'].unique())



print('判断结果为：')

is_subset(shops_test_set,shops_train_set)
#这里确定了测试集中的所有商店id都在训练集中。

#但是在项目的竞争讨论有一个关于重复店铺的问题被提到，这可能需要我们进行分析。
#对商店名称和ID进行比较

shops
#令人意外的是这些商店名是以城市和地区作为台头的，这可能是一个潜在特征

#而且仔细分析可以发现ID为0、1的商店名与ID为57、58几乎一致，区别在于商店0、1还附加了'фран'（弗兰）一词

#还有ID为10的商店名与ID为11几乎一致，都为"Жуковский ул. Чкалова 39м"（茹科夫斯基大街·契卡洛夫39m）

#二者唯一区别在于最后的角标字符不同，分别是'？'与'2'
#因此，我认为这些几乎重复的ID项应该被合并（训练集和测试集都要）

sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57

test.loc[test['shop_id'] == 0, 'shop_id'] = 57



sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58

test.loc[test['shop_id'] == 1, 'shop_id'] = 58



sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11

test.loc[test['shop_id'] == 10, 'shop_id'] = 11
#再查看一下合并后训练集和测试集的商店ID数目

shops_train = sales_train['shop_id'].nunique()

shops_test = test['shop_id'].nunique()

print('训练集中的商店ID有 {} 个 '.format(shops_train))

print('测试集中的商店ID有 {} 个 '.format(shops_test))
#查看前五个ID的商店名

shops['shop_name'][:5]
#提取商店名中的城市名

shop_cities = shops['shop_name'].str.split(' ').str[0]

shop_cities.unique()
#仔细观察后发现雅库茨克市用了'!Якутск'和'Якутск'两种表示。

#猜测它们的含义应该是相同的，所以我们将它们将合并为一类。并且将城市名作为一个新特征放入shops数据里。

shops['city'] = shop_cities

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
#查看现在的shops数据

shops
#将city特征转换为数值标签（简单使用数字编码）

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

shops['shop_city'] = label_encoder.fit_transform(shops['city'])
#现在我们不再需要'shop_name'和'city'这两个变量了，所以将其删除

shops = shops.drop(['shop_name', 'city'], axis = 1)

shops.head()
#返回训练集和测试集中的商品ID唯一值的数目

items_train = sales_train['item_id'].nunique()

items_test = test['item_id'].nunique()

print('训练集中的商品ID有 {} 个 '.format(items_train))

print('测试集中的商品ID有 {} 个 '.format(items_test))
#类似的包含关系查验

items_train_set = set(sales_train['item_id'].unique())

items_test_set = set(test['item_id'].unique())



print('判断结果为：')

is_subset(items_test_set,items_train_set) 
#确认这些非子集商品ID的数量

len(items_test_set.difference(items_train_set)) 
#可见，测试集中存在363项是在训练集中没有的。 

#但这并不意味着针对这些商品的销售预测必须为零的，因为可以将新商品添加到训练数据中，但怎么预测它们的值是一个难题。

#在处理之前，我们需要进一步了解这个测试集中的5100个商品。它们具体属于什么类别，哪些类别我们是不需要在测试集中进行预测的。

item_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))]

cats_in_test = item_in_test.item_category_id.unique()
#查看item_cats类别数据中的不在test里的类别信息（训练集中的常见类别）

item_cats.loc[~item_cats['item_category_id'].isin(cats_in_test)]
#查看item_cats中的类别数据

item_cats['item_category_name']
#以'-'号分隔字符

cats_ = item_cats['item_category_name'].str.split('-')



#提取主类别放入item_cats中

item_cats['main_category'] = cats_.map(lambda row: row[0].strip())  # 提取前面的字符，用strip()用于删除非字符单位



#提取子类别放入item_cats中（若无子类别，则用主类别作为子类别）

item_cats['sub_category'] = cats_.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
#对新类进行数字编码

label_encoder = preprocessing.LabelEncoder()



item_cats['main_category_id'] = label_encoder.fit_transform(item_cats['main_category'])

item_cats['sub_category_id'] = label_encoder.fit_transform(item_cats['sub_category'])
item_cats.head()
#转换销售数据中的时间比哪里，获取指定形式的时间和日期：'日/月/年'

sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')  
#创建一个迭代器，生成表示item1，item2等中元素的笛卡尔积的元组

#表示从2013年1月开始生产的笛卡尔产品

from itertools import product 

shops_in_jan = sales_train.loc[sales_train['date_block_num']==0, 'shop_id'].unique()  # 取出0月份开始的商店ID数

items_in_jan = sales_train.loc[sales_train['date_block_num']==0, 'item_id'].unique()  # 取出0月份开始的商品ID数

jan = list(product(*[shops_in_jan, items_in_jan, [0]]))    # 生成商店ID数与商品ID数的笛卡尔积的元组,然后转成列表
#查看笛卡尔元组的前五个结果，元祖内从左到右位置分别表示：（商店ID，商品ID，当前月份数）

print(jan[:5])
#笛卡尔元组的总个数（表示0月份）

print(len(jan))
#2013年2月（第二个月）生产的笛卡尔产品

shops_in_feb = sales_train.loc[sales_train['date_block_num']==1, 'shop_id'].unique()

items_in_feb = sales_train.loc[sales_train['date_block_num']==1, 'item_id'].unique()

feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
#第二个月的笛卡尔元组

print(feb[:5])
#第二个月的笛卡尔元组个数

print(len(feb))
#使用numpy的'vstack'数组堆叠方法将前两个月的笛卡尔元组数据合并，并创造一个dataframe格式便于显示。

cartesian_jf = np.vstack((jan, feb))    # vstack（垂直方向）将数组堆叠。

cartesian_jf_df = pd.DataFrame(cartesian_jf, columns=['shop_id', 'item_id', 'date_block_num'])   # 创建dataframe并给不同列命名

cartesian_jf_df.head().append(cartesian_jf_df.tail())
#将所有33个月份进行相同的数据合并与df创建

months = sales_train['date_block_num'].unique()

cartesian = []

for month in months:

    shops_in_month = sales_train.loc[sales_train['date_block_num']==month, 'shop_id'].unique()

    items_in_month = sales_train.loc[sales_train['date_block_num']==month, 'item_id'].unique()

    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))

    

cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
#所有月份整合后的数据形状

cartesian_df.shape
cartesian_df.head()
#对数据集依次使用shop_id,'item_id' 和 'date_block_num' 的序列对象进行分组，然后提取出月销售量'item_cnt_day'的总和

#即可以获得特定商店的特定商品的月销售总量

x = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

x.head()
x.shape
#pd.merge()方法进行合并连接，left表示只保留左边的主键，只在右边主键中存在的行就不取了

new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0) 
#使用numpy.clip将月销售量item_cnt_month缩放到[0,20]之内，这是项目说明中提到的

new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
new_train.head()
#使用sort_values对new_train依次按是按'date_block_num','shop_id','item_id元素内部排序的先后顺序来重新排列

new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)  

new_train.head()
#删除系统中不需要的列表，释放内存

del x

del cartesian_df

del cartesian

del cartesian_jf

del cartesian_jf_df

del feb

del jan

del items_test_set

del items_train_set

del sales_train
#现在我们为测试集插入date_block_num的属性（第34个月）和销售量'item_cnt_month'属性（暂定为0）。

#使用pandas的insert方法将此新列放置在特定索引处。这便于之后将测试集于训练集的相互连接

test.insert(loc=3, column='date_block_num', value=34)        # 在测试集第三列插入月份数，赋值为34

test['item_cnt_month'] = 0  # 在测试集插入新列'item_cnt_month'，赋值为0

test.head()
#删除测试集相对new_train中不含的的ID列，并与原训练集向下连接合并

new_train = new_train.append(test.drop('ID', axis = 1)) 

new_train.head().append(new_train.tail())
#合并商店数据，以获得对应ID下编码好的的城市类别

new_train = pd.merge(new_train, shops, on=['shop_id'], how='left') 

new_train.head()
#合并商品名称数据，以获得对应ID下编码好的商品类别

new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')

new_train.head()
#合并商品类别数据，以获得对应名称下编码号的商品父子类别

new_train = pd.merge(new_train,  item_cats.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')

new_train.head()
#删除非数值的列

new_train.drop(['main_category','sub_category'],axis=1,inplace=True)

new_train.head()
#删除无用的数据，释放内存

del items

del item_cats

del shops

del test
#定义滞后特征添加函数

def generate_lag(train, months, lag_column):

    for month in months:

        # 创建滞后特征

        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()

        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]

        train_shift['date_block_num'] += month

        #新列表连接到训练集中

        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    return train
#定义向下数据类型转变函数，作用是将float64类型转变成float16，将int64转变成int16（用于缩减内存量,否则后续无法运行）

from tqdm import tqdm_notebook   # 进度读取条使用

def downcast_dtypes(df):   

    # 选择需要处理的列 

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    #开始数据转换

    df[float_cols] = df[float_cols].astype(np.float16)

    df[int_cols]   = df[int_cols].astype(np.int16)

    

    return df
#使用变换函数来更数据类型

new_train = downcast_dtypes(new_train)  
%%time

#添加目标变量（月销量属性）的滞后特征，添加部分的月销量数据

new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')
%%time

#添加商品-目标均值的滞后特征

#按月份和商品id排序并取其月销量的均值

group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

#将新表添加到new_train的右侧，对应'date_block_num', 'item_id'属性

new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')

#对[1,2,3,6,12]月进行月销量滞后添加（均值填充）

new_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')

#删除不需要的'item_month_mean'属性

new_train.drop(['item_month_mean'], axis=1, inplace=True)
%%time

#添加商店-目标均值的滞后特征

group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')

new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')

new_train.drop(['shop_month_mean'], axis=1, inplace=True)
%%time

#添加商店-商品种类-目标均值的滞后特征

group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

new_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')

new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)
%%time

#添加商品父类别-目标均值的滞后特征

group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'main_category_month_mean')

new_train.drop(['main_category_month_mean'], axis=1, inplace=True)
%%time

#添加商品子类别-目标均值的滞后特征

group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'sub_category_month_mean')

new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
#滞后特征添加后的数据集形貌

new_train.tail()
#添加一个月属性的特征

new_train['month'] = new_train['date_block_num'] % 12
#添加一个月份中休假日期特征

#每一个月份对应的休息天数字典

holiday_dict = {0: 6,

                1: 3,

                2: 2,

                3: 8,

                4: 3,

                5: 3,

                6: 2,

                7: 8,

                8: 4,

                9: 8,

                10: 5,

                11: 4}



new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)
#添加已知的第二年开始俄罗斯证券交易所交易数据(万亿)特征

moex = {12: 659, 

        13: 640, 

        14: 1231,

        15: 881,

        16: 764, 

        17: 663,

        18: 743, 

        19: 627, 

        20: 692,

        21: 736, 

        22: 680, 

        23: 1092,

        24: 657, 

        25: 863, 

        26: 720,

        27: 819, 

        28: 574, 

        29: 568,

        30: 633, 

        31: 658, 

        32: 611,

        33: 770, 

        34: 723}



new_train['moex'] = new_train.date_block_num.map(moex)
#再一次数据类型转换

new_train = downcast_dtypes(new_train)

new_train.head().append(new_train.tail())
#因为第一年没有俄罗斯证券交易的数据特征，因此从第二年开始作为输入

new_train = new_train[new_train.date_block_num > 11]
#使用0来填补，表示没有数据的样本

def fill_nan(df):

    for col in df.columns:

        if ('_lag_' in col) & (df[col].isna().any()):

            df[col].fillna(0, inplace=True)         

    return df



new_train =  fill_nan(new_train)
#训练数据的特征提取

train_feature = new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)

#训练数据的标签提取

train_label = new_train[new_train.date_block_num < 33]['item_cnt_month']
#验证数据的特征提取

val_feature = new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)

#验证数据的标签提取

val_label = new_train[new_train.date_block_num == 33]['item_cnt_month']
test_feature = new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)
train_feature.shape,train_label.shape,val_feature.shape,val_label.shape
train_feature.head()
import gc

gc.collect()
from xgboost import XGBRegressor
#设定模型参数

model = XGBRegressor(n_estimators=3000,

                     max_depth=10,

                     colsample_bytree=0.5, 

                     subsample=0.5, 

                     learning_rate = 0.01

                    )
%%time

#进行模型训练，并设置早停函数(建议在kaggle端进行)

model.fit(train_feature.values, train_label.values, 

          eval_metric="rmse", 

          eval_set=[(train_feature.values, train_label.values), (val_feature.values, val_label.values)], 

          verbose=True, 

          early_stopping_rounds = 50)
#导出预测结果

y_pred = model.predict(test_feature.values)
#特征重要性查看

importances = pd.DataFrame({'feature':new_train.drop('item_cnt_month', axis = 1).columns,'importance':np.round(model.feature_importances_,3)}) 

importances = importances.sort_values('importance',ascending=False).set_index('feature') 

importances = importances[importances['importance'] > 0.01]



importances.plot(kind='bar',

                 title = 'Feature Importance',

                 figsize = (8,6),

                 grid= 'both')
submission['item_cnt_month'] = y_pred

submission.to_csv('future_sales_pred.csv', index=False)