import pandas as pd
df_tmdb = pd.read_csv('../input/tmdb-movies.csv')#加载数据集
df_tmdb.head()#查看数据前五行
# 对计划使用的所有数据包进行设置
#   导入语句。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
% matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# 加载数据并检查数据类型，查看是否有缺失数据或错误数据的情况。
df_tmdb = pd.read_csv('../input/tmdb-movies.csv')
df_tmdb.info()
#查看每列的索引号和标签
for i,v in enumerate(df_tmdb.columns):
    print(i,v)
#使用.iloc来选择部分列
movies_df = df_tmdb.iloc[:,np.r_[3:6,13,15:19]]
movies_df.head(2)
#查看重复项
sum(movies_df.duplicated())
#删除重复项
movies_df.drop_duplicates(inplace = True)
rows,col = movies_df.shape
print('该数据集中共有{}部电影，共有{}个特征列。'.format(rows - 1,col))
# 判断哪些列有缺失值,有缺失值的列将返回True
movies_df.isnull().any()
#直接删除有空值的行
movies_df.dropna(subset = ['genres'],inplace = True)
rows,col = movies_df.shape
print('经过清理后，总共保留了{}部电影。'.format(rows - 1))
movies_df.dtypes
#将'release_date'列转换为date格式
movies_df.release_date = pd.to_datetime(movies_df['release_date'])
movie_high = movies_df.query('vote_average >= 7.5 and vote_count >= 1000')#筛选出评价7.5分及以上，评价人数超过1000人的电影
movie_high.head()
genre_name = set() #创建一个去重框
#遍历出电影的类别集合
for x in movie_high['genres']:
    genre_name.update(x.split('|'))
genre_name
genres_df = pd.DataFrame()#创建新数据框
#对各电影类型进行one_hot编码，包含则编码为1，不包含则为0
#参考https://blog.csdn.net/tengyuan93/article/details/78930285
for gen in genre_name:
    genres_df[gen] = movie_high['genres'].str.contains(gen).map(lambda x:1 if x else 0)
genres_df.head()
genresum = genres_df.sum().sort_values(ascending = False)#加合所有的电影类别并按倒序排列
#绘制柱形图
genresum.plot(kind = 'bar');
plt.title('Genres of High Score Film',fontsize = 15)
plt.xlabel('genres',fontsize = 12)
plt.ylabel('count',fontsize  = 12);
movies_df.head(2)
#提取月份并创建一个新列
movies_df['month'] = movies_df['release_date'].dt.month
movies_df.head(2)
month_release = movies_df.groupby(['month']).count()#按月份进行分组计算每月上映电影总数
month_release['count'] = month_release['budget']
#绘制每月上映电影数量柱形图
month_name = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nav','Dec']
plt.plot(month_name,month_release['count'].values,marker = 'o');
plt.title('Numbers of Film Released in each Month',fontsize = 15)
plt.xlabel('Month',fontsize = 12)
plt.ylabel('count',fontsize = 12);
month_revenue = movies_df.groupby(['month'])['revenue'].sum()#按月份进行分组对票房收入进行加和
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(month_name,month_release['count'].values,marker = 'o',color = 'red');
ax1.set_ylabel('count',fontsize = 12)
ax1.set_xlabel('month',fontsize = 12)
plt.legend('count')
ax2 = ax1.twinx()#将上映电影数图与票房收入图合并呈现
plt.bar(month_name,month_revenue.values,alpha = 0.4);
ax2.set_ylabel('revenue',fontsize = 12)
plt.title('Quanlity and Revenue in Each Month',fontsize = 15);
sns.set_style('dark')
