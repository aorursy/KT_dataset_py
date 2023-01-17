import lxml

import requests

import os

from bs4 import BeautifulSoup

import time

import json



#构造请求网页函数，本函数功能为采集用户主页的信息，并对信息进行解析，最后返回每个视频的名称以及AV号

def get_page11(url):

    #构造请求头

    header = {"Referer": "https://www.bilibili.com/v/anime/serial/?spm_id_from=333.334.b_7072696d6172795f6d656e75.8",

          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}

    #请求网页数据

    r = requests.get(url,headers = header, timeout=30)

    #将得到的网页数据进行解析，并以字典形式返回

    soup = BeautifulSoup(r.text,'html.parser')

    webtext = json.loads(soup.text)['data']['list']['vlist'] #将soup数据转为json格式进行解析

    video = {}

    for i in range(len(webtext)):

        #将视频名称和av好分别作为键和值存入字典中

        video[webtext[i]['title']] = webtext[i]['aid']

    

    return video



videos = []

for page in range(1,11):

    #迭代采集数据

    url = f'https://api.bilibili.com/x/space/arc/search?mid=546195&ps=30&tid=0&pn={page}&keyword=&order=pubdate&jsonp=jsonp'

    videos.append(get_page11(url))

    time.sleep(1)

    

#不知道为什么这一段数据在kaggle上会报错，如果在运行时报错，则跳过此段采集数据部分，直接进入数据预处理部分
videos[1] #查看数据
#定义一个传入特定URL，然后返回相应soup解析格式数据的函数

def get(url):

    header = {"Referer": "https://www.bilibili.com/v/anime/serial/?spm_id_from=333.334.b_7072696d6172795f6d656e75.8",

              "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}

    #请求数据

    aa = requests.get(url,headers = header,timeout = 10)

    aaa =  BeautifulSoup(aa.text,'html.parser')#bs4解析

    return aaa
import re #引入正则表达式库

def get_data(aid):

    #为函数传入视频的av号，构造相应链接以供采集

    url1 = f'https://www.bilibili.com/video/av{aid}'

    url2 = f'https://api.bilibili.com/x/web-interface/archive/stat?aid={aid}'

    

    #调用前段代码构造的函数

    sss = str(get(url1).select('span'))

    

    #css选择器解析时间

    #time = get(url1).select('div.video-data:nth-child(2) > span:nth-child(2)')[0].text

    #解析发布时间，由于B站的网页存在不同版本公用的关系，这对我原本依靠css选择器采集造成了一些困扰，所以这里我选择使用正则表达式解析数据

    time  = re.search(r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2})",sss).groups(0)[0]

    

    #将采集数据转化为json格式以方便采集数据

    view = json.loads(get(url2).text)['data']['view']

    danmanku = json.loads(get(url2).text)['data']['danmaku']

    favorite = json.loads(get(url2).text)['data']['favorite']

    coin = json.loads(get(url2).text)['data']['coin']

    like = json.loads(get(url2).text)['data']['like']

    

    #把采集到的数据存入列表

    data = [aid,time,view,danmanku,favorite,coin,like]

    #每次采集结束后，都将以这个av号所对应的数据以列表形式返回

    return data
get_data(987974)

#样例数据采集
get_data(76543679)

#如果使用css选择器解析时间数据的话，那么此行代码和上一行代码就会返回不同数据
import time

import numpy as np

#迭代采集所有视频数据

for i in range(len(videos)):

    for video in videos[i].items():

        #可能会存在运行到一半报错的问题，所以我加入了一段判断命令

        if isinstance(video[1],list):

            pass

        else:

            videos[i][video[0]] = get_data(video[1])

            #time.sleep(2)
for i in videos[0].items():

    print(i[1])

#查看数据
import csv

import codecs

#将数据写入csv文件



columns = ['video name','aid','time','view','danmanku','favorite','coin','like']#构造CSV列表头数据



with codecs.open('tomato.csv','w',"utf_8_sig") as file:      

    myWriter=csv.writer(file)

    myWriter.writerow(columns)

    for b in range(len(videos)):

        for i in videos[b].items():

            tmp = [i[0]]

            tmp.extend(i[1])

            #原先的数据格式为字典格式，键与值分别为名称和列表形式的其他数据，所以上部分代码意为将每行数据存入一个临时列表中

            myWriter.writerow(tmp)
#我也上传了已经爬取成功后的文件，运行以下代码以查看数据

import pandas as pd

df= pd.read_csv('/kaggle/input/csvsss/tomatoo.csv',header=0)

df.head()
from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
df.index = pd.to_datetime(df.time)

df = df.drop(columns = ['time','aid'] )
plt.subplots(figsize = (10,6),dpi = 150)

plt.style.use("seaborn-darkgrid")         

plt.bar(df.index,df.view,width = 0.6)
df_Year =df.groupby(df.index.to_period('Y')).count() #以年为单位对数据进行采样，同时计算每年发布的视频数

df_Year.head()
import numpy as np



explode = (0, 0, 0, 0,0.03,0.03,0.03)  # 播放量较多的几年（2017-2019年）突出显示



cmap = plt.get_cmap("viridis")

colors = cmap(np.arange(7)*40) #设定色彩



fig1, ax1 = plt.subplots(figsize = (8,5),dpi = 100)

fig1 = plt.style.use("seaborn-darkgrid")



ax1.pie(df_Year.view.values, explode=explode, labels=df_Year.index, autopct='%1.1f%%',

        shadow=True, startangle=90,colors = colors) #绘制饼图

ax1.axis('equal')  # 确认绘制为正圆形的饼图

plt.show()
df_0709 = df['2019':'2017'] #选中索引在2017-2019区间的数据

df_0709.head()
plt.subplots(figsize = (10,6),dpi = 150)

plt.style.use("seaborn-darkgrid")         

plt.bar(df_0709.index.values,df_0709.view.values)
df_Q_cnt =df_0709.groupby(df_0709.index.to_period('Q')).count()

df_Q_mean =df_0709.groupby(df_0709.index.to_period('Q')).mean() #添加计数行，由于数据无太大特点，所以此行在后续操作中无太大提及

df_Q_mean['coount'] = df_Q_cnt.view

df_Q_mean
import matplotlib

myfont = matplotlib.font_manager.FontProperties(fname="/kaggle/input/chinesecharacter/fangzheng_heiti.TTF")

matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np



fig, ax1 = plt.subplots(figsize = (8,5),dpi = 150)

#|plt.rcParams['font.sans-serif'] = ['SimHei'] #设置字体和画板



x = np.arange(len(df_Q_mean.index.strftime('%Y-%m')))  # 设置x轴数据

ax1.set_ylabel('播放量',fontproperties=myfont) #设置轴所对应的数据名称



ax1.bar(x+0.2, df_Q_mean.view,width = 0.2,label = '播放量',color = 'y') #绘制柱状图

ax1.set_xticklabels(df_Q_mean.index.strftime('%Y-Q%q'), rotation=40, ha="right",fontproperties=myfont) # 标记横坐标刻度



ax1.tick_params(axis='y')



ax2 = ax1.twinx()  # 共享两个画布的横轴





ax2.tick_params(axis='y')

ax2.bar(x-0.2, df_Q_mean.coin,width = 0.2,label = '投币数') #绘制柱状图

ax2.bar(x, df_Q_mean.like,width = 0.2,label = '点赞数')

ax2.set_ylabel('投币数和点赞数',fontproperties=myfont) # 设置轴所对应的数据名称



plt.xticks(x, df_Q_mean.index.strftime('%Y-Q%q'),rotation=40) # 这句话好像和上述的功能重复了，但是我在删除后出现了bug，所以这一段代码仍然保留，等我有时间再来debug吧

ax1.legend(loc = 0,prop=myfont) #设置标签显示

ax2.legend(loc = 0,prop=myfont)



fig.tight_layout()  

plt.show()
from sklearn.preprocessing import MinMaxScaler

#这里选用MIN-MAX归一化方法处理数据

minmax = MinMaxScaler().fit_transform(df_0709[['view','coin','favorite','like']].sort_index(ascending=True))

minmax.shape
plt.subplots(figsize = (10,8),dpi= 100)

plt.plot(minmax[:,0],label = '浏览量')

plt.plot(minmax[:,1],label = '投币数')

plt.plot(minmax[:,2],label = '收藏数')

plt.plot(minmax[:,3],label = '点赞数')



plt.xticks(range(0,205,20), df_Q_mean.index.strftime('%Y-Q%q'),rotation=40) #设置坐标显示

plt.legend(loc = 0,prop=myfont)
import seaborn as sns

plt.subplots(figsize = (8,5),dpi= 150)



# 得到特征和目标拼合后的 DataFrame

df_pear = df.iloc[:,1:6]

sns.heatmap(df_pear.corr(), square=True, annot=True, cmap="YlGnBu")  # corr() 函数计算皮尔逊相关系数

#绘制热力图
df[df['video name'].str.contains("杀手")]
df_kiler = df[df['video name'].str.contains("杀手")]

df_escort = df[df['video name'].str.contains("镖客")]

df_detective = df[df['video name'].str.contains("大侦探")]

df_batman = df[df['video name'].str.contains("蝙蝠侠")][:-3]

df_student = pd.concat([df[df['video name'].str.contains("小学生")][:-2],df[df['video name'].str.contains("硬核黑叔叔")]])

df_pokemon = pd.concat([df[df['video name'].str.contains("口袋妖怪解说")],df[df['video name'].str.contains("赵赵")]]) #这两者的名称规律不大，所以筛选代码较长
#计算各短片集数据的均值，并将其合并至一个DataFrame中

index = ['杀手','镖客','大侦探','蝙蝠侠','行尸走肉','口袋妖怪']

df_pile = pd.DataFrame([df_kiler.mean(),df_escort.mean(),df_detective.mean(),df_batman.mean(),df_student.mean(),df_pokemon.mean()],index = index)


fig, ax1 = plt.subplots(figsize = (10,6),dpi = 120)

ax1.bar(np.arange(6),df_pile.view,label = '播放',width = 0.15,color = 'y')

plt.xticks(np.arange(6)+0.3, index,fontproperties=myfont) #X轴刻度展示



ax1.tick_params(axis='y')



ax2 = ax1.twinx()  # 由于数据规模有偏差，所以我将不同数据分在了不同轴上



ax2.bar(np.arange(6)+0.15,df_pile.like,label = '点赞',width = 0.15)

ax2.bar(np.arange(6)+0.3,df_pile.coin,label = '投币',width = 0.15 )

ax2.bar(np.arange(6)+0.45,df_pile.favorite,label = '收藏',width = 0.15 )

ax2.bar(np.arange(6)+0.6,df_pile.danmanku,label = '弹幕',width = 0.15 )



ax2.legend(bbox_to_anchor=(1, 0.9),prop=myfont) 

ax1.legend(bbox_to_anchor=(1, 0.94),prop=myfont) 



plt.show()


fig, ax1 = plt.subplots(figsize = (8,5),dpi = 120)

ax1.bar(np.arange(6),df_pile.coin,label = '投币')

plt.xticks(np.arange(6), index,fontproperties=myfont)



ax1.tick_params(axis='y')



ax1.legend(bbox_to_anchor=(1, 0.94),prop=myfont) 



plt.show()
#安装相应的库



!pip install jieba

!pip install wordcloud
import jieba

import jieba.analyse

import wordcloud  # 词云展示库

from PIL import Image  # 图像处理库
name = df[['video name']].reset_index().drop(['time'],axis = 1)

with open('name_tomato.csv', 'w') as f:

    name.to_csv('name_tomato.csv')  # 先将数据存档，避免后续调试常从头开始。
name = pd.read_csv('name_tomato.csv',index_col = [0])

name.head()
from tqdm import tqdm  # 使用进度条

import re



punctuation = """【】★“”！，。？、~@#￥%……&*（）！？｡――＂＃＄％＆＇<<>>（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""

re_punctuation = "[{}]+".format(punctuation)

remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、',

                u'中', u'-', u'在', u'了', u'通常', u'如果', u'我们', u'需要', u'什么', u'下', u'一', u'吗',

                u'有',u')',u'(',u'我',u'你']  # 自定义去除词库



#老番茄的标题字段中具有相当部分的词语会被分词，所以我们将绝大多数具有特征性的词语作为固有词汇，防止被分词

words = ['老番茄','带你','最骚','up主','教你','吃鸡']

for word in words:

    jieba.add_word(word, tag="nr") 

 



name_data = []

for name_ in tqdm(name['video name']):

    name_ = str(name_)

    name_ = re.sub(re_punctuation, '', name_)  # 去掉一些没用的符号



    seg_list_exact = jieba.cut(name_, cut_all=False)  # 精确模式分词

    for word in seg_list_exact:

        if word not in remove_words:  # 如果不在去除词库中

            name_data.append(word)  # 将分词加入 list 中。



len(name_data)
from collections import Counter



word_c = Counter(name_data)  # 对分词词频统计

word_top10 = word_c.most_common(10)  # 获取前10的词

print(word_top10)
plt.subplots(figsize = (10,6),dpi = 120)

wc = wordcloud.WordCloud(

    font_path='../input/chinesecharacter/fangzheng_heiti.TTF', max_words=200, max_font_size=100,background_color='white',scale = 10)

wc.generate_from_frequencies(word_c)  # 从字典生成词云

plt.imshow(wc)  # 显示词云

plt.axis('off')  # 关闭坐标轴

plt.show()  # 显示图像