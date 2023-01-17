import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='whitegrid')

myfont = matplotlib.font_manager.FontProperties(family="Droid Sans Fallback")#正常显示中文

# 用来正常显示负号

plt.rcParams['axes.unicode_minus'] = False



# 查询matplotlib支持的中文字体

from matplotlib.font_manager import fontManager

import os

fonts = [font.name for font in fontManager.ttflist if

         os.path.exists(font.fname) and os.stat(font.fname).st_size > 1e6]

for font in fonts:

    print(font)

# 来自Github，详情请移至https://github.com/canghailan/Wuhan-2019-nCoV，read_csv数据接口即可

ncov_city = pd.read_csv(

    "https://raw.githubusercontent.com/canghailan/Wuhan-2019-nCoV/master/Wuhan-2019-nCoV.csv")

# 来自个人统计，从1/21日截至目前累计确诊、死亡、治愈、疑似人数；每日增加人数以及其中湖北省增加人数，每日更新

ncov_coun = pd.read_excel("../input/2019ncovfrom121/ncov.xlsx",sheet_name="全国数据")

#最后一次更新时间

ncov_city['date'].max()
print('ncov_city columns:', ncov_city.columns)

print('ncov_coun columns:', ncov_coun.columns)

ncov_city.dtypes
# 将date类型修改为datetime，将城市代码修改为object

ncov_city.date = pd.to_datetime(ncov_city.date)

ncov_city['provinceCode'] = ncov_city['provinceCode'].astype(object)

ncov_city['cityCode'] = ncov_city['cityCode'].astype(object)

ncov_city.info()
ncov_city.head()
ncov_city.tail()
countries = ncov_city['country'].unique()



print(countries)

print('\nAll Countries:', len(countries))

provinces = ncov_city[ncov_city['country'] == '中国']['province'].unique()



print(provinces)

print('\nAll Provinces Of China:', len(provinces)-1)

# 返回截至目前最新的全国各省市累计数据

ncov_new = ncov_city[ncov_city['date'] ==

                     ncov_city['date'].max()][ncov_city['country'] == '中国']

# 全国每日累计数据

ncov_new_coun = ncov_new[ncov_new['province'].isna()]

# 各省份每日数据

ncov_new_prov = ncov_new[ncov_new['province'].notna()][ncov_new['city'].isna()]

# 各城市每日数据

ncov_new_city = ncov_new[ncov_new['city'].notna()]

ncov_prov_top = ncov_new_prov.sort_values(by="confirmed", ascending=False)[

    :20].sort_values(by="confirmed", ascending=True)



plt.figure(figsize=(12,8))

plt.barh(

    y=ncov_prov_top['province'],

    width=ncov_prov_top['confirmed'],

    height=0.9,

    color='r')

plt.suptitle('全国新型冠状病毒累计确诊人数最多的二十个省份', fontsize=20 , fontproperties=myfont)

plt.yticks(ncov_prov_top['province'], fontsize=10 , fontproperties=myfont)

for a, b in zip(ncov_prov_top['province'], ncov_prov_top['confirmed']):

    plt.text(x=b+100, y=a, s=b, ha='left', va='center', fontsize=10)



plt.grid(b=False)

plt.show()

ncov_new_prov[['date', 'province', 'confirmed', 'dead', 'cured']

              ].sort_values(by='confirmed', ascending=False).set_index('date')

def ncov_rise(c, d, e, f, g):

    

    plt.figure(figsize=(12,8))

    plt.bar(ncov_coun[c], ncov_coun[d], color='r', width=0.9, label=d)

    plt.bar(ncov_coun[c], ncov_coun[e], color='k', width=0.9, label=e)

    plt.plot(ncov_coun[c], ncov_coun[f], '-', color='#FF7F24', label=f)



    plt.suptitle(g, fontsize=20, fontproperties=myfont)

    plt.legend(loc='upper left', prop = myfont)

    plt.grid(b=False)

    plt.show()

ncov_rise('日期', '全国新增确诊', '新增重症', '新增疑似', '全国新型冠状病毒每日新增确诊和重症人数')

plt.figure(figsize=(12,8))

plt.plot(ncov_coun['日期'], ncov_coun['当日密切接触'], 'r-', label='当日密切接触')

plt.plot(ncov_coun['日期'], ncov_coun['当日解除观察'], 'g-', label='当日解除观察')



plt.suptitle('全国新型冠状病毒每日新增密切接触和解除观察人数',fontsize=20,fontproperties=myfont)

plt.legend(loc='upper left',prop = myfont)

plt.grid(b=False)

plt.show()

plt.figure(figsize=(12,8))

plt.plot(ncov_coun['日期'], ncov_coun['累计死亡'], 'r-', label='累计死亡')

plt.plot(ncov_coun['日期'], ncov_coun['累计治愈'], 'g-', label='累计治愈')



plt.suptitle('全国新增冠状病毒累计死亡和治愈人数', fontsize=20, fontproperties=myfont)

plt.legend(prop=myfont)

plt.grid(b=False)

plt.show()

dead = ncov_coun['累计死亡']/ncov_coun['累计确诊']

cured = ncov_coun['累计治愈']/ncov_coun['累计确诊']



plt.figure(figsize=(12,8))

plt.plot(ncov_coun['日期'], dead, 'r-', label='滚动死亡率')

plt.plot(ncov_coun['日期'], cured, 'g-', label='滚动治愈率')

plt.plot(ncov_coun['日期'], [0.03]*len(ncov_coun['日期']),

         'b--', label='百分之三', alpha=0.7)



plt.suptitle('全国新型冠状病毒每日滚动死亡率和滚动治愈率', fontsize=20, fontproperties=myfont)

plt.legend(prop=myfont)

plt.grid(b=False)

plt.show()

still = ncov_coun['累计确诊']-ncov_coun['累计死亡']-ncov_coun['累计治愈']

today = [ncov_coun['日期'].iloc[-8], still.iloc[-1]]

date = ncov_coun[still == max(still)]['日期']



plt.figure(figsize=(12,8))

plt.plot(ncov_coun['日期'], still, 'ro-', label='剩余患者')



plt.annotate('Max:{0}'.format(max(still)),

             xy=(date, max(still)), xytext=('2020-02-20', 40000),

             arrowprops=dict(facecolor='black', shrink=0.06))

plt.annotate('Today:{0}'.format(today[1]),

             xy=(date, max(still)),

             xytext=(today[0], today[1]-500))

plt.suptitle('全国新型冠状病毒每日剩余患者人数', fontsize=20, fontproperties=myfont)

plt.legend(prop=myfont)

plt.grid(False)

plt.show()

ncov_coun['湖北确诊占比'] = ncov_coun['湖北新增确诊']/ncov_coun['全国新增确诊']

ncov_coun['湖北死亡占比'] = ncov_coun['湖北新增死亡']/ncov_coun['全国新增死亡']

ncov_coun['湖北治愈占比'] = ncov_coun['湖北新增治愈']/ncov_coun['全国新增治愈']



plt.figure(figsize=(12,8))

plt.plot(ncov_coun['日期'], ncov_coun['湖北确诊占比'], 'k-', label='湖北确诊占比')

plt.plot(ncov_coun['日期'], ncov_coun['湖北死亡占比'], 'r-', label='湖北死亡占比')

plt.plot(ncov_coun['日期'], ncov_coun['湖北治愈占比'], 'g-', label='湖北治愈占比')



plt.suptitle('全国新型冠状病毒湖北确诊和死亡和治愈占比', fontsize=20, fontproperties=myfont)

plt.legend(prop=myfont)

plt.grid(False)

plt.show()

def other_province(arg1, arg2, arg3, arg4, arg5):

    other = ncov_coun[arg1]-ncov_coun[arg3]  # 得到除湖北外其他省份新增确诊人数



    plt.figure(figsize=(12,8))

    plt.plot(ncov_coun['日期'], ncov_coun[arg3], 'k-', label=arg2)

    plt.plot(ncov_coun['日期'], other, 'g-', label=arg4)



    plt.suptitle(arg5, fontsize=20, fontproperties=myfont)

    plt.legend(prop=myfont)

    plt.grid(False)

    plt.show()

other_province('全国新增确诊','湖北新增确诊','湖北新增确诊','除湖北外新增确诊','全国新型冠状病毒湖北新增确诊和除湖北外新增确诊人数')
other_province(

    '全国新增死亡',

    '湖北新增死亡',

    '湖北新增死亡',

    '除湖北外新增死亡',

    '全国新型冠状病毒湖北新增确诊和除湖北外新增确诊人数')

other_province(

    '全国新增治愈',

    '湖北新增治愈',

    '湖北新增治愈',

    '除湖北外新增治愈',

    '全国新型冠状病毒湖北新增确诊和除湖北外新增确诊人数')

other_country = ncov_city[ncov_city['city'] == '境外输入'][[

    'date', 'province', 'city', 'confirmed', 'cured', 'dead']]

other_country[other_country['date'] == other_country['date'].max()].set_index(

    'date').sort_values('confirmed', ascending=False)

# 入境人员确诊数据从3月1日开始统计

ncov_other_coun = ncov_coun[ncov_coun['日期'] >=

                            '2020-03-01'][['日期', '全国新增确诊', '新增入境', '累计入境']]

ncov_other_coun['国内新增确诊'] = ncov_other_coun['全国新增确诊']-ncov_other_coun['新增入境']



plt.figure(figsize=(12, 8))

plt.plot(ncov_other_coun['日期'], ncov_other_coun['累计入境'])



plt.suptitle('全国新型冠状病毒境外输入累计确诊人数', fontsize=20, fontproperties=myfont)

plt.grid(False)

plt.show()

plt.figure(figsize=(12, 8))



plt.plot(

    ncov_other_coun['日期'],

    ncov_other_coun['国内新增确诊'],

    label='国内新增确诊')

plt.plot(

    ncov_other_coun['日期'],

    ncov_other_coun['新增入境'],

    label='境外新增确诊')



plt.suptitle('全国新型冠状病毒本地和境外输入确诊人数', fontsize=20, fontproperties=myfont)

for a, b, c in zip(

        ncov_other_coun['日期'], ncov_other_coun['国内新增确诊'], ncov_other_coun['新增入境']):

    plt.text(x=a, y=b+3, s=b, ha='center', va='bottom', fontsize=10)

    plt.text(x=a, y=c+3, s=c, ha='center', va='bottom', fontsize=10)

plt.legend(prop=myfont)

plt.grid(False)

plt.show()

# 返回从1月20日开始中国所有的数据

ncov_copy = ncov_city[ncov_city['date'] >=

                      '2020-01-20'][ncov_city['country'] == '中国'][ncov_city['province'].notna()]

ncov_copy = ncov_copy[['date', 'province', 'city', 'confirmed']]

# 返回各省份数据

ncov_copy_prov = ncov_copy[ncov_copy['city'].isna()]

# 返回各城市数据

ncov_copy_city = ncov_copy[ncov_copy['city'].notna()]



def prov_city(prov='湖北省', city=['武汉市']):

    ncov_prov = ncov_copy_prov[ncov_copy_prov['province'] == prov]

    plt.figure(figsize=(18, 8))

    plt.plot(ncov_prov['date'], ncov_prov['confirmed'], 'r-', label=prov)



    plt.suptitle('{0}新型冠状病毒每日累计确诊情况'.format(prov), fontsize=20, fontproperties=myfont)

    plt.legend(prop=myfont)

    plt.grid(False)



    plt.show()



    for c in city:

        ncov_city_ = ncov_copy_city[ncov_copy_city['city'] == c]

        plt.figure(figsize=(18, 8))

        plt.plot(ncov_city_['date'], ncov_city_['confirmed'], 'r-', label=c)



        plt.suptitle('{0}新型冠状病毒每日累计确诊情况'.format(c), fontsize=20, fontproperties=myfont)

        plt.legend(prop=myfont)

        plt.grid(False)



        plt.show()

prov_city('北京市', ['西安市', '保定市', '海淀区'])
