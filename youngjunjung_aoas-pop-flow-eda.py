import numpy as np 
import pandas as pd 
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import geopandas as gpd
import matplotlib.font_manager as fm
import folium
import folium.plugins
from IPython.display import IFrame
import json
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import DualMap
from tqdm.notebook import tqdm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
fm.get_fontconfig_fonts()
font = fm.FontProperties(fname='../input/font-list/NanumBarunGothic.ttf')
matplotlib.rc('font', family=font.get_name())
plt.style.use('ggplot')
pd.set_option('display.float_format', '{:.2f}'.format)

time_df_list = []
age_df_list = []
for x in [201902,201903,201904,201905,202002,202003,202004,202005]:
    time_df_list.append('../input/sk-pop-flow/SK_유동인구/4_FLOW_TIME_'+str(x)+'.CSV')
    age_df_list.append('../input/sk-pop-flow/SK_유동인구/4_FLOW_AGE_'+str(x)+'.CSV')
time_df = pd.concat([pd.read_csv(path,sep='|') for path in time_df_list])
age_df = pd.concat([pd.read_csv(path,sep='|') for path in age_df_list])        
print('time_df shape:',time_df.shape)
print('age_df shape:',age_df.shape)
print('time_df Has 0 value in columns:',time_df.columns[time_df.min()==0].tolist()) 
print('age_df Has 0 value in columns:',age_df.columns[age_df.min()==0].tolist())
print('00~04 male mean pop flow:',age_df.MAN_FLOW_POP_CNT_0004.mean())
print('00~04 female mean pop flow:',age_df.WMAN_FLOW_POP_CNT_0004.mean())
# SK제공 유동인구데이터 (SK텔레콤 가입자 기준 --> 00~04 영유아 집계 X)
time_df.reset_index(drop=True,inplace=True)
age_df.reset_index(drop=True,inplace=True)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['figure.figsize'] = (15,15)

geo_meta = gpd.read_file('../input/sk-pop-flow/SK_유동인구/4_.SHP')
geo_meta.head(2)
print(geo_meta[:35].SIDO_NM.unique()) # index 0 ~34 : 대구    35 ~ 68 : 서울
print(geo_meta[:35].SGNG_NM.unique())
fig,ax = plt.subplots(1,2,figsize=(32,16))
geo_meta[:35].convex_hull.plot(color='c',edgecolor='w',ax=ax[0])
ax[0].set_title('수성구/중구 지도',fontproperties=font,size=32)
ax[0].set_axis_off()
ax[1] = geo_meta[35:].convex_hull.plot(color='c',edgecolor='w',ax=ax[1])
ax[1].set_title('노원구/중구 지도',fontproperties=font,size=32)
ax[1].set_axis_off()

plt.show()
geo_json = '../input/seoulgeo/seoul-dong.geojson'
geo_meta[['X_COORD','Y_COORD']]=geo_meta[['X_COORD','Y_COORD']].apply(pd.to_numeric)
seoul_dong_name=geo_meta[geo_meta.SIDO_NM=='서울특별시'].HDONG_NM.unique()
daegu_dong_name=geo_meta[geo_meta.SIDO_NM=='대구광역시'].HDONG_NM.unique()
print(seoul_dong_name)
print(daegu_dong_name)
print('# 중복된 동 이름:',len(seoul_dong_name.tolist()+daegu_dong_name.tolist())
      -len(set(seoul_dong_name.tolist()+daegu_dong_name.tolist())))
time_df['SI_NM'] = time_df.HDONG_NM.isin(seoul_dong_name)
time_df['SI_NM']=time_df['SI_NM'].replace({True:'서울',False:'대구'})
time_df.sample(2)
time_df['dayflow']=time_df.loc[:,'TMST_00':'TMST_23'].mean(axis=1)
time_df['date']=pd.to_datetime(time_df['STD_YMD'],format='%Y%m%d')
gu_dict=geo_meta[['HDONG_CD','SGNG_NM']].set_index('HDONG_CD').to_dict()['SGNG_NM']
time_df['GU_NM'] = time_df['HDONG_CD'].apply(lambda x: gu_dict[str(x)] )
time_df['HDONG_CD'] = time_df['HDONG_CD'].apply(pd.to_numeric)
time_df.loc[time_df[(time_df['HDONG_CD']>=2000000000)&(time_df['GU_NM']=='중구')]['GU_NM'].index,'GU_NM'] = '대구_중구'
time_df.sample(7)
pop=pd.read_csv('../input/dong-pop-data/___5__2011__20200919121627.csv',encoding = 'CP949')
pop=pop.rename(columns={'행정구역(동읍면)별':'행정구역'})
dong_pop=pop[pop.항목=='총인구수 (명)'][['행정구역','2019. 05','2020. 05']].set_index('행정구역').drop(['서울특별시','대구광역시','수성구','중구','노원구'],axis=0)
dong_pop.index=dong_pop.index.str.replace('신당제5동','신당5동')
dong_pop.loc['명동',:]
dong_pop.sort_index(inplace=True)
dong_pop['2019. 05']=dong_pop['2019. 05'].astype(int)
dong_pop['2020. 05']=dong_pop['2020. 05'].astype(int)
dong_pop_avg=pd.DataFrame(dong_pop.sum(axis=1),columns=['인구'])
dong_pop_avg['인구']=(dong_pop_avg['인구']/2).astype(int)
dong_meta=pd.merge(time_df.set_index('HDONG_NM').loc[~time_df.set_index('HDONG_NM').index.duplicated(keep='first')][['SI_NM','GU_NM']].reset_index().rename(columns={'HDONG_NM':'행정구역'}),dong_pop_avg.reset_index())
dong_meta=dong_meta.join(time_df[['HDONG_NM','HDONG_CD']],how='inner').drop('행정구역',1)
dong_meta=dong_meta[['SI_NM','GU_NM','HDONG_NM','HDONG_CD','인구']]
dong_meta.head()
fig,ax=plt.subplots(2,1,figsize=(32,16))
sns.lineplot(data=time_df.set_index('date')['2019'],x=time_df.set_index('date')['2019'].index,y='dayflow',hue='HDONG_NM',ax=ax[0])
sns.lineplot(data=time_df.set_index('date')['2020'],x=time_df.set_index('date')['2020'].index,y='dayflow',hue='HDONG_NM',ax=ax[1],legend=False)
ax[0].legend(prop=font,bbox_to_anchor=(1.1,1),ncol=2,fontsize=17)
plt.tight_layout()
ax[0].set_title('2019년 동별 유동인구',fontproperties=font,size=30)
ax[1].set_title('2020년 동별 유동인구',fontproperties=font,size=30)
ax[1].set_ylim([0,18750])
plt.show()
# 시간단위 동별 유동인구 
temp=time_df.set_index('date').loc[:,'HDONG_NM':].reset_index().melt(id_vars=['HDONG_NM','SI_NM','GU_NM','date','dayflow'])
temp['time']=pd.DatetimeIndex(temp['date'])+temp['variable'].str[5:].map(lambda x: timedelta(hours=int(x)))
fig,ax=plt.subplots(2,1,figsize=(32,16))
temp.set_index('time')[['HDONG_NM','value']].reset_index().pivot(index='time',columns='HDONG_NM',values='value')['2019'].plot(ax=ax[0],title='2019 Flow')
temp.set_index('time')[['HDONG_NM','value']].reset_index().pivot(index='time',columns='HDONG_NM',values='value')['2020'].plot(ax=ax[1],title='2020 Flow',legend=False)
ax[0].legend(prop=font,bbox_to_anchor=(1.1,1),ncol=2,fontsize=17)
plt.show() # 알아보기가 힘들다 --> 시계열 분해 후 트렌드 도출
temp_table=temp.set_index('time')[['HDONG_NM','value']].reset_index().pivot(index='time',columns='HDONG_NM',values='value')
temp_dict={}
temp_season={}
temp_trend={}
for col in temp_table.columns:
    temp_dict[col]=sm.tsa.seasonal_decompose(temp_table['2019'][col])
for col in temp_table.columns:
    temp_season[col]=temp_dict[col].seasonal
    temp_trend[col]=temp_dict[col].trend
season_df = pd.DataFrame(temp_season)
trend_df = pd.DataFrame(temp_trend)    
season_df.index.name = None
trend_df.index.name = None
fig,ax=plt.subplots(2,2,figsize=(128,80))
season_df.plot(ax=ax[0][0],title='2019 population flow seasonality')
trend_df.plot(ax=ax[0][1],title='2019 population flow trend')
temp_dict={}
temp_season={}
temp_trend={}
for col in temp_table.columns:
    temp_dict[col]=sm.tsa.seasonal_decompose(temp_table['2020'][col])
for col in temp_table.columns:
    temp_season[col]=temp_dict[col].seasonal
    temp_trend[col]=temp_dict[col].trend
season_df = pd.DataFrame(temp_season)
trend_df = pd.DataFrame(temp_trend)    
season_df.index.name = None
trend_df.index.name = None
season_df.plot(ax=ax[1][0],title='2020 population flow seasonality')
trend_df.plot(ax=ax[1][1],title='2020 population flow trend')
ax[0][0].title.set_size(50)
ax[0][1].title.set_size(50)
ax[1][0].title.set_size(50)
ax[1][1].title.set_size(50)
plt.show()
g=sns.clustermap(trend_df.corr(),figsize=(64,64))
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),fontproperties=font,fontsize=25,rotation=90)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(),fontproperties=font,fontsize=25)
plt.show()
dong_dayflow_2019=pd.DataFrame(time_df.set_index('date')['2019'].groupby('HDONG_NM').sum()['dayflow'].sort_values(ascending=False))
dong_dayflow_2020=pd.DataFrame(time_df.set_index('date')['2020'].groupby('HDONG_NM').sum()['dayflow'].sort_values(ascending=False))
fig,ax=plt.subplots(1,2,figsize=(32,8))
sns.barplot(data=dong_dayflow_2019,x=dong_dayflow_2019.index,y='dayflow',ax=ax[0])
ax[0].set_xticklabels(dong_dayflow_2019.index,fontproperties=font,rotation=90,size=12)
sns.barplot(data=dong_dayflow_2020,x=dong_dayflow_2020.index,y='dayflow',ax=ax[1])
ax[1].set_xticklabels(dong_dayflow_2020.index,fontproperties=font,rotation=90,size=12)
ax[0].set_title('2019 동별 일 평균 유동인구',fontproperties=font,size=20)
ax[1].set_title('2020 동별 일 평균 유동인구',fontproperties=font,size=20)
ax[0].ticklabel_format(style='plain', axis='y')
ax[1].ticklabel_format(style='plain',axis='y')
ax[0].set_ylim([0,1700000])
ax[1].set_ylim([0,1700000])
plt.show()
dong_merged_dayflow=pd.merge(dong_dayflow_2019.reset_index(),dong_dayflow_2020.reset_index(),on='HDONG_NM',suffixes=['_2019','_2020'])
dong_merged_dayflow=dong_merged_dayflow.set_index('HDONG_NM')
dong_merged_dayflow['change']=dong_merged_dayflow.dayflow_2019-dong_merged_dayflow.dayflow_2020
fig,ax=plt.subplots(figsize=(32,8))
sns.barplot(data=dong_merged_dayflow,x=dong_merged_dayflow.index,y='change',order=dong_merged_dayflow.sort_values('change',ascending=False).index)
ax.ticklabel_format(style='plain',axis='y')
ax.set_xticklabels(dong_merged_dayflow.sort_values('change',ascending=False).index,fontproperties=font,rotation=90,size=12)
ax.set_title('동별 일 평균 유동인구 감소량 (2019 - 2020)',fontproperties=font,size=20)
plt.show()
dong_merged_dayflow['change_rate']=(dong_merged_dayflow.dayflow_2020-dong_merged_dayflow.dayflow_2019)/dong_merged_dayflow.dayflow_2019*100
fig,ax=plt.subplots(figsize=(32,8))
sns.barplot(data=dong_merged_dayflow,x=dong_merged_dayflow.index,y='change_rate',order=dong_merged_dayflow.sort_values('change_rate',ascending=True).index)
ax.ticklabel_format(style='plain',axis='y')
ax.set_xticklabels(dong_merged_dayflow.sort_values('change_rate',ascending=True).index,fontproperties=font,rotation=90,size=12)
ax.set_title('동별 일 평균 유동인구 증감률 (2019 - 2020) %',fontproperties=font,size=20)
ax.set_ylabel('change_rate (%)')
plt.show()
dong_meta
seoul_dayflow_2019=time_df[(time_df['date']<'2020') & (time_df['HDONG_NM'].isin(seoul_dong_name))].groupby('HDONG_NM').sum().reset_index()[['HDONG_NM','dayflow']]
seoul_dayflow_2020=time_df[(time_df['date']>='2020') & (time_df['HDONG_NM'].isin(seoul_dong_name))].groupby('HDONG_NM').sum().reset_index()[['HDONG_NM','dayflow']]
daegu_dayflow_2019=time_df[(time_df['date']<'2020') & (time_df
                                                      ['HDONG_NM'].isin(daegu_dong_name))].groupby('HDONG_NM').sum().reset_index()[['HDONG_NM','dayflow']]
daegu_dayflow_2020=time_df[(time_df['date']>='2020') & (time_df['HDONG_NM'].isin(daegu_dong_name))].groupby('HDONG_NM').sum().reset_index()[['HDONG_NM','dayflow']]
seoul_dayflow_2019=pd.merge(pd.merge(dong_pop.reset_index()[['행정구역','2019. 05']],seoul_dayflow_2019,left_on='행정구역',right_on='HDONG_NM').drop('행정구역',axis=1),dong_meta[['HDONG_NM','HDONG_CD']])
seoul_dayflow_2020=pd.merge(pd.merge(dong_pop.reset_index()[['행정구역','2020. 05']],seoul_dayflow_2020,left_on='행정구역',right_on='HDONG_NM').drop('행정구역',axis=1),dong_meta[['HDONG_NM','HDONG_CD']])
daegu_dayflow_2019=pd.merge(pd.merge(dong_pop.reset_index()[['행정구역','2019. 05']],daegu_dayflow_2019,left_on='행정구역',right_on='HDONG_NM').drop('행정구역',axis=1),dong_meta[['HDONG_NM','HDONG_CD']])
daegu_dayflow_2020=pd.merge(pd.merge(dong_pop.reset_index()[['행정구역','2020. 05']],daegu_dayflow_2020,left_on='행정구역',right_on='HDONG_NM').drop('행정구역',axis=1),dong_meta[['HDONG_NM','HDONG_CD']])
seoul_dayflow_2019 = seoul_dayflow_2019.sort_values('HDONG_NM').reset_index(drop=True).rename(columns={'2019. 05':'인구'})
seoul_dayflow_2020 = seoul_dayflow_2020.sort_values('HDONG_NM').reset_index(drop=True).rename(columns={'2020. 05':'인구'})
daegu_dayflow_2019 = daegu_dayflow_2019.sort_values('HDONG_NM').reset_index(drop=True).rename(columns={'2019. 05':'인구'})
daegu_dayflow_2020 = daegu_dayflow_2020.sort_values('HDONG_NM').reset_index(drop=True).rename(columns={'2020. 05':'인구'})
seoul_dayflow_2019['HDONG_CD']=seoul_dayflow_2019.HDONG_CD.astype(str)
seoul_dayflow_2020['HDONG_CD']=seoul_dayflow_2020.HDONG_CD.astype(str)
daegu_dayflow_2019['HDONG_CD']=daegu_dayflow_2019.HDONG_CD.astype(str)
daegu_dayflow_2020['HDONG_CD']=daegu_dayflow_2020.HDONG_CD.astype(str)
seoul_dayflow_2019['dayflow']=seoul_dayflow_2019.dayflow.astype(int)
seoul_dayflow_2020['dayflow']=seoul_dayflow_2020.dayflow.astype(int)
daegu_dayflow_2019['dayflow']=daegu_dayflow_2019.dayflow.astype(int)
daegu_dayflow_2020['dayflow']=daegu_dayflow_2020.dayflow.astype(int)
# with open('../input/korgeo/HangJeongDong_ver20200701.geojson',mode='rt',encoding='utf-8') as f:
#     g = json.loads(f.read())
#     f.close()
# m = folium.plugins.DualMap(location=(37.615,127.046), zoom_start=11.8)
# temp=pd.concat([seoul_dayflow_2019,seoul_dayflow_2020])['dayflow']
# interval=(temp.max()-temp.min())/6
# bins=list(map(float,[temp.min(),temp.min()+interval,temp.min()+interval*2,temp.min()+interval*3,temp.min()+interval*4,temp.min()+interval*5,temp.max()]))
# m.m1.choropleth( geo_data=g, data=seoul_dayflow_2019, columns=('HDONG_CD', 'dayflow'), 
#                   key_on='feature.properties.adm_cd2', fill_color = 'YlOrRd',
#                 fill_opacity=0.8,
#     line_opacity=0.5,
#                 bins=bins)
# m.m2.choropleth( geo_data=g, data=seoul_dayflow_2020, columns=('HDONG_CD', 'dayflow'), 
#                   key_on='feature.properties.adm_cd2', fill_color = 'YlOrRd',fill_opacity=0.8,
#     line_opacity=0.5,
#                 bins=bins)

# m.save('./seoul_year_pop_map.html')
# IFrame(src='./seoul_year_pop_map.html', width=1000, height=600)


"""notebook 느려져서 주석처리 / 한국행정동 파일 (g파일) 읽는데 시간, 메모리 소모 큼 """
# m = folium.plugins.DualMap(location=(35.843, 128.626), zoom_start=11.8)
# temp=pd.concat([daegu_dayflow_2019,daegu_dayflow_2020])['dayflow']
# interval=(temp.max()-temp.min())/6
# bins=list(map(float,[temp.min(),temp.min()+interval,temp.min()+interval*2,temp.min()+interval*3,temp.min()+interval*4,temp.min()+interval*5,temp.max()]))
# m.m1.choropleth( geo_data=g, data=daegu_dayflow_2019, columns=('HDONG_CD', 'dayflow'), 
#                   key_on='feature.properties.adm_cd2', fill_color = 'YlOrRd',
#                 fill_opacity=0.8,
#     line_opacity=0.5,
#                 bins=bins)
# m.m2.choropleth( geo_data=g, data=daegu_dayflow_2020, columns=('HDONG_CD', 'dayflow'), 
#                   key_on='feature.properties.adm_cd2', fill_color = 'YlOrRd',fill_opacity=0.8,
#     line_opacity=0.5,
#                 bins=bins)

# m.save('./daegu_year_pop_map.html')
# IFrame(src='./daegu_year_pop_map.html', width=1000, height=600)
"""notebook 느려져서 주석처리"""
seoul_dayflow_2019_geo=pd.merge(seoul_dayflow_2019,geo_meta[['HDONG_NM','X_COORD','Y_COORD']])[['Y_COORD','X_COORD','dayflow']]
seoul_dayflow_2020_geo=pd.merge(seoul_dayflow_2020,geo_meta[['HDONG_NM','X_COORD','Y_COORD']])[['Y_COORD','X_COORD','dayflow']]
daegu_dayflow_2019_geo=pd.merge(daegu_dayflow_2019,geo_meta[['HDONG_NM','X_COORD','Y_COORD']])[['Y_COORD','X_COORD','dayflow']]
daegu_dayflow_2020_geo=pd.merge(daegu_dayflow_2020,geo_meta[['HDONG_NM','X_COORD','Y_COORD']])[['Y_COORD','X_COORD','dayflow']]
# min_flow=min(seoul_dayflow_2019.dayflow.min(),seoul_dayflow_2020.dayflow.min())
# max_flow=max(seoul_dayflow_2019.dayflow.max(),seoul_dayflow_2020.dayflow.max())

# seoul_dayflow_2019_geo['dayflow']=(seoul_dayflow_2019_geo['dayflow'].astype(float)-min_flow)/(max_flow-min_flow)*4000
# seoul_dayflow_2020_geo['dayflow']=(seoul_dayflow_2020_geo['dayflow'].astype(float)-min_flow)/(max_flow-min_flow)*4000
# m = DualMap(location=[37.615, 127.046],
#                     zoom_start = 11.8) 
# colormap = {0.0: 'pink', 0.3: 'blue', 0.5: 'green',  0.7: 'yellow', 1: 'red'}
# HeatMap(seoul_dayflow_2019_geo.values.tolist(),radius=30, blur = 1,min_opacity=0).add_to(m.m1)
# HeatMap(seoul_dayflow_2020_geo.values.tolist(),radius=30, blur = 1,min_opcaity=0,gradient=colormap).add_to(m.m2)
# m.save('seoul_year_pop_heatmap.html')
# m
"""HEATMAP - 부적합"""
# m = DualMap(location=[35.843, 128.626],
#                     zoom_start = 11.8) 
# HeatMap(daegu_dayflow_2019_geo,radius=30).add_to(m.m1)
# HeatMap(daegu_dayflow_2020_geo,radius=30).add_to(m.m2)
# m
"""HEATMAP - 부적합"""
daytime_geo=pd.merge(geo_meta[['HDONG_NM','X_COORD','Y_COORD']],time_df[time_df.date<'2020'].groupby('HDONG_NM').mean().loc[:,'TMST_00':'TMST_23'].reset_index())
### scaling
for col in daytime_geo.loc[:,'TMST_00':'TMST_23'].columns.tolist():
    daytime_geo[col] = daytime_geo[col]/daytime_geo[col].sum()*50
melted_df=pd.melt(daytime_geo.drop('HDONG_NM',axis=1),id_vars=['X_COORD','Y_COORD'])
melted_df=melted_df[['Y_COORD','X_COORD','variable','value']]
melted_df.variable=pd.to_numeric(melted_df.variable.str[5:])
melted_df
hm_arr=[]
for i in range(24):
    hm_arr.append(melted_df.set_index('variable').loc[i,:].values.tolist())
m=folium.Map(location=[37.615, 127.046],
                    zoom_start = 11.8) 
hm = plugins.HeatMapWithTime(hm_arr,auto_play=True,radius=80)
hm.add_to(m)
m
seoul_dayflow=time_df.groupby(['SI_NM','date']).sum().loc['서울','dayflow']
daegu_dayflow=time_df.groupby(['SI_NM','date']).sum().loc['대구','dayflow']
seoul_dayflow=seoul_dayflow.reset_index()
daegu_dayflow=daegu_dayflow.reset_index()
se_dayflow=seoul_dayflow.drop(seoul_dayflow[seoul_dayflow.date=='2020-02-29'].index,axis=0)
da_dayflow=daegu_dayflow.drop(daegu_dayflow[daegu_dayflow.date=='2020-02-29'].index,axis=0)
fig,ax=plt.subplots(2,1,figsize=(20,8))
sns.kdeplot(data=seoul_dayflow.set_index('date')['2019']['dayflow'],ax=ax[0],label='2019',shade=True)
sns.kdeplot(data=seoul_dayflow.set_index('date')['2020']['dayflow'],ax=ax[0],label='2020',shade=True)
sns.kdeplot(data=daegu_dayflow.set_index('date')['2019']['dayflow'],ax=ax[1],label='2019',shade=True)
sns.kdeplot(data=daegu_dayflow.set_index('date')['2020']['dayflow'],ax=ax[1],label='2020',shade=True)
ax[0].ticklabel_format(style='plain',axis='y')
ax[1].ticklabel_format(style='plain',axis='y')
ax[0].set_title('Seoul dayflow kernel density plot')
ax[1].set_title('Daegu dayflow kernel density plot')
ax[0].set_xlim([20000,200000])
ax[1].set_xlim([20000,200000])
plt.legend(['2019','2020'])
plt.show()
# TIME SERIES ANALYSIS

fig,ax=plt.subplots(2,2,sharey='row')
seoul_dayflow.set_index('date')['2019'].plot(ax=ax[0][0],title='2019 Seoul pop flow')
seoul_dayflow.set_index('date')['2020'].plot(ax=ax[0][1],c='b',title='2020 Seoul pop flow')
seoul_dayflow.set_index('date')['2019'].pct_change().plot(ax=ax[1][0],title='2019 Seoul pop flow percent change')
seoul_dayflow.set_index('date')['2020'].pct_change().plot(ax=ax[1][1],c='b',title='2020 Seoul pop flow percent change')
plt.show()
fig,ax=plt.subplots(2,2,sharey='row')
daegu_dayflow.set_index('date')['2019'].plot(ax=ax[0][0],title='2019 Daegu pop flow')
daegu_dayflow.set_index('date')['2020'].plot(ax=ax[0][1],c='b',title='2020 Daegu pop flow')
daegu_dayflow.set_index('date')['2019'].pct_change().plot(ax=ax[1][0],title='2019 Daegu pop flow percent change')
daegu_dayflow.set_index('date')['2020'].pct_change().plot(ax=ax[1][1],c='b',title='2020 Daegu pop flow percent change')
plt.show()
print(adfuller(daegu_dayflow.set_index('date')['2019'])) 
print(adfuller(daegu_dayflow.set_index('date')['2019'].diff().dropna()))
print(adfuller(daegu_dayflow.set_index('date')['2020'])) # non stationary by trend
print(adfuller(daegu_dayflow.set_index('date')['2020'].diff().dropna())) # non stationary by trend
print(adfuller(seoul_dayflow.set_index('date')['2019'])) # non stationary by trend
print(adfuller(seoul_dayflow.set_index('date')['2019'].diff().dropna()))
print(adfuller(seoul_dayflow.set_index('date')['2020'])) # non stationary by trend
print(adfuller(seoul_dayflow.set_index('date')['2020'].diff().dropna()))
daegu_dayflow.set_index('date')['2020'].diff().diff().dropna().plot(figsize=(4,4))
plot_acf(np.log(daegu_dayflow.set_index('date')['2020'].diff().diff()).dropna())
plot_pacf(np.log(daegu_dayflow.set_index('date')['2020'].diff().diff()).dropna())
plt.show()
temp=daegu_dayflow.set_index('date')['2020'].diff().diff().dropna()
adfuller(temp)[1]
k=plot_acf(temp)
g=plot_pacf(temp)
# ARIMA (p=7,d=2,q=?)
resDiff = sm.tsa.arma_order_select_ic(temp, max_ar=7, max_ma=7, ic='aic', trend='c')
print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')
# ARMA parameter Select
# ARIMA MODEL
tra=temp[:'2020-05-25']
tes=temp['2020-05-25':]
result = sm.tsa.statespace.SARIMAX(tra,order=(7,2,7),freq='D',seasonal_order=(0,0,0,0),
                                 enforce_stationarity=False, enforce_invertibility=False,).fit()
result.summary()
res = result.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()
pred = result.predict('2020-05-24','2020-05-31')[1:]
print('ARIMA model MSE:{}'.format(mean_squared_error(tes,pred)))
pd.DataFrame({'test':tes.dayflow,'pred':pred}).plot()
plt.show()
report=result.plot_diagnostics(figsize=(15, 12))
# SARIMAX MODEL
sarima = sm.tsa.statespace.SARIMAX(tra,order=(7,1,7),seasonal_order=(0,1,0,52),
                                enforce_stationarity=False, enforce_invertibility=False,freq='D').fit()
sarima.summary()
res = sarima.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()
pred = sarima.predict('2020-05-24','2020-05-31')[1:]
print('SARIMA model MSE:{}'.format(mean_squared_error(tes,pred)))
pd.DataFrame({'test':tes['dayflow'],'pred':pred}).plot()
plt.show()
# ARIMA >> SARIMA -> 마지막셀에 이어서 
# ARMA MODEL
# 상관분석 flow: (S)ARIMAX 시계열 모형으로 2월 유동인구 데이터 예측 exog=코로나 확진자수 (동별? 구별?) -> R square 값을 통해 동별 정렬
temp=np.log(daegu_dayflow.set_index('date')['2020']-daegu_dayflow.set_index('date')['2020'].shift(7)).dropna()
model=sm.tsa.SARIMAX(temp,order=(0,1,0), seasonal_order=(1,1,1,7))
results = model.fit()
print (results.summary())
# 일 평균 (각 시간대 유동인구의 평균) 유동인구 heatmap  
sns.heatmap(time_df.set_index('date').groupby(time_df.set_index('date').index).sum().loc[:,'TMST_00':].corr(),annot=True)
# 일 평균 (각 시간대 유동인구의 평균) 유동인구 clustermap  
sns.clustermap(time_df.set_index('date').groupby(time_df.set_index('date').index).sum().loc[:,'TMST_00':].corr(),annot=True)
# 연도별 일 평균 유동인구 시계열 분해 
df=time_df.set_index('date')['2019'].groupby('date').sum()['dayflow']
fig,ax=plt.subplots(4,1,figsize=(32,16))
decom = sm.tsa.seasonal_decompose(df)
decom.observed.plot(ax=ax[0],title='2019 dayflow')
decom.seasonal.plot(ax=ax[1],title='2019 dayflow seasonality')
decom.trend.plot(ax=ax[2],title='2019 dayflow trend')
decom.resid.plot(ax=ax[3],title='2019 dayflow residual')
plt.show()
df=time_df.set_index('date')['2020'].groupby('date').sum()['dayflow']
fig,ax=plt.subplots(4,1,figsize=(32,16))
decom = sm.tsa.seasonal_decompose(df)
decom.observed.plot(ax=ax[0],title='2020 dayflow',c='red')
decom.seasonal.plot(ax=ax[1],title='2020 dayflow seasonality',c='red')
decom.trend.plot(ax=ax[2],title='2020 dayflow trend',c='red')
decom.resid.plot(ax=ax[3],title='2020 dayflow residual',c='red')
plt.show()
def compare_dayflow(df1,df2,title_1,title_2):
    
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(2,1,figsize=(32,32))
    sns.lineplot(x=df1[df1.date<'2020']['date'].apply(lambda x : x.strftime('%m-%d')),
                 y=df1[df1.date<'2020']['dayflow'],ax=ax[0],label='2019')
    sns.lineplot(x=df1[df1.date>='2020']['date'].apply(lambda x : x.strftime('%m-%d')),
                 y=df1[df1.date>='2020']['dayflow'],ax=ax[0],label='2020')

    sns.lineplot(x=df2[df2.date<'2020']['date'].apply(lambda x : x.strftime('%m-%d')),
                 y=df2[df2.date<'2020']['dayflow'],ax=ax[1],label='2019')
    sns.lineplot(x=df2[df2.date>='2020']['date'].apply(lambda x : x.strftime('%m-%d')),
                 y=df2[df2.date>='2020']['dayflow'],ax=ax[1],label='2020')
    plt.setp(ax[0].get_xticklabels(), rotation=70, ha='right',size=14)
    plt.setp(ax[1].get_xticklabels(), rotation=70, ha='right',size=14)
    ax[0].set_title(title_1,fontproperties=font,size=40)
    ax[1].set_title(title_2,fontproperties=font,size=40)
    plt.legend()
    plt.show()
def compare_weekflow(df1,df2,title_1,title_2):
    
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(2,1,figsize=(32,32))
    sns.lineplot(x=df1[df1.week<'2020']['week'].apply(lambda x : x.strftime('%m-%d')),
                 y=df1[df1.week<'2020']['dayflow'],ax=ax[0],label='2019')
    sns.lineplot(x=df1[df1.week>='2020']['week'].apply(lambda x : x.strftime('%m-%d')),
                 y=df1[df1.week>='2020']['dayflow'],ax=ax[0],label='2020')

    sns.lineplot(x=df2[df2.week<'2020']['week'].apply(lambda x : x.strftime('%m-%d')),
                 y=df2[df2.week<'2020']['dayflow'],ax=ax[1],label='2019')
    sns.lineplot(x=df2[df2.week>='2020']['week'].apply(lambda x : x.strftime('%m-%d')),
                 y=df2[df2.week>='2020']['dayflow'],ax=ax[1],label='2020')
    plt.setp(ax[0].get_xticklabels(), rotation=70, ha='right',size=14)
    plt.setp(ax[1].get_xticklabels(), rotation=70, ha='right',size=14)
    ax[0].set_title(title_1,fontproperties=font,size=40)
    ax[1].set_title(title_2,fontproperties=font,size=40)
    plt.legend()
    plt.show()
compare_dayflow(se_dayflow,da_dayflow,'서울 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화',
               '대구 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화')
suseong_dayflow=time_df.groupby(['GU_NM','date']).sum().loc['수성구','dayflow']
d_jung_dayflow=time_df.groupby(['GU_NM','date']).sum().loc['대구_중구','dayflow']
noone_dayflow=time_df.groupby(['GU_NM','date']).sum().loc['노원구','dayflow']
jung_dayflow=time_df.groupby(['GU_NM','date']).sum().loc['중구','dayflow']

suseong_dayflow=suseong_dayflow.reset_index()
d_jung_dayflow=d_jung_dayflow.reset_index()
noone_dayflow=noone_dayflow.reset_index()
jung_dayflow=jung_dayflow.reset_index()

su_dayflow=suseong_dayflow.drop(suseong_dayflow[suseong_dayflow.date=='2020-02-29'].index,axis=0,)
djung_dayflow=d_jung_dayflow.drop(d_jung_dayflow[d_jung_dayflow.date=='2020-02-29'].index,axis=0,)
no_dayflow=noone_dayflow.drop(noone_dayflow[noone_dayflow.date=='2020-02-29'].index,axis=0,)
sjung_dayflow=jung_dayflow.drop(jung_dayflow[jung_dayflow.date=='2020-02-29'].index,axis=0,)
compare_dayflow(su_dayflow,djung_dayflow,'수성구 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화',
               '대구_중구 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화')
compare_dayflow(no_dayflow,sjung_dayflow,'노원구 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화',
               '중구 2019.02 ~ 2019.05 / 2020.02 ~ 2020.05 일별 유동인구 변화')
t_2019=time_df.set_index('date')['2019'].groupby('SI_NM').sum().loc[:,'TMST_00':'TMST_23'].T
t_2020=time_df.set_index('date')['2020'].groupby('SI_NM').sum().loc[:,'TMST_00':'TMST_23'].T
fig,ax=plt.subplots(1,2,figsize=(32,8))
t_2019['대구'].plot(kind='line',ax=ax[0],title='Daegu',label='2019')
t_2020['대구'].plot(kind='line',ax=ax[0],label='2020')
t_2019['서울'].plot(kind='line',ax=ax[1],title='Seoul',label='2019')
t_2020['서울'].plot(kind='line',ax=ax[1],label='2020')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
fig,ax=plt.subplots(figsize=(32,8))
dt=(t_2020-t_2019)/t_2019*100
sns.lineplot(data=dt,dashes=False,ax=ax)
ax.set_xticklabels(dt.index.str[5:],rotation=70,size=17)
ax.ticklabel_format(style='plain',axis='y')
ax.legend(['Daegu','Seoul'])
ax.set_title('2019 2020 서울 대구 시간대 별 평균 유동인구 증감률 (%)',fontproperties=font,size=30)
ax.set_ylim([-50,20])
plt.show()
import itertools
fig,axes=plt.subplots(4,1,figsize=(32,64))
palette = sns.color_palette(palette=sns.crayon_palette(sns.colors.crayons))
new_palette = itertools.cycle(palette)
for i in ['02','03','04','05']:
    ax=axes[int(i[1])-2]
    t_2019=time_df.set_index('date')['2019-'+i].groupby('SI_NM').sum().loc[:,'TMST_00':'TMST_23'].T
    t_2020=time_df.set_index('date')['2020-'+i].groupby('SI_NM').sum().loc[:,'TMST_00':'TMST_23'].T
    y=2019
    dt=(t_2020-t_2019)/t_2019*100
    sns.lineplot(data=dt,dashes=False,ax=ax)
    ax.set_xticklabels(dt.index.str[5:],rotation=70,size=17)
    ax.ticklabel_format(style='plain',axis='y')
    ax.legend(['Daegu','Seoul'])
    ax.set_title(i+'월 2019 2020 서울 대구 시간대 별 평균 유동인구 증감률 (%)',fontproperties=font,size=30)
    ax.set_ylim([-100,100])
    ax.axhline(0, ls='--',c='k')
# plt.tight_layout()
plt.show()

# 위 그래프 요일 요인 제거 필요 (28일만 집계)?
# 오전 3시에 유동인구 증가 / 2월 대구에서 75프로 증가 / 이유는?
fig,ax=plt.subplots(2,1,figsize=(24,16))
temp_2019_02=time_df[time_df['SI_NM']=='대구'].set_index('date')['2019-02'].groupby('date').sum()
temp_2020_02=time_df[time_df['SI_NM']=='대구'].set_index('date')['2020-02'].groupby('date').sum()
sns.lineplot(data=temp_2019_02,x=temp_2019_02.index.day,y='TMST_03',ax=ax[0])
sns.lineplot(data=temp_2020_02,x=temp_2020_02.index.day,y='TMST_03',ax=ax[0])
sns.lineplot(data=temp_2019_02,x=temp_2019_02.index.week,y='TMST_03',ax=ax[1])
sns.lineplot(data=temp_2020_02,x=temp_2020_02.index.week,y='TMST_03',ax=ax[1])
ax[0].set_xlabel('(일)',fontproperties=font)
ax[1].set_xlabel('(주차)',fontproperties=font)
ax[0].set_xlabel('유동인구',fontproperties=font)
ax[1].set_xlabel('유동인구',fontproperties=font)
ax[0].set_title('대구 2월 일 평균 새벽 3시 유동인구 총합',fontproperties=font,size=25)
ax[1].set_title('대구 2월 주 평균 새벽 3시 유동인구 총합',fontproperties=font,size=25)
plt.show()
# 2019 2020 2월 3월
# 새벽 12 ~3 유동인구 소폭 증가
# 일간 유동인구로 비교가 힘들다 
fig,ax=plt.subplots(2,1)
time_df.groupby('date').sum()['2019']['dayflow'].plot(figsize=(32,4),ax=ax[0],c='r')
time_df.groupby('date').sum()['2020']['dayflow'].plot(figsize=(32,4),ax=ax[1],c='b')
ax[0].set_title('2019/2020 compare population flow')
plt.show()
adfuller(time_df.groupby('date').sum()['2019']['dayflow'].diff().dropna())[1]
# 2019 2020 일간 유동인구 합 변화량 ACF

fig,ax=plt.subplots(2,2,figsize=(32,16))
plot_acf(time_df.groupby('date').sum()['2019']['dayflow'].diff().dropna(),ax=ax[0,0],alpha=0.05,title='2019 DAYFLOW change ACF')
plot_acf(time_df.groupby('date').sum()['2020']['dayflow'].diff().dropna(),ax=ax[0,1],c='b',alpha=0.05,title='2020 DAYFLOW change ACF')
plot_pacf(time_df.groupby('date').sum()['2019']['dayflow'].diff().dropna(),ax=ax[1,0],alpha=0.05,title='2019 DAYFLOW change PACF')
plot_pacf(time_df.groupby('date').sum()['2020']['dayflow'].diff().dropna(),ax=ax[1,1],c='b',title='2020 DAYFLOW change PACF')

plt.show()        
week_flow_2019=pd.DataFrame(time_df.groupby('date').sum()['2019'].resample('W-Fri')['dayflow'].sum()[1:]) # 토 ~ 금 집계 (제외 데이터 (2월 1일))
week_flow_2020=pd.DataFrame(time_df.groupby('date').sum()['2020'].resample('W-Sun')['dayflow'].sum()[1:]) # 월 ~ 일 집계 (제외 데이터 (2월 1일,2일))
week_flow_2019.index = week_flow_2019.index.week # 2019 6주 ~ 22주
week_flow_2020.index = week_flow_2020.index.week # 2020 6주 ~ 22주  

fig,ax=plt.subplots(figsize=(32,16))
sns.lineplot(data=week_flow_2019,x=week_flow_2019.index,y='dayflow',ax=ax,label='2019')
sns.lineplot(data=week_flow_2020,x=week_flow_2020.index,y='dayflow',ax=ax,label='2020')
ax.set_xlabel('Week')
ax.set_ylabel('4개 지역 주별 유동인구 합',fontproperties=font)
plt.title('2019 02~05 / 2020 02~05 주간 4개 지역 유동인구 합',fontproperties=font,size=30)
plt.legend()
plt.show()
sns.boxplot(data=time_df.replace({'서울':'Seoul','대구':'Daegu'}),x='STD_YM',y='dayflow',hue='SI_NM')
plt.show()
suseong_df=pd.read_csv('../input/suseonggu-corona/__.csv',sep=',',header=None)
suseong_df.columns=['date','누적 확진자 수']
suseong_df.head()
suseong_df=suseong_df.set_index('date')
suseong_df.index=suseong_df.index.astype(str)
suseong_ind = []
for i,ind in enumerate(suseong_df.index):
    ind=ind.replace('.','-')
    if len(ind) ==3:
        suseong_ind.append('2020-0'+ind+'0')
    else:
        suseong_ind.append('2020-0'+ind)
    
suseong_df.index= suseong_ind
suseong_df.index=pd.to_datetime(suseong_df.index,format='%Y-%m-%d')
suseong_df['누적 확진자 수']=suseong_df['누적 확진자 수'].str.replace(',','').astype(int)
suseong_df['당일 확진자 수']=suseong_df.diff()
suseong_df=suseong_df.fillna(5)
suseong_df['당일 확진자 수']=suseong_df['당일 확진자 수'].astype(int)
for i in pd.date_range("2/1/2020", "2/19/2020"):
        suseong_df.loc[i]=[0,0]
suseong_df=suseong_df.sort_index()
fig,ax = plt.subplots(1,2,figsize=(32,8))
sns.lineplot(data=suseong_df,x=suseong_df.index,y='당일 확진자 수',ax=ax[0])
sns.lineplot(data=suseong_df,x=suseong_df.index,y='누적 확진자 수',ax=ax[1])

sns.set_style('ticks')
suseong_flow_df=suseong_dayflow.set_index('date')
fig,ax=plt.subplots(figsize=(32,16))
sns.lineplot(data=suseong_df,x=suseong_df.index,y='당일 확진자 수',ax=ax,label='확진자 수',c='r')
ax2 = ax.twinx()
sns.lineplot(data=suseong_flow_df['2020'],x=suseong_df.index,y='dayflow',ax=ax2,c='b',label='일별 유동인구 합계')
ax.set_ylabel('확진자 수',fontproperties=font,size=40,c='r')
ax2.set_ylabel('일별 유동인구 합계',fontproperties=font,size=40,c='b')
ax.set_title('수성구 2020 02 ~ 05 확진자 수, 유동인구 그래프 ',fontproperties=font)
plt.show()
adfuller(suseong_df['2020']['당일 확진자 수'])[1]


# Regress BTC on ETH
s_add_const = sm.add_constant(suseong_df['당일 확진자 수'])
result = sm.OLS(suseong_flow_df['2020'],s_add_const).fit()

# Compute ADF
b = result.params[1]
adf_stats = adfuller(suseong_flow_df['2020']['dayflow'] - b*s_add_const['당일 확진자 수'])
print("The p-value for the ADF test is ", adf_stats[1])
print('2019 p-value:',adfuller(suseong_flow_df['2019'])[1])
print('2020 p-value:',adfuller(suseong_flow_df['2020-04'].diff().dropna())[1])
suseong_flow_df['2020-04'].plot(figsize=(15,5))
suseong_flow_df['2020-04'].diff().dropna().plot(figsize=(15,5))
plt.show()
"""여기까지"""
suseong_time_flow=time_df[time_df['GU_NM']=='수성구'].groupby('date').sum()
suseong_time_flow.head()

suseong_week_flow_2019=pd.DataFrame(suseong_time_flow['2019'].resample('W-Fri')['dayflow'].sum()[1:]) # 토 ~ 금 집계 (제외 데이터 (2월 1일))
suseong_week_flow_2020=pd.DataFrame(suseong_time_flow['2020'].resample('W-Sun')['dayflow'].sum()[1:]) # 월 ~ 일 집계 (제외 데이터 (2월 1일,2일))
suseong_week_flow_2019.index = suseong_week_flow_2019.index.week # 2019 6주 ~ 22주
suseong_week_flow_2020.index = suseong_week_flow_2020.index.week # 2020 6주 ~ 22주  
fig,ax=plt.subplots(figsize=(32,16))
sns.lineplot(data=suseong_week_flow_2019,x=suseong_week_flow_2019.index,y='dayflow',ax=ax,label='2019')
sns.lineplot(data=suseong_week_flow_2020,x=suseong_week_flow_2020.index,y='dayflow',ax=ax,label='2020')
ax.set_xlabel('Week')
ax.set_ylabel('수성구 주별 유동인구 합',fontproperties=font)
plt.title('2019 02~05 / 2020 02~05 주간 수성구 유동인구 합',fontproperties=font,size=30)
plt.legend()
plt.show()
fig,ax=plt.subplots()
# sns.pointplot(data=suseong_week_flow_2019-suseong_week_flow_2020,x=suseong_week_flow_2019-suseong_week_flow_2020.index,y=(suseong_week_flow_2019-suseong_week_flow_2020).dayflow,
#               ax=ax,label='전년대비 주간 유동인구 감소량')
ax.plot(suseong_week_flow_2019-suseong_week_flow_2020,marker='o')
ax2=ax.twinx()
# sns.pointplot(data=suseong_df.resample('W-sun').sum(),x=suseong_df.resample('W-sun').sum().index.week,y='당일 확진자 수',ax=ax2,label='주간 확진자 수',c='b')
ax2.plot(suseong_df.resample('W-sun').sum().index.week,suseong_df.resample('W-sun').sum()['당일 확진자 수'],c='b',marker='v')
ax.set_ylabel('전년대비 주간 유동인구 감소량',fontproperties=font,size=30,c='r')
ax2.set_ylabel('주간 확진자 수',fontproperties=font,size=30,c='b')
plt.show()
#주차별 수성구 유동인구 감소량 (2019 - 2020)

print('2020 전년 대비 유동인구 급감 주 마지막 날짜', suseong_time_flow['2020'].resample('W-Fri').sum().index[(suseong_time_flow['2020'].resample('W-Fri').sum()).index.week.tolist().index(18)])
print('2019 전년 대비 유동인구 급감 주 마지막 날짜', suseong_time_flow['2019'].resample('W-Fri').sum().index[(suseong_time_flow['2019'].resample('W-Fri').sum()).index.week.tolist().index(18)])
# 18번째 주에서  급격한 유동인구 감소 (4월 27일 ~ 5월 3일 / 5월5일 어린이날, 5월 6일 대체휴일) 
# 20년 대구 컬러풀 페스티벌 취소
# time_df[time_df['GU_NM']=='수성구'].groupby(['HDONG_NM','date']).sum().loc['고산1동','dayflow']
suseong_dong_list=time_df[time_df['GU_NM']=='수성구'].HDONG_NM.unique().tolist()
for dong_nm in suseong_dong_list:
    gu_temp=time_df[time_df['GU_NM']=='수성구'].groupby(['HDONG_NM','date']).sum().loc[dong_nm,'dayflow']
    gu_temp[(gu_temp.index>='2019-04-15')&(gu_temp.index<='2019-05-09')].plot()
plt.legend(labels=suseong_dong_list)
plt.show()
# time_df[time_df['GU_NM']=='수성구'].groupby(['HDONG_NM','date']).sum().loc['고산1동','dayflow']
suseong_dong_list=time_df[time_df['GU_NM']=='대구_중구'].HDONG_NM.unique().tolist()
for dong_nm in suseong_dong_list:
    gu_temp=time_df[time_df['GU_NM']=='대구_중구'].groupby(['HDONG_NM','date']).sum().loc[dong_nm,'dayflow']
    gu_temp[(gu_temp.index>='2019-04-15')&(gu_temp.index<='2019-05-09')].plot()
plt.legend(labels=suseong_dong_list)
plt.show()
#대구 중구, 수성구 다수의 동에서 4월 27일 다른 주의 토요일보다 유동인구가 많았다 
"""지난 4월 27일(토), 불기2563년 부처님오신 날(4월 초팔일, 양력 5월 12일)을 맞아 대구 두류공원에
서는 약 천명의 외국인을 비롯하여 권영진대구시장, 
효광동화사주지스님과 불교계인사 및 시민 등등 5만여 명이 운집한 거운데 형형색색 달구벌 관등놀이가 성대하게 거행되었다.
연호동 위치 삼성라이온즈 파크 -> 4월 27일 홈 경기
5월 1일 근로자의 날 
"""
home_training_df = pd.read_csv('../input/hometraining/hometrain.csv')
home_training_df
home_training_df.rename(columns=home_training_df.iloc[0])
home_training_df = home_training_df.iloc[1:,:]
home_training_df.head()
home_training_df.reset_index(inplace=True)
home_training_df.columns = ['date','ind']
home_training_df['date']=pd.to_datetime(home_training_df['date'],format='%Y-%m-%d')
home_training_df['ind']=home_training_df['ind'].apply(pd.to_numeric)
fig,ax=plt.subplots()
sns.lineplot(x=home_training_df[home_training_df.date<='2019-09-15']['date'].apply(lambda x : x.strftime('%m-%d')),
            y=home_training_df[home_training_df.date<='2019-09-15']['ind'],ax=ax)
ax2 =ax.twinx()
ax.set_ylim([0, 100])
ax2.set_ylim([0, 100])
ax3=ax.twinx()
# ax3.set_ylim([-100,100])
sns.lineplot(x=home_training_df[(home_training_df.date>='2020-01-01')&(home_training_df.date<='2020-09-15')]['date'].apply(lambda x : x.strftime('%m-%d')),
            y=home_training_df[(home_training_df.date>='2020-01-01')&((home_training_df.date<='2020-09-15'))]['ind'],ax=ax2,c='b')
ax.xaxis.set_tick_params(rotation=45)
# sns.lineplot(x=home_training_df[(home_training_df.date>='2020-01-01')&(home_training_df.date<='2020-09-15')]['date'].apply(lambda x : x.strftime('%m-%d')),
#              y=home_training_df[(home_training_df.date>='2020-01-01')&((home_training_df.date<='2020-09-15'))]['ind'].values-home_training_df[home_training_df.date<'2019-09-15']['ind'].values,
#              ax=ax3,c='c')
                                                                                                                           
plt.title('구글 검색어 트렌드 : 홈트레이닝',fontproperties=font,size=30)
plt.show()
age_df['date']=pd.to_datetime(age_df.STD_YMD,format='%Y%m%d')
age_df=age_df.set_index('date')
m_age_2019=age_df.loc['2019'].sum()['MAN_FLOW_POP_CNT_0004':'MAN_FLOW_POP_CNT_70U']
m_age_2020=age_df.loc['2020'].sum()['MAN_FLOW_POP_CNT_0004':'MAN_FLOW_POP_CNT_70U']
f_age_2019=age_df.loc['2019'].sum()['WMAN_FLOW_POP_CNT_0004':'WMAN_FLOW_POP_CNT_70U']
f_age_2020=age_df.loc['2020'].sum()['WMAN_FLOW_POP_CNT_0004':'WMAN_FLOW_POP_CNT_70U']
fig,ax=plt.subplots(1,2,figsize=(16,8))
temp=pd.DataFrame((m_age_2020-m_age_2019)/m_age_2019*100).sort_values(0)
sns.barplot(data=temp,x=temp.index,y=temp[0],ax=ax[0])
ax[0].set_xticklabels(temp.index,rotation=90)
temp=pd.DataFrame((f_age_2020-f_age_2019)/f_age_2019*100).sort_values(0)
sns.barplot(data=temp,x=temp.index,y=temp[0],ax=ax[1])
ax[1].set_xticklabels(temp.index,rotation=90)
ax[0].set_ylim([-100,0])
ax[1].set_ylim([-100,0])
ax[0].set_ylabel('%')
ax[1].set_ylabel('%')
plt.show()
exog=suseong_df['당일 확진자 수']
exog_tra=suseong_df['2020-02-03':'2020-05-25']['당일 확진자 수']
exog_tes=suseong_df['2020-05-25':]['당일 확진자 수']
arimax = sm.tsa.statespace.SARIMAX(tra,order=(7,1,7),seasonal_order=(0,0,0,0),exog = exog_tra,freq='D',
                                  enforce_stationarity=False, enforce_invertibility=False,).fit()
arimax.summary()
res = arimax.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()
pred = arimax.predict('2020-05-25','2020-05-31',exog = exog_tes[1:])
print('ARIMAX model MSE:{}'.format(mean_squared_error(tes,pred)))
pd.DataFrame({'test':tes['dayflow'],'pred':pred}).plot()
plt.show()
rp=arimax.plot_diagnostics(figsize=(15, 12))
sarimax = sm.tsa.statespace.SARIMAX(tra,order=(7,1,7),seasonal_order=(0,1,0,52),exog = exog_tra,
                                enforce_stationarity=False, enforce_invertibility=False,freq='D').fit()
sarimax.summary()
res = sarimax.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()
pred = sarimax.predict('2020-05-25','2020-05-31',exog = exog_tes[1:])
print('SARIMAX model MSE:{}'.format(mean_squared_error(tes,pred)))
pd.DataFrame({'test':tes.dayflow,'pred':pred}).plot();plt.show()
print('ARIMA model MSE:18276438.13483525')
print('SARIMA model MSE:67107012.27825963')
print('ARIMAX model MSE:14664808.6453892')
print('SARIMAX model MSE:96049210.37499459')
arimax.resid.plot();plt.show()
res_df = pd.DataFrame(arimax.resid,columns=['resid'])
res_df.sort_values(by='resid',ascending=False).head(5)
plt.figure(figsize=(10,15))
temp=daegu_dayflow.set_index('date')['2020'].diff().diff().dropna()
piv_val = temp.pivot_table(values='dayflow',
                          index=temp.index.day,
                          columns=temp.index.month,
                          aggfunc='mean')
sns.heatmap(piv_val)
plt.show()
temp