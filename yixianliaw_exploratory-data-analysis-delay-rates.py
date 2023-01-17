import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
Jan2019 = pd.read_csv("../input/flight-delay-prediction/Jan_2019_ontime.csv")

Jan2020 = pd.read_csv("../input/flight-delay-prediction/Jan_2020_ontime.csv")
Jan2019.info()
Jan2020.info()
Jan2019.drop('Unnamed: 21', axis=1, inplace=True)

Jan2020.drop('Unnamed: 21', axis=1, inplace=True)
Jan2019[Jan2019.OP_UNIQUE_CARRIER==Jan2019.OP_CARRIER].shape #All 583985 rows displayed
Jan2020[Jan2020.OP_UNIQUE_CARRIER==Jan2020.OP_CARRIER].shape #All 607346 rows displayed
Jan2019.drop('OP_UNIQUE_CARRIER',axis=1, inplace=True)

Jan2020.drop('OP_UNIQUE_CARRIER',axis=1, inplace=True)
CARRIER_NAME = pd.read_csv('../input/carrier-names-us/USCarrierNameCode.csv')

CARRIER_NAME
Jan2019 = pd.merge(left=Jan2019, right=CARRIER_NAME, how='inner', left_on='OP_CARRIER',right_on='IATA Code').drop('IATA Code',axis=1)

Jan2020 = pd.merge(left=Jan2020, right=CARRIER_NAME, how='inner', left_on='OP_CARRIER',right_on='IATA Code').drop('IATA Code',axis=1)
cols = Jan2019.columns.tolist() #Rearrange columns of the dataset after merging data with Carrier Name dataset

cols.insert(4,cols[-1])

del cols[-1]



Jan2019 = Jan2019[cols].rename(columns = {'Air Carrier Name': 'CARRIER_NAME'})

Jan2020 = Jan2020[cols].rename(columns = {'Air Carrier Name': 'CARRIER_NAME'})
Category_columns = list(range(0,13)) + list(range(14,16)) + list(range(17,20))

Jan2019.iloc[:,Category_columns] = Jan2019.iloc[:,Category_columns].astype('category')

Jan2020.iloc[:,Category_columns] = Jan2020.iloc[:,Category_columns].astype('category')
Jan2019.DEP_TIME = Jan2019.DEP_TIME.map(lambda x: str(x).zfill(4 + len(str(x)))[-6:-2]) #Extract string component of hour and minute

Jan2019.ARR_TIME = Jan2019.ARR_TIME.map(lambda x: str(x).zfill(4 + len(str(x)))[-6:-2])

Jan2020.DEP_TIME = Jan2020.DEP_TIME.map(lambda x: str(x).zfill(4 + len(str(x)))[-6:-2]) 

Jan2020.ARR_TIME = Jan2020.ARR_TIME.map(lambda x: str(x).zfill(4 + len(str(x)))[-6:-2])
Jan2019.loc[Jan2019['DEP_TIME'] =='2400','DEP_TIME']='0000' #46 rows (Set 2400 to 0000 per 24 hour system)

Jan2019.loc[Jan2019['ARR_TIME'] =='2400','ARR_TIME']='0000' #252 rows

Jan2020.loc[Jan2020['DEP_TIME'] =='2400','DEP_TIME']='0000' #30 rows

Jan2020.loc[Jan2020['ARR_TIME'] =='2400','ARR_TIME']='0000' #330 rows
NADelayARR_2019 = Jan2019[Jan2019.ARR_DEL15.isnull()] #18022 rows

NADelayDEP_2019 = Jan2019[Jan2019.DEP_DEL15.isnull()] #16355 rows

Jan2019_Comp = Jan2019[Jan2019.ARR_DEL15.notnull() & Jan2019.DEP_DEL15.notnull()] #565963 rows (Complete dataset with no missing values)
NADelayARR_2020 = Jan2020[Jan2020.ARR_DEL15.isnull()] #8078 rows

NADelayDEP_2020 = Jan2020[Jan2020.DEP_DEL15.isnull()] #6699 rows

Jan2020_Comp = Jan2020[Jan2020.ARR_DEL15.notnull() & Jan2020.DEP_DEL15.notnull()] #599268 rows (Complete dataset with no missing values)
temp = Jan2019_Comp.ARR_TIME.apply(lambda x: '0000-0559' if ((x>='0000') & (x<'0600')) else '0600-0659' if ((x>='0600') & (x<'0700')) \

else '0700-0759' if ((x>='0700') & (x<'0800')) else '0800-0859' if ((x>='0800') & (x<'0900')) else '0900-0959' if ((x>='0900') & (x<'1000')) \

else '1000-1059' if ((x>='1000') & (x<'1100')) else '1100-1159' if ((x>='1100') & (x<'1200')) else '1200-1259' if ((x>='1200') & (x<'1300')) \

else '1300-1359' if ((x>='1300') & (x<'1400')) else '1400-1459' if ((x>='1400') & (x<'1500')) else '1500-1559' if ((x>='1500') & (x<'1600')) \

else '1600-1659' if ((x>='1600') & (x<'1700')) else '1700-1759' if ((x>='1700') & (x<'1800')) else '1800-1859' if ((x>='1800') & (x<'1900')) \

else '1900-1959' if ((x>='1900') & (x<'2000')) else '2000-2059' if ((x>='2000') & (x<'2100')) else '2100-2159' if ((x>='2100') & (x<'2200')) \

else '2200-2259' if ((x>='2200') & (x<'2300')) else '2300-2359')

temp= temp.rename('ARR_TIME_BLK').astype('category')

Jan2019_Comp = pd.merge(left=Jan2019_Comp, right=temp, left_index=True, right_index=True)
temp = Jan2020_Comp.ARR_TIME.apply(lambda x: '0000-0559' if ((x>='0000') & (x<'0600')) else '0600-0659' if ((x>='0600') & (x<'0700')) \

else '0700-0759' if ((x>='0700') & (x<'0800')) else '0800-0859' if ((x>='0800') & (x<'0900')) else '0900-0959' if ((x>='0900') & (x<'1000')) \

else '1000-1059' if ((x>='1000') & (x<'1100')) else '1100-1159' if ((x>='1100') & (x<'1200')) else '1200-1259' if ((x>='1200') & (x<'1300')) \

else '1300-1359' if ((x>='1300') & (x<'1400')) else '1400-1459' if ((x>='1400') & (x<'1500')) else '1500-1559' if ((x>='1500') & (x<'1600')) \

else '1600-1659' if ((x>='1600') & (x<'1700')) else '1700-1759' if ((x>='1700') & (x<'1800')) else '1800-1859' if ((x>='1800') & (x<'1900')) \

else '1900-1959' if ((x>='1900') & (x<'2000')) else '2000-2059' if ((x>='2000') & (x<'2100')) else '2100-2159' if ((x>='2100') & (x<'2200')) \

else '2200-2259' if ((x>='2200') & (x<'2300')) else '2300-2359')

temp=temp.rename('ARR_TIME_BLK').astype('category')

Jan2020_Comp = pd.merge(left=Jan2020_Comp, right=temp, left_index=True, right_index=True)
Required_columns = [0,4,9] + list(range(12,18)) + [21] # Only relevant columns are extracted from the dataset for this part of the analysis

Jan2019_Ext = Jan2019_Comp.iloc[:,Required_columns]

Jan2020_Ext = Jan2020_Comp.iloc[:,Required_columns]
Jan2019_Ext.reset_index(drop=True, inplace=True) #Resetting index labels of extracted datasets

Jan2020_Ext.reset_index(drop=True, inplace=True)
Jan2019_Ext.head()
Jan2020_Ext.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = 16,12

sns.set(style='darkgrid')
Jan2019_Ext_1 = Jan2019_Ext.groupby(['DEP_DEL15','DEP_TIME_BLK']).size()

Jan2019_Ext_1 = (Jan2019_Ext_1 / Jan2019_Ext_1.groupby(level=1).sum()).reset_index().tail(19).reset_index(drop=True)

MeanDelayDep_1_2019 = Jan2019_Ext_1[0].mean()

Jan2019_Ext_1
Jan2019_Ext_2 = Jan2019_Ext.groupby(['ARR_DEL15','ARR_TIME_BLK']).size()

Jan2019_Ext_2 = (Jan2019_Ext_2 / Jan2019_Ext_2.groupby(level=1).sum()).reset_index().tail(19).reset_index(drop=True)

MeanDelayArr_2_2019 = Jan2019_Ext_2[0].mean()

Jan2019_Ext_2
fig1, ax = plt.subplots(2,2, sharex = True, sharey = False)

Chart1 = sns.barplot(data = Jan2019_Ext_1, x='DEP_TIME_BLK', y= 0, palette='Reds',hue=0, dodge=False, ax=ax[0,0], edgecolor='black')

Chart2 = sns.countplot(data=Jan2019_Ext, x='DEP_TIME_BLK', dodge=True, palette='autumn', ax=ax[1,0], edgecolor='black')

Chart3 = sns.barplot(data = Jan2019_Ext_2, x='ARR_TIME_BLK', y= 0, palette='Greens',hue=0, dodge=False, ax=ax[0,1], edgecolor='black')

Chart4 = sns.countplot(data=Jan2019_Ext, x='ARR_TIME_BLK', dodge=True, palette='cool', ax=ax[1,1], edgecolor='black')



ax[0,0].tick_params(axis="y", labelsize=15)

ax[0,0].get_legend().remove()

ax[0,0].text(0.2,0.24,'---- y={}'.format(round(MeanDelayDep_1_2019,4)),fontsize=16)



ax[1,0].tick_params(axis="x", labelsize=15, rotation=90)

ax[1,0].tick_params(axis="y", labelsize=15)



ax[0,1].tick_params(axis="y", labelsize=15)

ax[0,1].get_legend().remove()

ax[0,1].text(13,0.45,'---- y={}'.format(round(MeanDelayArr_2_2019,4)),fontsize=16)



ax[1,1].tick_params(axis="x", labelsize=15, rotation=90)

ax[1,1].tick_params(axis="y", labelsize=15)



Chart1.set_xlabel('')

Chart1.set_ylabel('Departure Delay Rate', fontsize=20)

Chart1.axhline(y=MeanDelayDep_1_2019, linestyle='--', color='black')



Chart2.set_xlabel('Departure Timeslots', fontsize=20)

Chart2.set_ylabel('Number of flights', fontsize=20)



Chart3.set_xlabel('')

Chart3.set_ylabel('Arrival Delay Rate', fontsize=20)

Chart3.axhline(y=MeanDelayArr_2_2019, linestyle='--', color='black')



Chart4.set_xlabel('Arrival Timeslots', fontsize=20)

Chart4.set_ylabel('')



plt.suptitle('Flight Departure and Arrival Delay Rate on January 2019', fontsize=30,y=0.93)

plt.show()
Jan2020_Ext_1 = Jan2020_Ext.groupby(['DEP_DEL15','DEP_TIME_BLK']).size()

Jan2020_Ext_1 = (Jan2020_Ext_1 / Jan2020_Ext_1.groupby(level=1).sum()).reset_index().tail(19).reset_index(drop=True)

MeanDelayDep_1_2020 = Jan2020_Ext_1[0].mean()

Jan2020_Ext_1
Jan2020_Ext_2 = Jan2020_Ext.groupby(['ARR_DEL15','ARR_TIME_BLK']).size()

Jan2020_Ext_2 = (Jan2020_Ext_2 / Jan2020_Ext_2.groupby(level=1).sum()).reset_index().tail(19).reset_index(drop=True)

MeanDelayArr_2_2020 = Jan2020_Ext_2[0].mean()

Jan2020_Ext_2
plt.rcParams['figure.figsize'] = 16,12

fig2, ax = plt.subplots(2,2, sharex = True, sharey = False)



Chart5 = sns.barplot(data = Jan2020_Ext_1, x='DEP_TIME_BLK', y= 0, palette='Reds',hue=0, dodge=False, ax=ax[1,0], edgecolor='black')

Chart6 = sns.barplot(data = Jan2019_Ext_1, x='DEP_TIME_BLK', y= 0, palette='Reds',hue=0, dodge=False, ax=ax[0,0], edgecolor='black')

Chart7 = sns.barplot(data = Jan2020_Ext_2, x='ARR_TIME_BLK', y= 0, palette='Greens',hue=0, dodge=False, ax=ax[1,1], edgecolor='black')

Chart8 = sns.barplot(data = Jan2019_Ext_2, x='ARR_TIME_BLK', y= 0, palette='Greens',hue=0, dodge=False, ax=ax[0,1], edgecolor='black')



ax[0,0].tick_params(axis="y", labelsize=15)

ax[0,0].get_legend().remove()

ax[0,0].text(0.2,0.24,'---- y={}'.format(round(MeanDelayDep_1_2019,4)),fontsize=16)



ax[0,1].tick_params(axis="y", labelsize=15)

ax[0,1].get_legend().remove()

ax[0,1].text(13,0.45,'---- y={}'.format(round(MeanDelayArr_2_2019,4)),fontsize=16)



ax[1,0].tick_params(axis="x", labelsize=15, rotation = 90)

ax[1,0].tick_params(axis="y", labelsize=15)

ax[1,0].get_legend().remove()

ax[1,0].text(0.2,0.18,'---- y={}'.format(round(MeanDelayDep_1_2020,4)),fontsize=16)



ax[1,1].tick_params(axis="x", labelsize=15, rotation = 90)

ax[1,1].tick_params(axis="y", labelsize=15)

ax[1,1].get_legend().remove()

ax[1,1].text(13,0.31,'---- y={}'.format(round(MeanDelayArr_2_2020,4)),fontsize=16)



Chart5.set_xlabel('Departure Timeslots', fontsize=20)

Chart5.set_ylabel('Departure Delay Rate (2020)', fontsize=20)

Chart5.axhline(y=MeanDelayDep_1_2020, linestyle='--', color='black')



Chart6.set_xlabel('')

Chart6.set_ylabel('Departure Delay Rate (2019)', fontsize=20)

Chart6.axhline(y=MeanDelayDep_1_2019, linestyle='--', color='black')



Chart7.set_xlabel('Arrival Timeslots', fontsize=20)

Chart7.set_ylabel('Arrival Delay Rate (2020)', fontsize=20)

Chart7.axhline(y=MeanDelayArr_2_2020, linestyle='--', color='black')



Chart8.set_xlabel('')

Chart8.set_ylabel('Arrival Delay Rate (2019)', fontsize=20)

Chart8.axhline(y=MeanDelayArr_2_2019, linestyle='--', color='black')



plt.suptitle('Departure and Arrival Delay Rate over Timeslots', fontsize=25,y=0.95)

plt.show()
Jan2019_Ext_3 = Jan2019_Ext.groupby(['DEP_DEL15','CARRIER_NAME']).size()

Jan2019_Ext_3 = (Jan2019_Ext_3 / Jan2019_Ext_3.groupby(level=1).sum()).reset_index().tail(17).reset_index(drop=True).sort_values(by=0, ascending=False)

GROUP3_Order_2019 = Jan2019_Ext_3['CARRIER_NAME']

MeanDelayDep_2019 = Jan2019_Ext_3[0].mean()

Jan2019_Ext_3
plt.rcParams['figure.figsize'] = 16,12

sns.set(style='white')

fig3, ax = plt.subplots(1,2, sharex = False, sharey = True)

Chart9 = sns.barplot(data = Jan2019_Ext_3, y='CARRIER_NAME', x= 0, palette='Reds',hue=0, ax=ax[0],dodge=False, edgecolor='black',order=GROUP3_Order_2019)

Chart10 = sns.countplot(data=Jan2019_Ext, y='CARRIER_NAME', ax=ax[1],dodge=False, palette='autumn', edgecolor='black',order=GROUP3_Order_2019)

ax[0].tick_params(axis="both", labelsize=15)

ax[1].tick_params(axis="both", labelsize=15)

ax[0].get_legend().remove()

ax[0].text(0.2,16,'---- x={}'.format(round(MeanDelayDep_2019,4)),fontsize=16)

Chart9.set_xlabel('Departure Delay Rate', fontsize=20)

Chart9.set_ylabel('Carrier Name', fontsize=20)

Chart9.axvline(x=MeanDelayDep_2019, linestyle='--', color='black')

Chart10.set_ylabel('')

Chart10.set_xlabel('Number of flights', fontsize=20)

plt.suptitle('Ranking of Flight Departure Delay Rate by Carriers on January 2019', fontsize=30, x=0.42, y=0.95)

plt.show()
Jan2019_Ext_4 = Jan2019_Ext.groupby(['ARR_DEL15','CARRIER_NAME']).size()

Jan2019_Ext_4 = (Jan2019_Ext_4 / Jan2019_Ext_4.groupby(level=1).sum()).reset_index().tail(17).sort_values(by=0, ascending=False)

GROUP4_Order_2019 = Jan2019_Ext_4['CARRIER_NAME']

MeanDelayArr_2019 = Jan2019_Ext_4[0].mean()

Jan2019_Ext_4
plt.rcParams['figure.figsize'] = 16,12

sns.set(style='white')

fig4, ax = plt.subplots(1,2, sharex = False, sharey = True)

Chart11 = sns.barplot(data = Jan2019_Ext_4, y='CARRIER_NAME', x= 0, palette='Greens',hue=0, ax=ax[0],dodge=False, edgecolor='black',order=GROUP4_Order_2019)

Chart12 = sns.countplot(data=Jan2019_Ext, y='CARRIER_NAME', ax=ax[1],dodge=False, palette='cool', edgecolor='black',order=GROUP4_Order_2019)

ax[0].tick_params(axis="both", labelsize=15)

ax[1].tick_params(axis="both", labelsize=15)

ax[0].get_legend().remove()

ax[0].text(0.21,16,'---- x={}'.format(round(MeanDelayArr_2019,4)),fontsize=16)

Chart11.set_xlabel('Arrival Delay Rate', fontsize=20)

Chart11.set_ylabel('Carrier Name', fontsize=20)

Chart11.axvline(x=MeanDelayArr_2019, linestyle='--', color='black')

Chart12.set_ylabel('')

Chart12.set_xlabel('Number of flights', fontsize=20)

plt.suptitle('Ranking of Flight Arrival Delay Rates by Carriers on January 2019', fontsize=30, x=0.42, y=0.95)

plt.show()
Jan2020_Ext_3 = Jan2020_Ext.groupby(['DEP_DEL15','CARRIER_NAME']).size()

Jan2020_Ext_3 = (Jan2020_Ext_3 / Jan2020_Ext_3.groupby(level=1).sum()).reset_index().tail(17).reset_index(drop=True).sort_values(by=0, ascending=False)

GROUP3_Order_2020 = Jan2020_Ext_3['CARRIER_NAME']

MeanDelayDep_2020 = Jan2020_Ext_3[0].mean()

Jan2020_Ext_3
Jan2020_Ext_4 = Jan2020_Ext.groupby(['ARR_DEL15','CARRIER_NAME']).size()

Jan2020_Ext_4 = (Jan2020_Ext_4 / Jan2020_Ext_4.groupby(level=1).sum()).reset_index().tail(17).reset_index(drop=True).sort_values(by=0, ascending=False)

GROUP4_Order_2020 = Jan2020_Ext_4['CARRIER_NAME']

MeanDelayArr_2020 = Jan2020_Ext_4[0].mean()

Jan2020_Ext_4
plt.rcParams['figure.figsize'] = 20,20

fig5, ax = plt.subplots(2,2, sharex = False, sharey = False)



Chart13 = sns.barplot(data = Jan2019_Ext_3, y='CARRIER_NAME', x= 0, palette='Reds',hue=0, ax=ax[0,0],dodge=False, edgecolor='black',order=GROUP3_Order_2019)

Chart14 = sns.barplot(data = Jan2019_Ext_4, y='CARRIER_NAME', x= 0, palette='Greens',hue=0, ax=ax[1,0],dodge=False, edgecolor='black',order=GROUP4_Order_2019)

Chart15 = sns.barplot(data = Jan2020_Ext_3, y='CARRIER_NAME', x= 0, palette='Reds',hue=0, ax=ax[0,1],dodge=False, edgecolor='black',order=GROUP3_Order_2020)

Chart16 = sns.barplot(data = Jan2020_Ext_4, y='CARRIER_NAME', x= 0, palette='Greens',hue=0, ax=ax[1,1],dodge=False, edgecolor='black',order=GROUP4_Order_2020)



ax[0,0].tick_params(axis="both", labelsize=15)

ax[0,0].get_legend().remove()

ax[0,0].text(0.21,16,'---- x={}'.format(round(MeanDelayDep_2019,4)),fontsize=16)



ax[0,1].tick_params(axis="both", labelsize=15)

ax[0,1].get_legend().remove()

ax[0,1].text(0.15,16,'---- x={}'.format(round(MeanDelayDep_2020,4)),fontsize=16)



ax[1,0].tick_params(axis="both", labelsize=15)

ax[1,0].get_legend().remove()

ax[1,0].text(0.21,16,'---- x={}'.format(round(MeanDelayArr_2019,4)),fontsize=16)



ax[1,1].tick_params(axis="both", labelsize=15)

ax[1,1].get_legend().remove()

ax[1,1].text(0.17,16,'---- x={}'.format(round(MeanDelayArr_2020,4)),fontsize=16)



Chart13.set_xlabel('Departure Delay Rate (2019)', fontsize=20)

Chart13.set_ylabel('Carrier Name', fontsize=20)

Chart13.axvline(x=MeanDelayDep_2019, linestyle='--', color='black')



Chart15.set_xlabel('Departure Delay Rate (2020)', fontsize=20)

Chart15.set_ylabel('')

Chart15.axvline(x=MeanDelayDep_2020, linestyle='--', color='black')



Chart14.set_xlabel('Arrival Delay Rate (2019)', fontsize=20)

Chart14.set_ylabel('Carrier Name', fontsize=20)

Chart14.axvline(x=MeanDelayArr_2019, linestyle='--', color='black')



Chart16.set_xlabel('Arrival Delay Rate (2020)', fontsize=20)

Chart16.set_ylabel('')

Chart16.axvline(x=MeanDelayArr_2020, linestyle='--', color='black')



plt.suptitle('Ranking of Departure and Arrival Delay Rate by Carriers ', fontsize=35, y=1.01)

fig5.tight_layout(pad=2.0)

plt.show()
Jan2019_Ext_5 = Jan2019_Ext.groupby(['DEP_DEL15','CARRIER_NAME','DAY_OF_MONTH']).size()

Total_5 = Jan2019_Ext_5.loc[pd.IndexSlice[1,],]+Jan2019_Ext_5.loc[pd.IndexSlice[0,],]

Prop_NoDelayDep = Jan2019_Ext_5.loc[pd.IndexSlice[0,],]/Total_5

Prop_DelayDep = Jan2019_Ext_5.loc[pd.IndexSlice[1,],]/Total_5

Jan2019_Ext_5 = pd.DataFrame([Prop_NoDelayDep,Prop_DelayDep]).round(4).transpose().sort_values(by=['CARRIER_NAME','DAY_OF_MONTH'],ascending=[True,True]).reset_index()

MeanDelayDepCar_2019= Jan2019_Ext_5.groupby('CARRIER_NAME').mean().reset_index()[1]

MeanDelayDep_2019 = Jan2019_Ext_5[1].mean()

Jan2019_Ext_5['YEAR'] = 2019

Jan2019_Ext_5
Jan2020_Ext_5 = Jan2020_Ext.groupby(['DEP_DEL15','CARRIER_NAME','DAY_OF_MONTH']).size()

Total_5 = Jan2020_Ext_5.loc[pd.IndexSlice[1,],]+Jan2020_Ext_5.loc[pd.IndexSlice[0,],]

Prop_NoDelayDep = Jan2020_Ext_5.loc[pd.IndexSlice[0,],]/Total_5 

Prop_DelayDep = Jan2020_Ext_5.loc[pd.IndexSlice[1,],]/Total_5

Jan2020_Ext_5 = pd.DataFrame([Prop_NoDelayDep,Prop_DelayDep]).round(4).transpose().sort_values(by=['CARRIER_NAME','DAY_OF_MONTH'],ascending=[True,True]).reset_index()

MeanDelayDepCar_2020= Jan2020_Ext_5.groupby('CARRIER_NAME').mean().reset_index()[1]

MeanDelayDep_2020 = Jan2020_Ext_5[1].mean()

Jan2020_Ext_5['YEAR'] = 2020

Jan2020_Ext_5
Jan_Comb_Ext_5 = pd.concat([Jan2019_Ext_5, Jan2020_Ext_5]).reset_index(drop=True) # Merging 2019 and 2020 grouped data set.

Jan_Comb_Ext_5['YEAR'] = Jan_Comb_Ext_5.YEAR.astype('category')

Jan_Comb_Ext_5['CATEGORY'] = (Jan_Comb_Ext_5['CARRIER_NAME'].astype('str') + ' ' + Jan_Comb_Ext_5['YEAR'].astype('str')).astype('category')
colorlist = ['#FF0000','#FF4600','#FF8300','#FFC100','#FFF000','#C9FF00','#83FF00','#00FF1B','#00FF74','#00FFD1','#00E4FF','#0080FF','#0032FF','#7400FF','#B200FF','#FF00E8','#FF006C']

plt.rcParams['figure.figsize'] = 50,50

sns.set(style='darkgrid', font_scale=2)

grid1 = sns.FacetGrid(Jan_Comb_Ext_5, col='CATEGORY', col_wrap=4, height=6, aspect=1.1)



CarrierList = Jan_Comb_Ext_5.CARRIER_NAME.unique()

count=0



for ax in grid1.axes.flat:

    if(count%2==0): #plotting 2019 and 2020 chart for each airline carrier side by side

        sns.lineplot(x='DAY_OF_MONTH', y=1, data = Jan_Comb_Ext_5[(Jan_Comb_Ext_5['CARRIER_NAME'] == CarrierList[count//2]) & (Jan_Comb_Ext_5['YEAR']==2019)], ax=ax, color = colorlist[count//2], linewidth=3)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2019')

        ax.axhline(y=MeanDelayDepCar_2019[count//2],ls='--',color='black') # average departure delay rates for each carrier in 2019

        ax.text(20,0.59,'---- y={}'.format(MeanDelayDepCar_2019[count//2].round(4)),fontsize=20)

        ax.axhline(y=MeanDelayDep_2019,color='purple') # overall average departure delay rate in 2019

        ax.text(20,0.55,'---- y={}'.format(round(MeanDelayDep_2019,4)),fontsize=20,color='purple')

    else:

        sns.lineplot(x='DAY_OF_MONTH', y=1, data = Jan_Comb_Ext_5[(Jan_Comb_Ext_5['CARRIER_NAME'] == CarrierList[count//2]) & (Jan_Comb_Ext_5['YEAR']==2020)], ax=ax, color = colorlist[count//2], linewidth=3)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2020')

        ax.axhline(y=MeanDelayDepCar_2020[count//2],ls='--',color='black') # average departure delay rates for each carrier in 2020

        ax.text(20,0.59,'---- y={}'.format(MeanDelayDepCar_2020[count//2].round(4)),fontsize=20)

        ax.axhline(y=MeanDelayDep_2020,color='purple') # overall average departure delay rate in 2020

        ax.text(20,0.55,'---- y={}'.format(round(MeanDelayDep_2020,4)),fontsize=20,color='purple')

    count=count+1



grid1.set_axis_labels('Day of Month', 'Departure Delay Rate')

plt.suptitle('Daily Trend of Flight Departure Delay Rates on January 2019 & 2020', fontsize=30, x=0.5, y=1.01)

plt.show()
Jan2019_Ext_6 = Jan2019_Ext.groupby(['ARR_DEL15','CARRIER_NAME','DAY_OF_MONTH']).size()

Total_6 = Jan2019_Ext_6.loc[pd.IndexSlice[1,],]+Jan2019_Ext_6.loc[pd.IndexSlice[0,],]

Prop_NoDelayArr = Jan2019_Ext_6.loc[pd.IndexSlice[0,],]/Total_6 

Prop_DelayArr = Jan2019_Ext_6.loc[pd.IndexSlice[1,],]/Total_6

Jan2019_Ext_6= pd.DataFrame([Prop_NoDelayArr,Prop_DelayArr]).round(4).transpose().sort_values(by=['CARRIER_NAME','DAY_OF_MONTH'],ascending=[True,True]).reset_index()

MeanDelayArrCar_2019= Jan2019_Ext_6.groupby('CARRIER_NAME').mean().reset_index()[1]

MeanDelayArr_2019 = Jan2019_Ext_6[1].mean()

Jan2019_Ext_6['YEAR'] = 2019

Jan2019_Ext_6
Jan2020_Ext_6 = Jan2020_Ext.groupby(['ARR_DEL15','CARRIER_NAME','DAY_OF_MONTH']).size()

Total_6 = Jan2020_Ext_6.loc[pd.IndexSlice[1,],]+Jan2020_Ext_6.loc[pd.IndexSlice[0,],]

Prop_NoDelayArr = Jan2020_Ext_6.loc[pd.IndexSlice[0,],]/Total_6 

Prop_DelayArr = Jan2020_Ext_6.loc[pd.IndexSlice[1,],]/Total_6

Jan2020_Ext_6= pd.DataFrame([Prop_NoDelayArr,Prop_DelayArr]).round(4).transpose().sort_values(by=['CARRIER_NAME','DAY_OF_MONTH'],ascending=[True,True]).reset_index()

MeanDelayArrCar_2020= Jan2020_Ext_6.groupby('CARRIER_NAME').mean().reset_index()[1]

MeanDelayArr_2020 = Jan2020_Ext_6[1].mean()

Jan2020_Ext_6['YEAR'] = 2020

Jan2020_Ext_6
Jan_Comb_Ext_6 = pd.concat([Jan2019_Ext_6, Jan2020_Ext_6]).reset_index(drop=True)

Jan_Comb_Ext_6['YEAR'] = Jan_Comb_Ext_6.YEAR.astype('category')

Jan_Comb_Ext_6['CATEGORY'] = (Jan_Comb_Ext_6['CARRIER_NAME'].astype('str') + ' ' + Jan_Comb_Ext_6['YEAR'].astype('str')).astype('category')
colorlist = ['#FF0000','#FF4600','#FF8300','#FFC100','#FFF000','#C9FF00','#83FF00','#00FF1B','#00FF74','#00FFD1','#00E4FF','#0080FF','#0032FF','#7400FF','#B200FF','#FF00E8','#FF006C']

plt.rcParams['figure.figsize'] = 50,50

grid2 = sns.FacetGrid(Jan_Comb_Ext_6, col='CATEGORY', col_wrap=4, height=6, aspect=1.1)

count=0



CarrierList = Jan_Comb_Ext_6.CARRIER_NAME.unique()

for ax in grid2.axes.flat:

    if(count%2==0): #plotting 2019 and 2020 chart for each airline carrier side by side

        sns.lineplot(x='DAY_OF_MONTH', y=1, data = Jan_Comb_Ext_6[(Jan_Comb_Ext_6['CARRIER_NAME'] == CarrierList[count//2]) & (Jan_Comb_Ext_6['YEAR']==2019)], ax=ax, color = colorlist[count//2], linewidth=2)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2019')

        ax.axhline(y=MeanDelayArrCar_2019[count//2],ls='--',color='black') # average arrival delay rate for each carrier in 2019

        ax.text(20,0.59,'---- y={}'.format(MeanDelayArrCar_2019[count//2].round(4)),fontsize=20)

        ax.axhline(y=MeanDelayArr_2019,color='purple') # overall average arrival delay rate in 2019

        ax.text(20,0.55,'---- y={}'.format(round(MeanDelayArr_2019,4)),fontsize=20,color='purple')

    else:

        sns.lineplot(x='DAY_OF_MONTH', y=1, data = Jan_Comb_Ext_6[(Jan_Comb_Ext_6['CARRIER_NAME'] == CarrierList[count//2]) & (Jan_Comb_Ext_6['YEAR']==2020)], ax=ax, color = colorlist[count//2], linewidth=2)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2020')

        ax.axhline(y=MeanDelayArrCar_2020[count//2],ls='--',color='black') # average arrival delay rate for each carrier in 2020

        ax.text(20,0.59,'---- y={}'.format(MeanDelayArrCar_2020[count//2].round(4)),fontsize=20)

        ax.axhline(y=MeanDelayArr_2020,color='purple') # overall average arrival delay rate in 2020

        ax.text(20,0.55,'---- y={}'.format(round(MeanDelayArr_2020,4)),fontsize=20,color='purple')

    count=count+1



grid2.set_axis_labels('Day of Month', 'Arrival Delay Rate')

plt.suptitle('Daily Trend of Flight Arrival Delay Rates on January 2019 & 2020', fontsize=30, x=0.5, y=1.01)

plt.show()
Top10Dest_2019 = Jan2019_Ext.groupby("DEST").size().reset_index().sort_values(by=0,ascending=False).head(10).reset_index(drop=True)

Top10Dest_2019['DEST'] = Top10Dest_2019['DEST'].astype('str').astype('category')

Top10Arr_Sub_2019 = Jan2019_Ext[Jan2019_Ext['DEST'].isin(list(Top10Dest_2019['DEST']))].reset_index()

Top10Arr_Sub_2019['DEST'] = Top10Arr_Sub_2019['DEST'].astype('str').astype('category')

Top10Arr_Sub_2019.head()
Top10Dest_2020 = Jan2020_Ext.groupby("DEST").size().reset_index().sort_values(by=0,ascending=False).head(10).reset_index(drop=True)

Top10Dest_2020['DEST'] = Top10Dest_2020['DEST'].astype('str').astype('category')

Top10Arr_Sub_2020 = Jan2020_Ext[Jan2020_Ext['DEST'].isin(list(Top10Dest_2020['DEST']))].reset_index()

Top10Arr_Sub_2020['DEST'] = Top10Arr_Sub_2020['DEST'].astype('str').astype('category')

Top10Arr_Sub_2020.head()
plt.rcParams['figure.figsize'] = 20,10

sns.set(style='white')

fig7, ax = plt.subplots(1,2, sharex = False, sharey = False)

Chart17 = sns.countplot(data=Top10Arr_Sub_2019, x='DEST', dodge=True, palette='cool',edgecolor='black', ax=ax[0], order=Top10Dest_2019.DEST)

Chart17.set_xlabel('Destinations', fontsize=20)

Chart17.set_ylabel('Number of flights (2019)', fontsize=20)

Chart17.tick_params(axis="both", labelsize=15)



Chart18 = sns.countplot(data=Top10Arr_Sub_2020, x='DEST', dodge=True, palette='cool',edgecolor='black', ax=ax[1], order=Top10Dest_2020.DEST)

Chart18.set_xlabel('Destinations', fontsize=20)

Chart18.set_ylabel('Number of flights (2020)', fontsize=20)

Chart18.tick_params(axis="both", labelsize=15)



plt.suptitle('Top 10 Most Frequent Destinations for Travel on January 2019 and 2020', fontsize=20,y=0.92)

plt.show()
Jan2019_Ext_8 = Jan2019_Ext.groupby(['DEP_DEL15','DEST']).size()

Jan2019_Ext_8 = (Jan2019_Ext_8 / Jan2019_Ext_8.groupby(level=1).sum()).reset_index().tail(346).reset_index(drop=True)

Jan2019_Ext_8 = Jan2019_Ext_8[Jan2019_Ext_8['DEST'].isin(Top10Dest_2019['DEST'])]

MeanDelayDep_2019 = Jan2019_Ext_8[0].mean()

Jan2019_Ext_8
Jan2020_Ext_8 = Jan2020_Ext.groupby(['DEP_DEL15','DEST']).size()

Jan2020_Ext_8 = (Jan2020_Ext_8 / Jan2020_Ext_8.groupby(level=1).sum()).reset_index().tail(346).reset_index(drop=True)

Jan2020_Ext_8 = Jan2020_Ext_8[Jan2020_Ext_8['DEST'].isin(Top10Dest_2020['DEST'])]

MeanDelayDep_2020 = Jan2020_Ext_8[0].mean()

Jan2020_Ext_8
Jan2019_Ext_9 = Jan2019_Ext.groupby(['ARR_DEL15','DEST']).size()

Jan2019_Ext_9 = (Jan2019_Ext_9 / Jan2019_Ext_9.groupby(level=1).sum()).reset_index().tail(346).reset_index(drop=True)

Jan2019_Ext_9 = Jan2019_Ext_9[Jan2019_Ext_9['DEST'].isin(Top10Dest_2019['DEST'])]

MeanDelayArr_2019 = Jan2019_Ext_9[0].mean()

Jan2019_Ext_9
Jan2020_Ext_9 = Jan2019_Ext.groupby(['ARR_DEL15','DEST']).size()

Jan2020_Ext_9 = (Jan2020_Ext_9 / Jan2020_Ext_9.groupby(level=1).sum()).reset_index().tail(346).reset_index(drop=True)

Jan2020_Ext_9 = Jan2020_Ext_9[Jan2020_Ext_9['DEST'].isin(Top10Dest_2020['DEST'])]

MeanDelayArr_2020 = Jan2020_Ext_9[0].mean()

Jan2020_Ext_9
plt.rcParams['figure.figsize'] = 18,18

fig8, ax = plt.subplots(2,2, sharex = False, sharey = False)

Chart19 = sns.barplot(data = Jan2019_Ext_8, x='DEST', y= 0, palette='Reds',ax=ax[0,0], hue=0, dodge=False, edgecolor='black',order=Top10Dest_2019.DEST)

Chart20 = sns.barplot(data = Jan2020_Ext_8, x='DEST', y= 0, palette='Reds',ax=ax[1,0], hue=0, dodge=False, edgecolor='black',order=Top10Dest_2020.DEST)

Chart21 = sns.barplot(data = Jan2019_Ext_9, x='DEST', y= 0, palette='Greens',ax=ax[0,1], hue=0, dodge=False, edgecolor='black',order=Top10Dest_2019.DEST)

Chart22 = sns.barplot(data = Jan2020_Ext_9, x='DEST', y= 0, palette='Greens',ax=ax[1,1], hue=0, dodge=False, edgecolor='black',order=Top10Dest_2020.DEST)



ax[0,0].tick_params(axis="both", labelsize=15)

ax[0,0].get_legend().remove()

ax[0,0].text(7,0.33,'---- y={}'.format(round(MeanDelayDep_2019,4)),fontsize=16)



ax[1,0].tick_params(axis="both", labelsize=15)

ax[1,0].get_legend().remove()

ax[1,0].text(7,0.16,'---- y={}'.format(round(MeanDelayDep_2020,4)),fontsize=16)



ax[0,1].text(7,0.38,'---- y={}'.format(round(MeanDelayArr_2019,4)),fontsize=16)

ax[0,1].tick_params(axis="both", labelsize=15)

ax[0,1].get_legend().remove()



ax[1,1].text(7,0.38,'---- y={}'.format(round(MeanDelayArr_2020,4)),fontsize=16)

ax[1,1].tick_params(axis="both", labelsize=15)

ax[1,1].get_legend().remove()



Chart19.set_xlabel('')

Chart19.set_ylabel('Departure Delay Rate (2019)', fontsize=20)

Chart19.axhline(y=MeanDelayDep_2019, linestyle='--', color='black')

Chart19.set_ylim(0,0.35)



Chart20.set_xlabel('Destination', fontsize=20)

Chart20.set_ylabel('Departure Delay Rate (2020)', fontsize=20)

Chart20.axhline(y=MeanDelayDep_2020, linestyle='--', color='black')



Chart21.set_xlabel('')

Chart21.set_ylabel('Arrival Delay Rate (2019)', fontsize=20)

Chart21.axhline(y=MeanDelayArr_2019, linestyle='--', color='black')

Chart21.set_ylim(0,0.40)



Chart22.set_xlabel('Destination', fontsize=20)

Chart22.set_ylabel('Arrival Delay Rate (2020)', fontsize=20)

Chart22.axhline(y=MeanDelayArr_2020, linestyle='--', color='black')

Chart22.set_ylim(0,0.40)



plt.suptitle('Flight Delay Rates for Top 10 Destinations on January 2019 & 2020', fontsize=20,y=0.9)

plt.show()
labels = ['No Delay', 'Arrival delay only', 'Departure delay only', 'Departure and arrival delay']

Jan2019_Comp_Sub = Jan2019_Comp.groupby(['DEP_DEL15','ARR_DEL15']).size().reset_index()

Jan2020_Comp_Sub = Jan2020_Comp.groupby(['DEP_DEL15','ARR_DEL15']).size().reset_index()
plt.rcParams['figure.figsize'] = 16,16

sns.set(font_scale=2)

color = ['#5564FF', '#FFF094', '#87FD8C', '#FE67B5']

fig8, ax = plt.subplots(1,2, sharex = False, sharey = False)

Pie1 = ax[0].pie(Jan2019_Comp_Sub[0], autopct='%1.1f%%', startangle=90, shadow=True, textprops={'size': 'x-small'}, pctdistance=0.85, colors=color)

Pie2 = ax[1].pie(Jan2020_Comp_Sub[0], autopct='%1.1f%%', startangle=90, shadow=True, textprops={'size': 'x-small'}, pctdistance=0.85, colors=color)

leg = plt.legend(labels, loc='center right',bbox_to_anchor=(1.1, -0.1, 0.5, 1), title='Delay Status', fontsize=14, facecolor='white')

leg._legend_box.align = "left"

plt.suptitle('Flight delay status on January 2019 and 2020', y=0.7)

plt.text(s='January 2019', y=-1.3, x=-3.5)

plt.text(s='January 2020', y=-1.3, x=-0.4)

plt.show()
Jan2019_Comp_SubCar = Jan2019_Comp.groupby(['DEP_DEL15','ARR_DEL15','CARRIER_NAME']).size().reset_index()

Jan2019_Comp_SubCar['YEAR'] = 2019

Jan2020_Comp_SubCar = Jan2020_Comp.groupby(['DEP_DEL15','ARR_DEL15','CARRIER_NAME']).size().reset_index()

Jan2020_Comp_SubCar['YEAR'] = 2020
Jan_Comb_SubCar = pd.concat([Jan2019_Comp_SubCar,Jan2020_Comp_SubCar])

Jan_Comb_SubCar['CATEGORY'] = (Jan_Comb_SubCar['CARRIER_NAME'].astype('str') + ' ' + Jan_Comb_SubCar['YEAR'].astype('str')).astype('category')
plt.rcParams['figure.figsize'] = 20,20

grid3 = sns.FacetGrid(Jan_Comb_SubCar, col='CATEGORY', col_wrap=4, height=6, aspect=1.1)

count=0

CarrierList = Jan_Comb_SubCar.CARRIER_NAME.unique()

for ax in grid3.axes.flat:

    if(count%2==0): #plotting 2019 and 2020 chart for each airline carrier side by side

        ax.pie(Jan_Comb_SubCar[(Jan_Comb_SubCar['CARRIER_NAME']==CarrierList[count//2]) & (Jan_Comb_SubCar['YEAR']==2019)][0], autopct='%1.1f%%', startangle=90, textprops={'size': 'small'}, pctdistance = 1.17, colors=color)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2019')

    else:

        ax.pie(Jan_Comb_SubCar[(Jan_Comb_SubCar['CARRIER_NAME']==CarrierList[count//2]) & (Jan_Comb_SubCar['YEAR']==2020)][0], autopct='%1.1f%%', startangle=90, textprops={'size': 'small'}, pctdistance = 1.17, colors=color)

        ax.set_title(str(CarrierList[count//2]).strip() + ' | ' + '2020')

    count=count+1



leg = plt.legend(labels, loc='center right',bbox_to_anchor=(1.8, 0, 0.5, 1), title='Delay Status', fontsize=20, facecolor='white')

leg._legend_box.align = "left"



plt.suptitle('Flight Delay Status across Carriers on January 2019 and January 2020', fontsize=30, y=1.02)

plt.show()