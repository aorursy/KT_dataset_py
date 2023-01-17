import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/pivot.csv")
df.head(10)
cleaned = df.loc[2:37,"Unnamed: 2":"Unnamed: 7"]
cleaned.columns = ['category', 'year', 'q1','q2', 'q3', 'q4']
cleaned['year'] = cleaned['year'].astype('int')
cleaned['q1'] = cleaned['q1'].astype('float')
cleaned['q2'] = cleaned['q2'].astype('float')
cleaned['q3'] = cleaned['q3'].astype('float')
cleaned['q4'] = cleaned['q4'].astype('float')
cleaned = cleaned.reset_index(drop=True)
import warnings
warnings.filterwarnings("ignore")
cleaned.category[0:9] = 'goat_meat'
cleaned.category[9:18] = 'mutton'
cleaned.category[18:27] = 'water_buffalo'
cleaned.category[27:] = 'cattle_calf'
cleaned
gmp_q1 = cleaned[0:9][['year','q1']]
gmp_q1 = gmp_q1.reset_index(drop=True)
gmp_q1
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q1', data=gmp_q1, color="gold", alpha=0.45,)
plt.plot( 'year', 'q1', data=gmp_q1,label = 'Goat Meat', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Goat Meat Production Q1')
ax = plt.gca()
ax.set_ylim(0,10000,2000)
plt.show()
gmp_q2 = cleaned[0:9][['year','q2']]
gmp_q2 = gmp_q2.reset_index(drop=True)
gmp_q2
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q2', data=gmp_q2, color="red", alpha=0.45,)
plt.plot( 'year', 'q2', data=gmp_q2,label = 'Goat Meat', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Goat Meat Production Q2')
ax = plt.gca()
ax.set_ylim(0,10000,2000)
plt.show()
gmp_q3 = cleaned[0:9][['year','q3']]
gmp_q3 = gmp_q3.reset_index(drop=True)
gmp_q3
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q3', data=gmp_q3, color="darkviolet", alpha=0.45,)
plt.plot( 'year', 'q3', data=gmp_q3,label = 'Goat Meat', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Goat Meat Production Q3')
ax = plt.gca()
ax.set_ylim(0,16000,2000)
plt.show()
gmp_q4 = cleaned[0:9][['year','q4']]
gmp_q4 = gmp_q4.reset_index(drop=True)
gmp_q4
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q4', data=gmp_q4, color="limegreen", alpha=0.45,)
plt.plot( 'year', 'q4', data=gmp_q4,label = 'Goat Meat', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Goat Meat Production Q4')
ax = plt.gca()
ax.set_ylim(0,18000,2000)
plt.show()
plt.figure(figsize=(8,4))
plt.plot( 'year', 'q1', data=gmp_q1,label = 'Quarter 1', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q2', data=gmp_q2,label = 'Quarter 2', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q3', data=gmp_q3,label = 'Quarter 3', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q4', data=gmp_q4,label = 'Quarter 4', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Goat Meat Production')
plt.grid(True)
plt.show()
mp_q1 = cleaned[9:18][['year','q1']]
mp_q1 = mp_q1.reset_index(drop=True)
mp_q1
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q1', data=mp_q1, color="gold", alpha=0.45,)
plt.plot( 'year', 'q1', data=mp_q1,label = 'Mutton', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Mutton Production Q1')
ax = plt.gca()
ax.set_ylim(5000,40000,10000)
plt.show()
mp_q2 = cleaned[9:18][['year','q2']]
mp_q2 = mp_q2.reset_index(drop=True)
mp_q2
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q2', data=mp_q2, color="red", alpha=0.45,)
plt.plot( 'year', 'q2', data=mp_q2,label = 'Mutton', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Mutton Production Q2')
ax = plt.gca()
ax.set_ylim(15000,30000,5000)
plt.show()
mp_q3 = cleaned[9:18][['year','q3']]
mp_q3 = mp_q3.reset_index(drop=True)
mp_q3
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q3', data=mp_q3, color="darkviolet", alpha=0.45,)
plt.plot( 'year', 'q3', data=mp_q3,label = 'Mutton', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Mutton Production Q3')
ax = plt.gca()
ax.set_ylim(10000,40000,5000)
plt.show()
mp_q4 = cleaned[9:18][['year','q4']]
mp_q4 = mp_q4.reset_index(drop=True)
mp_q4
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q4', data=mp_q4, color="limegreen", alpha=0.45,)
plt.plot( 'year', 'q4', data=mp_q4,label = 'Mutton', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Mutton Production Q4')
ax = plt.gca()
ax.set_ylim(0,70000,10000)
plt.show()
plt.figure(figsize=(8,4))
plt.plot( 'year', 'q1', data=mp_q1,label = 'Quarter 1', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q2', data=mp_q2,label = 'Quarter 2', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q3', data=mp_q3,label = 'Quarter 3', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q4', data=mp_q4,label = 'Quarter 4', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Mutton Production')
plt.grid(True)
plt.show()
wb_q1 = cleaned[18:27][['year','q1']]
wb_q1 = wb_q1.reset_index(drop=True)
wb_q1
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q1', data=wb_q1, color="gold", alpha=0.45,)
plt.plot( 'year', 'q1', data=wb_q1,label = 'Water Buffalo', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Water Buffalo Production Q1')
plt.show()
wb_q2 = cleaned[18:27][['year','q2']]
wb_q2 = wb_q2.reset_index(drop=True)
wb_q2
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q2', data=wb_q2, color="red", alpha=0.45,)
plt.plot( 'year', 'q2', data=wb_q2,label = 'Water Buffalo', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Water Buffalo Production Q2')
plt.show()
wb_q3 = cleaned[18:27][['year','q3']]
wb_q3 = wb_q3.reset_index(drop=True)
wb_q3
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q3', data=wb_q3, color="darkviolet", alpha=0.45,)
plt.plot( 'year', 'q3', data=wb_q3, label = 'Water Buffalo', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Water Buffalo Production Q3')
plt.show()
wb_q4 = cleaned[18:27][['year','q4']]
wb_q4 = wb_q4.reset_index(drop=True)
wb_q4
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q4', data=wb_q4, color="limegreen", alpha=0.45,)
plt.plot( 'year', 'q4', data=wb_q4,label = 'Water Buffalo', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Water Buffalo Production Q4')
plt.show()
plt.figure(figsize=(8,4))
plt.plot( 'year', 'q1', data=wb_q1,label = 'Quarter 1', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q2', data=wb_q2,label = 'Quarter 2', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q3', data=wb_q3,label = 'Quarter 3', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q4', data=wb_q4,label = 'Quarter 4', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Water Buffalo Production')
plt.grid(True)
plt.show()
cc_q1 = cleaned[27:][['year','q1']]
cc_q1 = cc_q1.reset_index(drop=True)
cc_q1
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q1', data=cc_q1, color="gold", alpha=0.45,)
plt.plot( 'year', 'q1', data=cc_q1,label = 'Cattle-Calf', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Cattle-Calf Production Q1')
ax = plt.gca()
ax.set_ylim(100000,250000)
plt.show()
cc_q2 = cleaned[27:][['year','q2']]
cc_q2 = cc_q2.reset_index(drop=True)
cc_q2
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q2', data=cc_q2, color="red", alpha=0.45,)
plt.plot( 'year', 'q2', data=cc_q2,label = 'Cattle-Calf', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Cattle-Calf Production Q2')
ax = plt.gca()
ax.set_ylim(100000,250000)
plt.show()
cc_q3 = cleaned[27:][['year','q3']]
cc_q3 = cc_q3.reset_index(drop=True)
cc_q3
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q3', data=cc_q3, color="darkviolet", alpha=0.45,)
plt.plot( 'year', 'q3', data=cc_q3, label = 'Cattle-Calf', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Cattle-Calf Production Q3')
ax = plt.gca()
ax.set_ylim(10000,400000)
plt.show()
cc_q4 = cleaned[27:][['year','q4']]
cc_q4 = cc_q4.reset_index(drop=True)
cc_q4
plt.figure(figsize=(10,5))
plt.fill_between( 'year', 'q4', data=cc_q4, color="limegreen", alpha=0.45,)
plt.plot( 'year', 'q4', data=cc_q4,label = 'Cattle-Calf', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Cattle-Calf Production Q4')
ax = plt.gca()
ax.set_ylim(150000,400000)
plt.show()
plt.figure(figsize=(8,4))
plt.plot( 'year', 'q1', data=cc_q1,label = 'Quarter 1', marker='.', markersize=12, color='gold', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q2', data=cc_q2,label = 'Quarter 2', marker='.', markersize=12, color='red', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q3', data=cc_q3,label = 'Quarter 3', marker='.', markersize=12, color='darkviolet', linewidth=3,alpha = 0.5,)
plt.plot( 'year', 'q4', data=cc_q4,label = 'Quarter 4', marker='.', markersize=12, color='limegreen', linewidth=3,alpha = 0.5,)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years')              # label = name of label
plt.ylabel('Production (Tons)')
plt.title('Cattle-Calf Production')
plt.grid(True)
plt.show()
#Cattle-Calf total products
cc_total = cleaned.loc[27:34,"year":"q4"]
cc_total["total_cc"] = cc_total['q1'] + cc_total['q2'] + cc_total['q3'] + cc_total['q4']
cc_total = cc_total[['total_cc']]
cc_total = cc_total.reset_index(drop=True)
#cc_total
#Goat Meat total products
gmp_total = cleaned.loc[0:7,"year":"q4"]
gmp_total["total_gmp"] = gmp_total['q1'] + gmp_total['q2'] + gmp_total['q3'] + gmp_total['q4']
gmp_total = gmp_total[['year', 'total_gmp']]
#gmp_total
#Mutton total products
mp_total = cleaned.loc[9:16,"year":"q4"]
mp_total["total_mp"] = mp_total['q1'] + mp_total['q2'] + mp_total['q3'] + mp_total['q4']
mp_total = mp_total[['total_mp']]
mp_total = mp_total.reset_index(drop=True)
#mp_total
#Water Buffalo total products
wb_total = cleaned.loc[18:25,"year":"q4"]
wb_total["total_wb"] = wb_total['q1'] + wb_total['q2'] + wb_total['q3'] + wb_total['q4']
wb_total = wb_total[['total_wb']]
wb_total = wb_total.reset_index(drop=True)
#wb_total
#Cattle-Calf total products
cc_total = cleaned.loc[27:34,"year":"q4"]
cc_total["total_cc"] = cc_total['q1'] + cc_total['q2'] + cc_total['q3'] + cc_total['q4']
cc_total = cc_total[['total_cc']]
cc_total = cc_total.reset_index(drop=True)
#cc_total

#concat
conc = pd.concat([gmp_total,mp_total,wb_total,cc_total], axis = 1)
conc
f, (axarr) = plt.subplots(4, 1,figsize=(6,8), )
#goat meat
axarr[0].plot('year', 'total_gmp', data=conc, label = 'Goat Meat', marker='.', markersize=12, color='gold', linewidth=3)
axarr[0].set_ylim(10000,50000,10000)
axarr[0].legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)              
#mutton
axarr[1].plot('year', 'total_mp', data=conc,label ='Mutton', marker='.', markersize=12, color='red', linewidth=3)
axarr[1].set_ylim(80000,150000,30000)
axarr[1].legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
#water buffalo
axarr[2].plot('year', 'total_wb', data=conc, label ='Water Buffalo', marker='.', markersize=12, color='darkviolet', linewidth=3)
axarr[2].set_ylim(0,4000,1000)
axarr[2].legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
#cattle-calf
axarr[3].plot('year', 'total_cc', data=conc, label ='Cattle-Calf', marker='.', markersize=12, color='limegreen', linewidth=3)
axarr[3].legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
axarr[3].set_ylim(500000,1200000,200000)
#make grids true
axarr[0].grid(True)
axarr[1].grid(True)
axarr[2].grid(True)
axarr[3].grid(True)
#This kernel will be uploaded after TSI published the 2018 Quarter 4 data.