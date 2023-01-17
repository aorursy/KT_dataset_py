import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
kb_df = pd.read_csv('../input/data.csv')
kb_df.head(2)
kb_df.info()
df_main = kb_df[kb_df['shot_made_flag']>=0]
df_main.head(2)
df_missed = df_main[df_main.shot_made_flag==0].season.value_counts().sort_index()
df_success = df_main[df_main.shot_made_flag==1].season.value_counts().sort_index()
df_shot = pd.concat([df_success,df_missed],axis=1)
df_shot.columns=['Success','Missed']


fig = plt.figure(figsize=(17,5))
df_shot.plot(ax=fig.gca(), kind='bar',stacked=True,rot=1,color=['#FDB927','#552582'])
plt.xlabel('Season')
plt.legend(fontsize=18)
plt.figure(figsize=(15,10))
plt.scatter(df_main[df_main.shot_made_flag==0].loc_x,df_main[df_main.shot_made_flag==0].loc_y,color='#552582',label='Missed',alpha=0.5)
plt.scatter(df_main[df_main.shot_made_flag==1].loc_x,df_main[df_main.shot_made_flag==1].loc_y,color='#FDB927',label='Success',alpha=0.5)

plt.ylim(-10,300)
plt.legend(fontsize=18,loc='upper left')
loc_x_range = np.arange(-300,300,10)
binned = pd.cut(df_main.loc_x,loc_x_range)
binned_df = pd.DataFrame(binned)
binned_df.columns = ['loc_x_bin']
df_main = pd.concat([df_main,binned_df],axis=1)
loc_y_range = np.arange(-50,300,50)
binned_y = pd.cut(df_main.loc_y,loc_y_range)
binned_y_df = pd.DataFrame(binned_y)
binned_y_df.columns = ['loc_y_bin']
df_main = pd.concat([df_main,binned_y_df],axis=1)
loc_df =df_main.groupby(["loc_x_bin", "loc_y_bin"]).aggregate({'loc_x':np.mean,
                                                      'loc_y':np.mean,
                                                      'shot_made_flag':['count',np.sum]}).reset_index()
loc_df.columns = [' '.join(col).strip() for col in loc_df.columns.values]
loc_df.rename(columns={'shot_made_flag count':'total_shot','shot_made_flag sum':'Success','loc_x mean':'avg_x_loc',
                       'loc_y mean':'avg_y_loc'},inplace=True)
#loc_df.columns=["loc_x_bin", "loc_y_bin",'total_shot','Success','avg_x_loc','avg_y_loc']
loc_df['Success_Rate']= loc_df.Success / loc_df.total_shot
loc_df = loc_df[(loc_df.Success>50) &(loc_df.Success_Rate>0.4)]
loc_df.sort_values('Success_Rate',ascending=False).head(2)

cm = plt.cm.get_cmap('Reds') 
fig, ax = plt.subplots(figsize=(15,10))

bubble_size = (loc_df.Success_Rate)*5000

sc = ax.scatter(loc_df.avg_x_loc,loc_df.avg_y_loc,s=bubble_size,linewidths=2, edgecolor='w',c=bubble_size,cmap=cm)
sc = ax.scatter(df_main[df_main.shot_made_flag==0].loc_x,df_main[df_main.shot_made_flag==0].loc_y,color='#552582',
                label='Missed',alpha=0.05)
sc = ax.scatter(df_main[df_main.shot_made_flag==1].loc_x,df_main[df_main.shot_made_flag==1].loc_y,color='#FDB927',
                label='Success',alpha=0.05)
ax.grid()

# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.yaxis.set_major_formatter(plt.NullFormatter())
plt.ylim(-10,300)
plt.xlim(-300,300)
plt.title('Area with success rate above 40%')
df_missed_zone = df_main[df_main.shot_made_flag==0].shot_zone_area.value_counts().sort_index()
df_success_zone = df_main[df_main.shot_made_flag==1].shot_zone_area.value_counts().sort_index()
df_shot_zone = pd.concat([df_success_zone,df_missed_zone],axis=1)
df_shot_zone.columns=['Success','Missed']

fig = plt.figure(figsize=(10,5))
df_shot_zone.sort_values('Success',ascending=True).plot(ax=fig.gca(), kind='barh',stacked=True,rot=1,color=['#FDB927','#552582'])
plt.ylabel('shot_zone_area')
plt.legend(fontsize=18,loc='lower right')
df_missed_opponent = df_main[df_main.shot_made_flag==0].opponent.value_counts().sort_index()
df_success_opponent = df_main[df_main.shot_made_flag==1].opponent.value_counts().sort_index()
df_shot_opponent = pd.concat([df_success_opponent,df_missed_opponent],axis=1)
df_shot_opponent.columns=['Success','Missed']

fig = plt.figure(figsize=(10,10))
df_shot_opponent.sort_values('Success',ascending=True).plot(ax=fig.gca(), kind='barh',stacked=True,rot=1,color=['#FDB927','#552582'])
plt.ylabel('Opponent')
plt.legend(fontsize=18,loc='lower right')


plt.figure(figsize=(15,5))
sns.distplot(df_main[df_main['shot_made_flag']==1].shot_distance,bins=50,color='#FDB927',kde=True,label='Success')
sns.distplot(df_main.shot_distance,bins=50,color='#552582',kde=True,label='Missed')
plt.legend(fontsize=18)
period_df = df_main.groupby(['period','minutes_remaining']).aggregate({'shot_made_flag':['count',np.sum]}).reset_index()
period_df.columns = [' '.join(col).strip() for col in period_df.columns.values]
period_df.rename(columns={'shot_made_flag count':'Total_shots', 'shot_made_flag sum':'Success'},inplace=True)
period_df['Success_Rate'] = period_df.Success/period_df.Total_shots
period_df = period_df[period_df.Success>20]
period_df.drop(['Total_shots','Success'],axis=1,inplace=True)
period_df.head()
plt.figure(figsize=(20,10))
sns.heatmap(period_df.pivot('period','minutes_remaining','Success_Rate').fillna(value=0),
            annot=True,linewidth=0.5,vmax=1,cmap='Blues')
plt.title("Did Kobe perform better during clutch time? \n Kobe's average field goal rate is 0.447")
