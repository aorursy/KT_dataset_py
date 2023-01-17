import pandas as pd
import numpy as np
from matplotlib import dates as dates, patches as mpatches, pyplot as plt
import seaborn as sns
from plotly import plotly as py, graph_objs  as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
df_Ins = pd.read_csv('../input/inspections.csv')
df_Vio = pd.read_csv('../input/violations.csv')
df_Ins['facility_name'] = df_Ins['facility_name'].str.upper() 
df_Ins.activity_date = pd.to_datetime(df_Ins.activity_date)
df_merged = pd.merge(df_Ins,df_Vio, on='serial_number', how='left')
df_merged['activity_date'] = pd.to_datetime(df_merged['activity_date'])
df_merged['year'] = df_merged['activity_date'].dt.year
df_merged['month'] = df_merged['activity_date'].dt.month
df_merged.sort_values(by = 'activity_date', ascending = True, inplace = True)
df_merged["time"] = df_merged['activity_date'].apply(lambda x:x.strftime('%Y%m')) 
df_selected = df_merged[['facility_address', 'facility_name', 'facility_city', 'grade', 'score', 'serial_number', 'points',
          'violation_description', 'violation_status','time', 'activity_date', 'year']]
df_selected['points'].fillna(0, inplace = True) 
df_selected['violation_description'].fillna('No violation', inplace = True)
df_selected['violation_status'].fillna('No violation', inplace = True)
df_visits = df_selected.groupby(by = ['facility_name', 'activity_date']).first()
df_first_visit = df_visits.groupby('facility_name').head(1)
df_first_visit.reset_index(level = 0, inplace = True)
df_first_visit['score'] = pd.to_numeric(df_first_visit.score)
df_first_visit = df_first_visit.sort_values(by = 'score', ascending = True )
df_worst = df_first_visit.head(10)
fig, ax = plt.subplots(figsize = (15, 9))
sns.barplot(x = 'score', y = 'facility_name', data = df_worst, color = 'g')
ax.tick_params(axis='x',length = 10.0, labelsize = 18)
ax.tick_params(axis='y',length = 10,  labelsize = 18)
plt.title("Worst Violators Initial Scores", fontsize = 25, y =1.01)
plt.xlabel("Score", fontsize = 20, labelpad = 15)
plt.ylabel("Facility Name", fontsize = 20, labelpad = 15)
plt.xlim(60, 72)
plt.show()
init_notebook_mode(connected = True)
trace1 = go.Scatter(
    x = df_merged[df_merged.facility_name =='FORTUNE CORNER CHINESE FOOD'].activity_date,
    y = df_merged[df_merged.facility_name =='FORTUNE CORNER CHINESE FOOD'].score,
    mode = 'lines+markers',
    name = 'FORTUNE CORNER CHINESE FOOD'
)
trace2 = go.Scatter(
    x = df_merged[df_merged.facility_name =='SOL NIGHT CLUB'].activity_date,
    y = df_merged[df_merged.facility_name =='SOL NIGHT CLUB'].score,
    mode = 'lines+markers',
    name = 'SOL NIGHT CLUB'
)
trace3 = go.Scatter(
    x = df_merged[df_merged.facility_name =='THREE BEARS BAR-B-Q'].activity_date,
    y = df_merged[df_merged.facility_name =='THREE BEARS BAR-B-Q'].score,
    mode = 'lines+markers',
    name = 'THREE BEARS BAR-B-Q'
)
trace4 = go.Scatter(
    x = df_merged[df_merged.facility_name =='ANTOJITOS Y GARNACHERIA DONA ROSITA'].activity_date,
    y = df_merged[df_merged.facility_name =='ANTOJITOS Y GARNACHERIA DONA ROSITA'].score,
    mode = 'lines+markers',
    name = 'ANTOJITOS Y GARNACHERIA DONA ROSITA'
)
trace5 = go.Scatter(
    x = df_merged[df_merged.facility_name =='EASTERN EXPRESS CAFE'].activity_date,
    y = df_merged[df_merged.facility_name =='EASTERN EXPRESS CAFE'].score,
    mode = 'lines+markers',
    name = 'EASTERN EXPRESS CAFE'
)
data = [trace1,trace2,trace3,trace4,trace5]
layout = go.Layout(title = 'Worst Violators Over Time',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Score'),
              width = 990,
              height = 500,
              legend = dict(x =.35, y =.1)
              )
fig = dict(data = data, layout = layout)
iplot(fig, filename = 'resturant')
df_worst5 = df_merged[(df_merged.facility_name == 'FORTUNE CORNER CHINESE FOOD') | (df_merged.facility_name == 'SOL NIGHT CLUB') | 
    (df_merged.facility_name == 'ANTOJITOS Y GARNACHERIA DONA ROSITA') | (df_merged.facility_name == 'EASTERN EXPRESS CAFE') |
    (df_merged.facility_name == 'THREE BEARS BAR-B-Q')].groupby(['facility_name', 'violation_description']).agg({
    'points':'mean',
    'score':'mean',
    'facility_address': 'count'
}).reset_index(level=['facility_name', 'violation_description'])

df_worst5.columns = ['Restaurant', 'Violation Description', 'Avg_Points', 'Avg_Scores', 'Count']
df_worst5['Losing Point'] = df_worst5.Avg_Points * df_worst5.Count
df_worst5['Violation Description'] = df_worst5['Violation Description'].apply(lambda x: x[6:])
df_frequency = df_worst5[['Restaurant', 'Violation Description', 'Count']]
plt.figure(figsize = (16, 12))
sns.set(font_scale = 1.2)
sns.heatmap(df_frequency.pivot(index = 'Violation Description', columns = 'Restaurant', values = 'Count'),  
            annot = True, fmt = "g", cmap = 'Blues')
plt.title('Most Common Violations in Worst 5 (frequency)', fontsize = 36,  y = 1.05)
plt.xlabel('Restaurant', fontsize = 32)
plt.ylabel('Violation Description', fontsize = 32)
plt.xticks(fontsize = 18, rotation = '80')
plt.yticks(fontsize = 14)
plt.show()
df_losing_point = df_worst5[['Restaurant', 'Violation Description', 'Losing Point']]
plt.figure(figsize = (16,12))
sns.set(font_scale = 1.2)
sns.heatmap(df_losing_point.pivot(index = 'Violation Description', columns = 'Restaurant', values = 'Losing Point'),  
            annot = True, fmt = "g", cmap = 'YlGnBu')
plt.title('Most Common Violations in Worst 5 (Losing Points)', fontsize = 36,  y = 1.05)
plt.xlabel('Restaurant', fontsize = 32)
plt.ylabel('Violation Description', fontsize= 32)
plt.xticks(fontsize = 18, rotation='80')
plt.yticks(fontsize = 14)
plt.show()
#Filter more rows and focus on more frequency & losing points base on 50% metrics
df_median = df_worst5.groupby('Violation Description').sum().reset_index()
df_median.describe().iloc[5, :]
df_frequency_Median = df_worst5[df_worst5['Violation Description'].isin(df_median[df_median.Count > 5][
    'Violation Description'])][['Restaurant', 'Violation Description', 'Count']]
df_Compared_50 = df_worst5[df_worst5['Violation Description'].isin(df_median[df_median.Count > 5][
    'Violation Description']) | df_worst5['Violation Description'].isin(df_median[df_median['Losing Point'] > 8][
    'Violation Description'])]
sns.set(font_scale = 1.8)
plt.figure(figsize = (16, 10))
sns.heatmap(df_Compared_50[['Restaurant', 'Violation Description', 'Count']].pivot(index = 'Violation Description',
            columns = 'Restaurant', values = 'Count'), annot = True, fmt = "g", cmap = 'Blues', 
            xticklabels =
            ['ANTOJITOS Y \nGARNACHERIA \nDONA \nROSITA',
             'EASTERN\n EXPRESS \nCAFE',
             'FORTUNE \nCORNER \nCHINESE \nFOOD'
             ,'SOL \nNIGHT \nCLUB',
             'THREE \nBEARS \nBAR-B-Q'], 
            yticklabels = 
            ['Food safety certification',
             'Water available',
             'Handwashing facilities',
             'Ventilation and lighting',
             'Equipment / Utensils',
             'Floors, walls and ceilings',
             'Food contact surfaces',
             'Food in good condition',
             'Food storage containers',
             'Food separated and protected',
             'No insects or animals',
             'Nonfood-contact surfaces',
             'Plumbing',
             'Proper holding temperatures',
             'Sewage & wastewater disposed',
             'Toilet facilities',
             'Toxic substances'
             'Warewashing facilities'
            ])
plt.title('Frequency of Violations by Resturant', fontsize = 36,  y = 1.05)
plt.xlabel('Restaurant', fontsize = 32, y = 0.85)
plt.ylabel('Violation Description', fontsize= 32)

plt.legend(labels = ['1','2','3','4','5'], loc = 2)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 28)
plt.show()
sns.set(font_scale = 1.8)
plt.figure(figsize = (16, 10))
sns.heatmap(df_Compared_50[['Restaurant', 'Violation Description', 'Losing Point']].pivot(index = 'Violation Description',
            columns = 'Restaurant', values = 'Losing Point'), annot = True, fmt = "g", cmap = 'YlGnBu', 
            xticklabels =
            ['ANTOJITOS Y \nGARNACHERIA \nDONA \nROSITA',
             'EASTERN\n EXPRESS \nCAFE',
             'FORTUNE \nCORNER \nCHINESE \nFOOD'
             ,'SOL \nNIGHT \nCLUB',
             'THREE \nBEARS \nBAR-B-Q'], 
            yticklabels = 
            ['Food safety certification',
             'Water available',
             'Handwashing facilities',
             'Ventilation and lighting',
             'Equipment / Utensils',
             'Floors, walls and ceilings',
             'Food contact surfaces',
             'Food in good condition',
             'Food storage containers',
             'Food separated and protected',
             'No insects or animals',
             'Nonfood-contact surfaces',
             'Plumbing',
             'Proper holding temperatures',
             'Sewage & wastewater disposed',
             'Toilet facilities',
             'Toxic substances'
             'Warewashing facilities'
            ])
plt.title('Points Lost by Violation and Resturant', fontsize = 36,  y = 1.05)
plt.xlabel('Restaurant', fontsize = 32, y = 0.85)
plt.ylabel('Violation Description', fontsize= 32)

plt.legend(labels = ['1','2','3','4','5'], loc = 2)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 28)
plt.show()
df_Agg_Violation = df_merged.groupby(['violation_description']).agg({
    'points':'mean',
    'facility_address':'count'
})
df_Agg_Violation['Losing_Points'] = df_Agg_Violation.points * df_Agg_Violation.facility_address
df_Agg_Violation.columns = ['Avg_points', 'Count', 'Losing_Points']
df_Agg_Violation = df_Agg_Violation[(df_Agg_Violation.Avg_points > 0) & (
    df_Agg_Violation.Losing_Points.nlargest(10))].reset_index()
df_Agg_Violation = df_Agg_Violation.sort_values(by = 'Losing_Points', ascending = False)
df_Agg_Violation['violation_description'] = df_Agg_Violation['violation_description'].apply(lambda x: x[6:])
fig, ax = plt.subplots(figsize=(15, 9))
sns.barplot(x = 'Losing_Points', y = 'violation_description', data = df_Agg_Violation, ax = ax)
plt.title('Total Points Lost per Violation',fontsize = 36, y = 1.05 )
plt.xlabel('Points Lost',fontsize = 30)
plt.ylabel('Violation Description', fontsize = 30)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 26)
plt.show();
df_trend = df_merged.groupby(['time']).agg({
    'score':'mean',
    'points' : 'mean'
}).dropna().sort_index().reset_index()
df_trend['mean'] = df_trend.score.mean()
df_trend['time'] = df_trend['time'].apply(lambda x: x[4:]) + '/' + df_trend['time'].apply(lambda x: x[2:4])
fig, ax = plt.subplots(figsize = (15, 9))
plt.style.use('ggplot')
plt.plot( 'score', data = df_trend, marker = 'o', markerfacecolor = 'blue', markersize = 10, 
         color = 'blue', linewidth=2.5)
plt.plot( 'time', 'mean', data = df_trend, marker = '+', markerfacecolor = 'green', markersize = 10, 
         color = 'green', linewidth=2.5)
plt.title('Trend of Violation',fontsize = 24)
plt.xlabel('2015/07 - 2017/12 (Month)',fontsize = 16)
plt.ylabel('Score', fontsize=20)
plt.xticks(df_trend.index, df_trend.time, fontsize=14, rotation = '60')
plt.yticks(fontsize = 14)
plt.show();
