import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/digidb/DigiDB_digimonlist.csv', index_col = 'Number')
data
data.info()
data.describe()
stage_cnt = data.Stage.value_counts()
fig = px.bar(stage_cnt, x=stage_cnt.index, y=stage_cnt, labels={'y':'count', 'index':'Stage'}, 
             color_continuous_scale='Civids')
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.show()
type_cnt = data.Type.value_counts()
fig = px.pie(type_cnt, values=type_cnt, names=type_cnt.index)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
att_cnt = data.Attribute.value_counts()
fig = px.pie(att_cnt, values=att_cnt, names=att_cnt.index)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
data['cnt'] = 1
fig = px.sunburst(data, path=['Type', 'Attribute'], values='cnt', color='Type')
fig.update_layout(autosize=False, width=700, height=700)
fig.show()
cm = sns.light_palette("green", as_cmap=True)
table = data[data.Stage == 'Mega'][['Type', 'Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']]
table = table.groupby(["Type"]).mean()
table.style.background_gradient(cmap=cm)
table_2 = data[data.Stage == 'Mega'][['Attribute', 'Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']]
table_2 = table_2.groupby(["Attribute"]).mean()
table_2.style.background_gradient(cmap=cm)
