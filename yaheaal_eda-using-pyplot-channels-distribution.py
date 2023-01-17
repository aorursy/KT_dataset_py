import pandas as pd

import numpy as np

import os

import cv2



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



from tqdm.notebook import tqdm_notebook

import warnings

warnings.filterwarnings('ignore')





path = '../input/panda-resized-train-data-512x512'

read_path = '../input/prostate-cancer-grade-assessment'
def read_img(path):

    img = cv2.imread(path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb
train_data = pd.read_csv(os.path.join(read_path,'train.csv'))
print(train_data.shape)

train_data.head()
SIZE = 512

N = train_data.shape[0]
X_train = np.empty((N, SIZE, SIZE, 3), dtype=np.uint8)



for i,img_id in enumerate(tqdm_notebook(train_data.image_id)):

    load_path = os.path.join(path,'train_images/train_images',train_data.image_id[i]+'.png')

    X_train[i,:,:,:] = read_img(load_path)



y_train = pd.get_dummies(train_data['isup_grade'])
fig = px.imshow(X_train[0]) # ISUP 0

fig
fig = px.imshow(X_train[6]) # ISUP 1

fig
fig = px.imshow(X_train[28]) # ISUP 2

fig
fig = px.imshow(X_train[15]) # ISUP 3

fig
fig = px.imshow(X_train[2]) # ISUP 4

fig
fig = px.imshow(X_train[22]) # ISUP 5

fig
train_karolinska = train_data[train_data['data_provider'] == 'karolinska']['data_provider']

train_radboud = train_data[train_data['data_provider'] == 'radboud']['data_provider']
print('ratio of karolinska in train_data :', round(len(train_karolinska)/len(train_data),3))

print('ratio of radboud in train_data :', round(len(train_radboud)/len(train_data),3))
isup_grade_count = train_data.isup_grade.value_counts().reset_index()

isup_grade_count.columns = ["isup_grade", "count"]





fig = make_subplots(1,2, specs=[[{"type": "bar"}, {"type": "pie"}]])



colors=px.colors.sequential.Plasma[:6]



fig.add_trace(go.Bar(

        x=isup_grade_count["isup_grade"].values, 

        y=isup_grade_count["count"].values,

        marker=dict(color=colors)

          

), row=1, col=1)



fig.add_trace(go.Pie(

        labels = isup_grade_count["isup_grade"].values,

        values = isup_grade_count["count"].values,

        marker=dict(colors=colors)

), row=1, col=2)



fig.update_layout(title="Isup_grade - Count plots")

fig.show()
karolinska = train_data.groupby(["data_provider", "isup_grade"])["data_provider"].count().loc["karolinska"].reset_index()

radboud = train_data.groupby(["data_provider", "isup_grade"])["data_provider"].count().loc["radboud"].reset_index()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=karolinska.isup_grade,

    y=karolinska.data_provider,

    name='karolinska',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=radboud.isup_grade,

    y=radboud.data_provider,

    name='rodboud',

    marker_color='lightsalmon'

))



fig.update_layout(title="targets count based on data provider")

fig.show()
gleason_score_count = train_data.gleason_score.value_counts().reset_index()

gleason_score_count.columns = ["gleason_score", "count"]





fig = make_subplots(1,2, specs=[[{"type": "bar"}, {"type": "pie"}]])



colors=px.colors.sequential.Plotly3



fig.add_trace(go.Bar(

        x=gleason_score_count["gleason_score"].values, 

        y=gleason_score_count["count"].values,

        marker=dict(color=colors)

          

), row=1, col=1)



fig.add_trace(go.Pie(

        labels = gleason_score_count["gleason_score"].values,

        values = gleason_score_count["count"].values,

        marker=dict(colors=colors)

), row=1, col=2)



fig.update_layout(title="Gleason_score - Count plots")

fig.show()
karolinska = train_data.groupby(["data_provider", "gleason_score"])["data_provider"].count().loc["karolinska"].reset_index()

radboud = train_data.groupby(["data_provider", "gleason_score"])["data_provider"].count().loc["radboud"].reset_index()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=karolinska.gleason_score,

    y=karolinska.data_provider,

    name='karolinska',

    marker_color=px.colors.sequential.Blackbody[1]

))

fig.add_trace(go.Bar(

    x=radboud.gleason_score,

    y=radboud.data_provider,

    name='rodboud',

    marker_color=px.colors.sequential.Blackbody[2]

))



fig.update_layout(title="gleason_score count based on data provider")

fig.show()
red_values = [np.mean(X_train[idx][:, :, 0]) for idx in range(len(X_train))]

green_values = [np.mean(X_train[idx][:, :, 1]) for idx in range(len(X_train))]

blue_values = [np.mean(X_train[idx][:, :, 2]) for idx in range(len(X_train))]

values = [np.mean(X_train[idx]) for idx in range(len(X_train))]
fig = ff.create_distplot([red_values, green_values, blue_values,values],

                         group_labels=['red','green','blue','values'],

                         colors=["red", "green", "blue", "purple"])

fig.update_layout(title_text="Distribution of channels based on each channel", template="simple_white")

fig
isup_0_values = list(np.array(values)[[y_train[y_train[0]==1].index.values]])

isup_1_values = list(np.array(values)[[y_train[y_train[1]==1].index.values]])

isup_2_values = list(np.array(values)[[y_train[y_train[2]==1].index.values]])

isup_3_values = list(np.array(values)[[y_train[y_train[3]==1].index.values]])

isup_4_values = list(np.array(values)[[y_train[y_train[4]==1].index.values]])

isup_5_values = list(np.array(values)[[y_train[y_train[5]==1].index.values]])
isup_0_red_values = list(np.array(red_values)[[y_train[y_train[0]==1].index.values]])

isup_1_red_values = list(np.array(red_values)[[y_train[y_train[1]==1].index.values]])

isup_2_red_values = list(np.array(red_values)[[y_train[y_train[2]==1].index.values]])

isup_3_red_values = list(np.array(red_values)[[y_train[y_train[3]==1].index.values]])

isup_4_red_values = list(np.array(red_values)[[y_train[y_train[4]==1].index.values]])

isup_5_red_values = list(np.array(red_values)[[y_train[y_train[5]==1].index.values]])
fig = ff.create_distplot([isup_0_red_values, isup_1_red_values, isup_2_red_values,isup_3_red_values, isup_4_red_values, isup_5_red_values],

                         group_labels=['0','1','2','3','4','5'],

                         colors=["red", "green", "blue", "goldenrod", "magenta", "black"])

fig.update_layout(title_text="Distribution of red channel based on ISUP", template="simple_white")

fig
isup_0_green_values = list(np.array(green_values)[[y_train[y_train[0]==1].index.values]])

isup_1_green_values = list(np.array(green_values)[[y_train[y_train[1]==1].index.values]])

isup_2_green_values = list(np.array(green_values)[[y_train[y_train[2]==1].index.values]])

isup_3_green_values = list(np.array(green_values)[[y_train[y_train[3]==1].index.values]])

isup_4_green_values = list(np.array(green_values)[[y_train[y_train[4]==1].index.values]])

isup_5_green_values = list(np.array(green_values)[[y_train[y_train[5]==1].index.values]])
fig = ff.create_distplot([isup_0_green_values, isup_1_green_values, isup_2_green_values,isup_3_green_values, isup_4_green_values, isup_5_green_values],

                         group_labels=['0','1','2','3','4','5'],

                         colors=["red", "green", "blue", "goldenrod", "magenta", "black"])

fig.update_layout(title_text="Distribution of green channel values based on ISUP", template="simple_white")

fig
isup_0_blues_values = list(np.array(blue_values)[[y_train[y_train[0]==1].index.values]])

isup_1_blues_values = list(np.array(blue_values)[[y_train[y_train[1]==1].index.values]])

isup_2_blues_values = list(np.array(blue_values)[[y_train[y_train[2]==1].index.values]])

isup_3_blues_values = list(np.array(blue_values)[[y_train[y_train[3]==1].index.values]])

isup_4_blues_values = list(np.array(blue_values)[[y_train[y_train[4]==1].index.values]])

isup_5_blues_values = list(np.array(blue_values)[[y_train[y_train[5]==1].index.values]])
fig = ff.create_distplot([isup_0_blues_values, isup_1_blues_values, isup_2_blues_values,isup_3_blues_values, isup_4_blues_values, isup_5_blues_values],

                         group_labels=['0','1','2','3','4','5'],

                         colors=["red", "green", "blue", "goldenrod", "magenta", "black"])

fig.update_layout(title_text="Distribution of blue channel based on ISUP", template="simple_white")

fig
isup_0_red_values = list(np.array(red_values)[[y_train[y_train[0]==1].index.values]])

isup_0_green_values = list(np.array(green_values)[[y_train[y_train[0]==1].index.values]])

isup_0_blue_values = list(np.array(blue_values)[[y_train[y_train[0]==1].index.values]])
fig = ff.create_distplot([isup_0_red_values, isup_0_green_values, isup_0_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 0)", template="simple_white")

fig
isup_1_red_values = list(np.array(red_values)[[y_train[y_train[1]==1].index.values]])

isup_1_green_values = list(np.array(green_values)[[y_train[y_train[1]==1].index.values]])

isup_1_blue_values = list(np.array(blue_values)[[y_train[y_train[1]==1].index.values]])
fig = ff.create_distplot([isup_1_red_values, isup_1_green_values, isup_1_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 1)", template="simple_white")

fig
isup_2_red_values = list(np.array(red_values)[[y_train[y_train[2]==1].index.values]])

isup_2_green_values = list(np.array(green_values)[[y_train[y_train[2]==1].index.values]])

isup_2_blue_values = list(np.array(blue_values)[[y_train[y_train[2]==1].index.values]])
fig = ff.create_distplot([isup_2_red_values, isup_2_green_values, isup_2_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 2)", template="simple_white")

fig
isup_3_red_values = list(np.array(red_values)[[y_train[y_train[3]==1].index.values]])

isup_3_green_values = list(np.array(green_values)[[y_train[y_train[3]==1].index.values]])

isup_3_blue_values = list(np.array(blue_values)[[y_train[y_train[3]==1].index.values]])
fig = ff.create_distplot([isup_3_red_values, isup_3_green_values, isup_3_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 3)", template="simple_white")

fig
isup_4_red_values = list(np.array(red_values)[[y_train[y_train[4]==1].index.values]])

isup_4_green_values = list(np.array(green_values)[[y_train[y_train[4]==1].index.values]])

isup_4_blue_values = list(np.array(blue_values)[[y_train[y_train[4]==1].index.values]])
fig = ff.create_distplot([isup_4_red_values, isup_4_green_values, isup_4_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 4)", template="simple_white")

fig
isup_5_red_values = list(np.array(red_values)[[y_train[y_train[5]==1].index.values]])

isup_5_green_values = list(np.array(green_values)[[y_train[y_train[5]==1].index.values]])

isup_5_blue_values = list(np.array(blue_values)[[y_train[y_train[5]==1].index.values]])
fig = ff.create_distplot([isup_5_red_values, isup_5_green_values, isup_5_blue_values],

                         group_labels=['R','G','B'],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of channel values (ISUP 5)", template="simple_white")

fig