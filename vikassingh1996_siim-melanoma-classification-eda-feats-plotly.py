'''Import basic modules.'''

import pandas as pd

import numpy as np

import os

import pydicom as dcm

import cv2

import time



'''Customize visualization

Seaborn and matplotlib visualization.'''

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline

import folium 

from IPython.core.display import HTML

import urllib.request

from PIL import Image

import imageio







'''Plotly visualization .'''

import plotly.express as px

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook







'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))



import warnings

warnings.filterwarnings('ignore')
incidence_rates = pd.read_csv('../input/melanoma-skin-cancer-dataset/annual_incidence_rates.csv')

death_rates = pd.read_csv('../input/melanoma-skin-cancer-dataset/annual_death_rates.csv')

age_incidence_rates = pd.read_csv('../input/melanoma-skin-cancer-dataset/age_specific_incidence_rate.csv')

age_death_rates = pd.read_csv('../input/melanoma-skin-cancer-dataset/age_specific_death_rate.csv')

state_death_rates = pd.read_csv('../input/melanoma-skin-cancer-dataset/state_death_rates.csv')

code = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

state_death_rates['code'] = code
trace1 = go.Scatter(

                x=incidence_rates['Year of Diagnosis'],

                y=incidence_rates['All Races,Males'],

                name="All Races,Males",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='dodgerblue')



trace2 = go.Scatter(

                x=incidence_rates['Year of Diagnosis'],

                y=incidence_rates['All Races,Females'],

                name="All Races,Females",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='fuchsia')



layout = go.Layout(template = 'plotly_white', width=700, height=500, title_text = '<b>Incidencea Rates by Year, All Race and Sex </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2015',

            y0=6.5,

            x1='2015',

            y1=35,

            line=dict(

                color="black",

                width=1,

                dash="dashdot"

            )))

fig.add_annotation( # add a text callout with arrow

    text=" Slightly Decreasing", x='2015', y=23, arrowhead=1, showarrow=True

)

fig.show()



trace1 = go.Scatter(

                x=incidence_rates['Year of Diagnosis'],

                y=incidence_rates['Whites,Both Sexes'],

                name="Whites,Both Sexes",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='lightsalmon')



trace2 = go.Scatter(

                x=incidence_rates['Year of Diagnosis'],

                y=incidence_rates['Blacks,Both Sexes'],

                name="Blacks,Both Sexes",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='maroon')



layout = go.Layout(template = 'plotly_white', width=700, height=500, title_text = '<b>Incidencea Rates by Year, Between Race</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()
trace1 = go.Scatter(

                x=death_rates['Year of Death'],

                y=death_rates['All Races,Males'],

                name="All Races,Males",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='dodgerblue')



trace2 = go.Scatter(

                x=death_rates['Year of Death'],

                y=death_rates['All Races,Females'],

                name="All Races,Females",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='fuchsia')



layout = go.Layout(template = 'plotly_white', width=700, height=500, title_text = '<b>Death  Rates by Year, All Race and Sex </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2009',

            y0=1.3,

            x1='2009',

            y1=4.5,

            line=dict(

                color="black",

                width=1,

                dash="dashdot"

            )))

fig.add_annotation( # add a text callout with arrow

    text="Start Decreasing", x='2009', y=2.5, arrowhead=1, showarrow=True

)

fig.show()



trace1 = go.Scatter(

                x=death_rates['Year of Death'],

                y=death_rates['Whites,Both Sexes'],

                name="Whites,Both Sexes",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='lightsalmon')



trace2 = go.Scatter(

                x=death_rates['Year of Death'],

                y=death_rates['Blacks,Both Sexes'],

                name="Blacks,Both Sexes",

                marker=dict(size=3.5),

                mode='lines+markers',

                line_color='maroon')



layout = go.Layout(template = 'plotly_white', width=700, height=500, title_text = '<b>Death Rates by Year, Between Race</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()


trace1 = go.Pie(

                labels=age_incidence_rates['  Age at Diagnosis'],

                values=age_incidence_rates['All Races,  Both Sexes'],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=px.colors.sequential.RdBu, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=600, height=500,title_text = '<b>Age-Specific Incidence Rates By All Race and Sex ,  2013-2017<b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()



trace2 = go.Bar(

            x=age_incidence_rates['  Age at Diagnosis'], 

            y=age_incidence_rates['All Races,Males'],

            text=age_incidence_rates['All Races,Males'],

            name = 'All Races,Males',

            textposition='auto',

            marker_color='dodgerblue')

trace3 = go.Bar(

            x=age_incidence_rates['  Age at Diagnosis'], 

            y=age_incidence_rates['All Races,Females'],

            text=age_incidence_rates['All Races,Females'],

            name = 'All Races,Females',

            textposition='auto',

            marker_color='fuchsia')

layout = go.Layout(barmode='group', template = 'plotly_white',width=700, height=500, 

                  title_text = '<b>Age-Specific Incidence Rates Between Sex, 2013-2017<b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace2, trace3], layout = layout)

fig.show()
trace1 = go.Pie(

                labels=age_death_rates['  Age at Death'],

                values=age_death_rates['All Races,  Both Sexes'],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=px.colors.sequential.RdBu, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=600, height=500,title_text = '<b>Age-Specific Death Rates By All Race and Sex ,  2013-2017<b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()



trace2 = go.Bar(

            x=age_death_rates['  Age at Death'], 

            y=age_death_rates['All Races,Males'],

            text=age_death_rates['All Races,Males'],

            name = 'All Races,Males',

            textposition='auto',

            marker_color='dodgerblue')

trace3 = go.Bar(

            x=age_death_rates['  Age at Death'], 

            y=age_death_rates['All Races,Females'],

            text=age_death_rates['All Races,Females'],

            name = 'All Races,Females',

            textposition='auto',

            marker_color='fuchsia')

layout = go.Layout(barmode='group', template = 'plotly_white',width=650, height=500, 

                  title_text = '<b>Age-Specific Death Rates Between Sex, 2013-2017<b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace2, trace3], layout = layout)

fig.show()
fig = go.Figure(data=go.Choropleth(

    locations=state_death_rates['code'], 

    z = state_death_rates['Both Sex '], 

    locationmode = 'USA-states',

    colorscale = 'brbg',

    colorbar_title = "Death Rate",

    text=state_death_rates['State'],

))



fig.update_layout(width=650, height=600, 

                  title_text = '<b>Individual State Death Rates<b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'),

    geo = dict(

        scope='usa',

        projection=go.layout.geo.Projection(type = 'albers usa'), 

))



fig.show()



fig = px.bar(state_death_rates.sort_values(by='Both Sex ', ascending=False).head(10).sort_values('Both Sex ', ascending=True), 

             x='Both Sex ', y="State", 

             title='<b>Top 10 States by Death Rates<b>',

             text='Both Sex ', 

             orientation='h', 

             width=700, height=500)

fig.update_traces(marker_color='khaki', opacity=0.8, textposition='inside',)



fig.update_layout(template = 'plotly_white')

fig.show()
Female = np.array([-1.50,-1.21,-1.67,-1.35,-1.26,-1.97,-1.31,-1.69,-0.00,-1.48,-1.39,-0.77,-2.15,-1.45,

         -1.72,-1.88,-1.82,-1.87,-1.07,-1.93,-1.22,-1.73,-1.41,-1.60,-1.24,-1.59,-1.50,-1.52,-1.35,

         -1.84,-1.44,-1.44,-1.17,-1.54,-1.26,-1.71,-1.61,-1.73,-1.72,-1.62,-1.49,-1.51,-1.71,-1.19,

         -1.64,-1.67,-1.51,-1.74,-2.00,-1.46,-1.69])



data = [go.Bar(y=state_death_rates['State'],

               x=state_death_rates['Male'],

               orientation='h',

               name='Male',

               hoverinfo='x',

               marker=dict(color='dodgerblue')

               ),

        go.Bar(y=state_death_rates['State'],

               x=Female,

               orientation='h',

               name='Female',

               text=-1 * Female,

               hoverinfo='text',

               marker=dict(color='fuchsia')

               )]



layout = go.Layout(width=650, height=700,

                   template = 'plotly_white',

                   yaxis=go.layout.YAxis(title='State'),

                   xaxis=go.layout.XAxis(

                       range=[-6, 6],

                       tickvals=[-5.5,-4.5,-3.5, -2.5, -1.5, 0, 1.5, 2.5, 3.5, 4.5, 5.5],

                       ticktext=[5.5,4.5,3.5, 2.5, 1.5, 0, 1.5, 2.5, 3.5, 4.5, 5.5],

                       title='Death Rate'),

                    barmode='overlay',

                    bargap=0.1,

                    title_text = '<b>Pyramid Chart of States Death Rates Between Sex<b>',

                      font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))



py.iplot(dict(data=data, layout=layout))
IMAGE_PATH = "../input/siim-isic-melanoma-classification/"



train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')





#Training data

print('Training data shape: ', train_df.shape)

print(' ')

print('Test data shape: ', test_df.shape)



bold('**TRAINING DATA**')

display(train_df.head(5))

bold('**TEST DATA**')

display(test_df.head(5))
bold('**MISSING DATA AND DATA TYPES**')

print('===== Train Set =====')

print(train_df.info())

print('\n')

print('===== Train Set =====')

print(train_df.info())



bold('**TOTAL NUMBER OF IMAGES**')

print("Total images in Train set: ",train_df['image_name'].count())

print("Total images in Test set: ",test_df['image_name'].count())



bold('**UNIQUE IDs**')

print('Total nuber of patient ids in train set ', train_df['patient_id'].count())

print('Total nuber of patient ids test set ', test_df['patient_id'].count())

print('Unique ids in train set ',train_df['patient_id'].nunique())

print('Unique ids in test set ', test_df['patient_id'].nunique())



ids_train = train_df.patient_id.values

ids_test = test_df.patient_id.values

patient_overlap = list(set(ids_train).intersection(set(ids_test)))

print('Patient IDs in both the training and test sets ', len(patient_overlap))

target_count = train_df['target'].value_counts().reset_index()

target_count['percent']=np.round(train_df['target'].value_counts(normalize=True), 2)

target_count.rename(columns={'index': 'target', 'target':'count'}, inplace=True)



fig = px.bar(target_count, 

             x='target', y="count", 

             title='<b>Distribution of the Target (binarized version)<b>',

             text='percent', 

             orientation='v', 

             width=500, height=600)

fig.update_traces(opacity=0.8, marker=dict(color='yellowgreen',

                                  line=dict(width=2, color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
sex_count = train_df['sex'].value_counts().reset_index()

sex_count['percent']=np.round(train_df['sex'].value_counts(normalize=True).reset_index()['sex'], 2)

sex_count.rename(columns={'index': 'sex', 'sex':'count'}, inplace=True)



fig = px.bar(sex_count, 

             x='sex', y="count", 

             title='<b>Distribution of the gender<b>',

             text='percent', 

             orientation='v', 

             width=500, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(color='lightsalmon',

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
temp = train_df.groupby(['target','sex'])['target'].count().to_frame('count').reset_index()

fig = px.bar(temp, 

             x="target", y="count", 

             color='sex', 

             barmode='group',

             title='<b>Distribution of the gender by target<b>',

             text='count', 

             orientation='v', 

             width=500, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
fig = px.histogram(train_df, 

             x="age_approx", 

             nbins=30,

             barmode='group',

             title='<b>Distribution of the gender by target<b>',

             marginal="box",

             width=600, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(color='palegoldenrod',

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
temp = train_df[['age_approx', 'sex']].dropna()

fig = px.histogram(temp, 

             x="age_approx",

             color = 'sex',

             nbins=30,

             barmode='group',

             title='<b>Distribution of patients age by gender<b>',

             marginal="box",

             width=600, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
benign_count = train_df['benign_malignant'].value_counts().reset_index()

benign_count['percent']=np.round(train_df['benign_malignant'].value_counts(normalize=True).reset_index()['benign_malignant'], 2)

benign_count.rename(columns={'index': 'benign_malignant', 'benign_malignant':'count'}, inplace=True)



fig = px.bar(benign_count, 

             x='benign_malignant', y="count", 

             title='<b>Distribution of the benign malignant<b>',

             text='percent', 

             orientation='v', 

             width=500, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(color='moccasin',

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
temp = train_df[['age_approx', 'benign_malignant']].dropna()

fig = px.histogram(temp, 

             x="age_approx",

             color = 'benign_malignant',

             nbins=30,

             barmode='group',

             title='<b>Distribution of patients age by benign malignant<b>',

             marginal="box",

             width=600, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
anatom_count = train_df['anatom_site_general_challenge'].value_counts().reset_index()

anatom_count['percent']=np.round(train_df['anatom_site_general_challenge'].value_counts(normalize=True).reset_index()['anatom_site_general_challenge'], 2)

anatom_count.rename(columns={'index': 'anatom_site_general_challenge', 'anatom_site_general_challenge':'count'}, inplace=True)



fig = px.bar(anatom_count, 

             x='anatom_site_general_challenge', y="count", 

             title='<b>Distribution of the Location of imaged site<b>',

             text='percent', 

             orientation='v', 

             width=600, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(color='lightcoral',

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
temp = train_df.groupby(['anatom_site_general_challenge','sex'])['anatom_site_general_challenge'].count().to_frame('count').reset_index()

fig = px.bar(temp, 

             x="anatom_site_general_challenge", y="count", 

             color='sex', 

             barmode='group',

             title='<b>Distribution of the Location of imaged site by gender<b>',

             text='count', 

             orientation='v', 

             width=700, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
diagnosis_count = train_df['diagnosis'].value_counts().reset_index()

diagnosis_count['percent']=np.round(train_df['diagnosis'].value_counts(normalize=True).reset_index()['diagnosis'], 2)

diagnosis_count.rename(columns={'index': 'diagnosis', 'diagnosis':'count'}, inplace=True)



fig = px.bar(diagnosis_count, 

             x='diagnosis', y="count", 

             title='<b>Distribution of the diagnosis<b>',

             text='count', 

             orientation='v', 

             width=600, height=600,

            )

fig.update_traces(opacity=0.8, marker=dict(color='rebeccapurple',

                                  line=dict(width=2,

                                        color='DarkSlateGrey')))



fig.update_layout(template = 'plotly_white')

fig.show()
bold('**Let’s load an image and observe its various properties in general**')



images = train_df['image_name'].values

img_dir = IMAGE_PATH+'/jpeg/train'



img = imageio.imread(os.path.join(img_dir, (images+'.jpg')[1]))

plt.figure(figsize = (5,5))

plt.imshow(img)

plt.show()



bold('**Observe Basic Properties of Image**')

print('Type of the image : ' , type(img)) 

print('Shape of the image : {}'.format(img.shape)) 

print('Image Hight {}'.format(img.shape[0])) 

print('Image Width {}'.format(img.shape[1])) 

print('Dimension of Image {}'.format(img.ndim))



bold('**calculate the size of an RGB image**')

print('Image size {}'.format(img.size)) 

print('Maximum RGB value in this image {}'.format(img.max())) 

print('Minimum RGB value in this image {}'.format(img.min()))

print('Value of only R channel {}'.format(img[ 100, 50, 0])) 

print('Value of only G channel {}'.format(img[ 100, 50, 1])) 

print('Value of only B channel {}'.format(img[ 100, 50, 2]))
bold('**view of random images in single channel**')

random_images = [np.random.choice(images+'.jpg') for i in range(9)]

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = imageio.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img[ : , : , 0], cmap='gray')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout() 
# https://www.kaggle.com/gpreda/siim-isic-melanoma-classification-eda

def show_dicom_images(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['image_name']+'.dcm'

        imagePath = os.path.join(IMAGE_PATH,"train/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.gray) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f"ID: {data_row['image_name']}\nModality: {modality} Age: {age} Sex: {sex}\nDiagnosis: {data_row['diagnosis']}")

    plt.show()
bold('**Images with Malignant lesions**')

show_dicom_images(train_df[train_df['target']==1].sample(9))
bold('**Images with benign lesions**')

show_dicom_images(train_df[train_df['target']==0].sample(9))
benign = train_df[train_df['benign_malignant']=='benign']

malignant = train_df[train_df['benign_malignant']=='malignant']





f = plt.figure(figsize=(16,8))

f.add_subplot(2,2, 1)



benign_img = benign['image_name'][1]+'.jpg'

benign_img = plt.imread(os.path.join(img_dir, benign_img))

plt.imshow(benign_img[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Benign Image', fontsize=15)



f.add_subplot(2,2, 2)

_ = plt.hist(benign_img.ravel(),256,[0,256])



f.add_subplot(2,2, 3)

malignant_img = malignant['image_name'][235]+'.jpg'

malignant_img = plt.imread(os.path.join(img_dir, malignant_img))

plt.imshow(malignant_img[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Malignant Image', fontsize=15)



f.add_subplot(2,2, 4)

_ = plt.hist(malignant_img.ravel(),256,[0,256])



plt.tight_layout() 

plt.show()
f = plt.figure(figsize=(10,5))

f.add_subplot(2,2, 1)



benign_img = benign['image_name'][2]+'.jpg'

img = plt.imread(os.path.join(img_dir, benign_img))



# create a mask

mask = np.zeros(img.shape[:2], np.uint8)

mask[200:800, 500:1400] = 255

masked_img = cv2.bitwise_and(img,img,mask = mask)



# Calculate histogram with mask and without mask

# Check third argument for mask

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])

hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])



raw_image = plt.imread(os.path.join(img_dir, benign_img))

plt.imshow(raw_image[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Benign Image', fontsize=15)



f.add_subplot(2,2, 2)

plt.imshow(mask, cmap='gray')

plt.colorbar()

plt.title('Mask', fontsize=15)



f.add_subplot(2,2, 3)

plt.imshow(masked_img[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Mask Image', fontsize=15)



f.add_subplot(2,2, 4)

plt.plot(hist_full)

plt.plot(hist_mask)

plt.title('Histogram', fontsize=15)



plt.tight_layout() 

plt.show()
f = plt.figure(figsize=(10,5))

f.add_subplot(2,2, 1)



malignant_img = malignant['image_name'][235]+'.jpg'

img = plt.imread(os.path.join(img_dir, malignant_img))



# create a mask

mask = np.zeros(img.shape[:2], np.uint8)

mask[500:2000, 500:2500] = 255

masked_img = cv2.bitwise_and(img,img,mask = mask)



# Calculate histogram with mask and without mask

# Check third argument for mask

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])

hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])



raw_image = plt.imread(os.path.join(img_dir, malignant_img))

plt.imshow(raw_image[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Malignant Image', fontsize=15)



f.add_subplot(2,2, 2)

plt.imshow(mask, cmap='gray')

plt.colorbar()

plt.title('Mask', fontsize=15)



f.add_subplot(2,2, 3)

plt.imshow(masked_img[ : , : , 0], cmap='gray')

plt.colorbar()

plt.title('Mask Image', fontsize=15)



f.add_subplot(2,2, 4)

plt.plot(hist_full)

plt.plot(hist_mask)

plt.title('Histogram', fontsize=15)



plt.tight_layout() 

plt.show()
# https://www.kaggle.com/parulpandey/melanoma-classification-eda-starter

f = plt.figure(figsize=(16,8))

f.add_subplot(2,2, 1)



benign_img = benign['image_name'][1]+'.jpg'

benign_img = plt.imread(os.path.join(img_dir, benign_img))

plt.imshow(benign_img, cmap='gray')

plt.colorbar()

plt.title('Benign Image', fontsize=15)



f.add_subplot(2,2, 2)

_ = plt.hist(benign_img.ravel(),bins = 256, color = 'orange', alpha=0.3)

_ = plt.hist(benign_img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(benign_img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(benign_img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])



f.add_subplot(2,2, 3)

malignant_img = malignant['image_name'][235]+'.jpg'

malignant_img = plt.imread(os.path.join(img_dir, malignant_img))

plt.imshow(malignant_img, cmap='gray')

plt.colorbar()

plt.title('Malignant Image', fontsize=15)



f.add_subplot(2,2, 4)

_ = plt.hist(malignant_img.ravel(),bins = 256, color = 'orange', alpha = 0.3)

_ = plt.hist(malignant_img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(malignant_img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(malignant_img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])



plt.tight_layout() 

plt.show()
def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)
def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array)

    plt.show()

    

i = 1

num_to_plot = 5

for file_name in os.listdir('../input/siim-isic-melanoma-classification/train/'):

        file_path = os.path.join('../input/siim-isic-melanoma-classification/train/',file_name)

        dataset = dcm.dcmread(file_path)

        show_dcm_info(dataset)

        plot_pixel_array(dataset)

    

        if i >= num_to_plot:

            break

    

        i += 1
%%time

# https://www.kaggle.com/tunguz/melanoma-tsne-and-umap-embeddings-with-rapids/?



# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
# load resized image numpy arry

from cuml.manifold import TSNE

from cuml.decomposition import PCA



train = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')

train = train.reshape((train.shape[0], 32*32*3))

train.shape
time_start = time.time()

pca = PCA(n_components=2)

pca_2D = pca.fit_transform(train.astype(np.float32))

print(pca.explained_variance_ratio_)

print(' ')

print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
pca_2D_one  = pca_2D[:,0]

pca_2D_two = pca_2D[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x=pca_2D_one, y=pca_2D_two,

    hue=train_df['target'].values,

    palette=sns.color_palette("Paired", 2),

    legend="full",

    alpha=0.7

)

plt.xlabel('Principal Component 1', fontsize = 15)

plt.ylabel('Principal Component 2', fontsize = 15)

plt.title('2 component PCA', fontsize = 20)

plt.show()
time_start = time.time()

tsne = TSNE(n_components=2)

tsne_2D = tsne.fit_transform(train)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tsne_2D_one  = tsne_2D[:,0]

tsne_2D_two = tsne_2D[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x=tsne_2D_one, y=tsne_2D_two,

    hue=train_df['target'].values,

    palette=sns.color_palette("Set2", 2),

    legend="full",

    alpha=0.7

)

plt.xlabel('t-SNE 1', fontsize = 15)

plt.ylabel('t-SNE 2', fontsize = 15)

plt.title('2 component t-SNE' ,fontsize = 20)

plt.show()