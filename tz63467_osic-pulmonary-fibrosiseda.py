import os
import pydicom
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
sample_sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df_train.shape
#Function below gives us all the details about the number of Patients, smoking status, percent group of the patients, Sex,etc.

def general_info(df):
    info = dict()
    info['patients'] = df['Patient'].nunique()
    info['Male'] = df['Patient'][df.Sex == 'Male'].nunique()
    info['Female'] = df['Patient'][df.Sex == 'Female'].nunique()
    info['Ex-smoker'] = df['Patient'][df.SmokingStatus == 'Ex-smoker'].nunique()
    info['Ex-smoker-male'] = df['Patient'].loc[(df['SmokingStatus'] == 'Ex-smoker') & (df['Sex'] == 'Male')].nunique()
    info['Ex-smoker-female'] = df['Patient'].loc[(df['SmokingStatus'] == 'Ex-smoker') & (df['Sex'] == 'Female')].nunique()
    info['Never smoked'] = df['Patient'][df.SmokingStatus == 'Never smoked'].nunique()
    info['Never smoked-male'] = df['Patient'].loc[(df['SmokingStatus'] == 'Never smoked') & (df['Sex'] == 'Male')].nunique()
    info['Never smoked-female'] = df['Patient'].loc[(df['SmokingStatus'] == 'Never smoked') & (df['Sex'] == 'Female')].nunique()
    info['Currently smokes'] = df['Patient'][df.SmokingStatus == 'Currently smokes'].nunique()
    info['Currently smokes-male'] = df['Patient'].loc[(df['SmokingStatus'] == 'Currently smokes') & (df['Sex'] == 'Male')].nunique()
    info['Currently smokes-female'] = df['Patient'].loc[(df['SmokingStatus'] == 'Currently smokes') & (df['Sex'] == 'Female')].nunique()
    info['null_values'] = df.isna().sum()

    return info
general_info(df_train)
base = alt.Chart(df_train.groupby('Patient').head(1)).mark_bar(size=8).encode(
            alt.X('Age', title='Age of patients'),
            alt.Y('count(Patient)', title='Number of Patents'),
            tooltip = ['Patient', 'Age', 'Sex', 'SmokingStatus']
            ).properties(
            width=400,
            height=300
            ).interactive()

alt.concat(
    base.encode(color='Sex:N'),
    base.encode(color='SmokingStatus:N'))
base = alt.Chart(df_train.groupby('Patient').head(1)).mark_circle(size=100).encode(
    x = 'Age:O',
    y = 'FVC:Q',
    color = 'Sex',
    tooltip = ['Patient', 'FVC', 'Sex', 'SmokingStatus']
    ).properties(
        width=400,
        height=300
    ).interactive()

alt.concat(
    base.encode(color='Sex:N'),
    base.encode(color='SmokingStatus:N'))
brush = alt.selection_interval()  # selection of type "interval"

chart = alt.Chart(df_train.groupby('Patient').head(1)).mark_circle(size=80).encode(
x = 'Age:O',
color = alt.condition(brush,'Sex:N', alt.value('lightgray')),
    tooltip = ['Age', 'Percent', 'FVC', 'Sex']
).properties(
    width=320,
    height=300
).add_selection(
    brush
)

chart.encode(y='FVC:Q') | chart.encode(y='Percent:Q')
chart = alt.Chart(df_train.groupby('Patient').head(1)).mark_circle(size=80).encode(
x = 'Age:O',
color = alt.condition(brush,'SmokingStatus:N', alt.value('lightgray')),
    tooltip = ['Age','Percent', 'FVC', 'Sex']
).properties(
    width=320,
    height=300
).add_selection(
    brush
)

chart.encode(y='FVC:Q') | chart.encode(y='Percent:Q')