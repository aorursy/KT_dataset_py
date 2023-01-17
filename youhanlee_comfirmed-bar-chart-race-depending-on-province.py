import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import random

import warnings

warnings.filterwarnings('ignore')
case = pd.read_csv('../input/coronavirusdataset/Case.csv')
print(case.shape)

case.head()
patientinfo = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')

print(patientinfo.shape)

patientinfo.head()
patientroute = pd.read_csv('../input/coronavirusdataset/PatientRoute.csv')
print(patientroute.shape)

patientroute.head()
timeprovince = pd.read_csv('../input/coronavirusdataset/TimeProvince.csv')

timeprovince
df = timeprovince[['date', 'province', 'confirmed']]
df.shape
import numpy as np
#df = df.pivot(index='province', columns='date', values='confirmed')

#df = df.reset_index()



#for p in range(2):

#    i = 0

#    while i < len(df.columns):

#        try:

#            a = np.array(df.iloc[:, i + 1])

#            b = np.array(df.iloc[:, i + 2])

#            c = (a + b) / 2

#            df.insert(i+2, str(df.iloc[:, i + 1].name) + '^' + str(len(df.columns)), c)

#        except:

#            print(f"\n  Interpolation No. {p + 1} done...")

#        i += 2
df = timeprovince[['date', 'province', 'confirmed']]
df.head()
df.columns = ['date', 'province', 'value']
#df = pd.melt(df, id_vars = 'province', var_name = 'date')
fnames_list = df['date'].unique().tolist()
def random_color_generator(number_of_colors):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                 for i in range(number_of_colors)]

    return color
province_list = df['province'].unique().tolist()
province_to_kor = {'Seoul': '서울',

 'Busan' : '부산',

 'Daegu': '대구',

 'Incheon': '인천',

 'Gwangju': '광주',

 'Daejeon': '대전',

 'Ulsan': '울산',

 'Sejong': '세종',

 'Gyeonggi-do': '경기도',

 'Gangwon-do': '강원도',

 'Chungcheongbuk-do': '충청북도',

 'Chungcheongnam-do': '충청남도',

 'Jeollabuk-do': '전라북도',

 'Jeollanam-do': '전라남도',

 'Gyeongsangbuk-do': '경상북도',

 'Gyeongsangnam-do': '경상남도',

 'Jeju-do':'제주도'}
df['province_kr'] = df['province'].map(province_to_kor)
colors = dict(zip(province_list, random_color_generator(len(province_list))))
num_of_elements = 10
import matplotlib.colors as mc

import colorsys

from random import randint
def transform_color(color, amount = 0.5):



    try:

        c = mc.cnames[color]

    except:

        c = color

        c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



random_hex_colors = []

for i in range(len(province_list)):

    random_hex_colors.append('#' + '%06X' % randint(0, 0xFFFFFF))



rgb_colors = [transform_color(i, 1) for i in random_hex_colors]

rgb_colors_opacity = [rgb_colors[x] + (0.825,) for x in range(len(rgb_colors))]

rgb_colors_dark = [transform_color(i, 1.12) for i in random_hex_colors]
normal_colors = dict(zip(province_list, rgb_colors_opacity))

dark_colors = dict(zip(province_list, rgb_colors_dark))
import re
fig, ax = plt.subplots(figsize = (36, 20))



def draw_barchart(current_date):

    dff = df[df['date'].eq(current_date)].sort_values(by='value', ascending=True).tail(num_of_elements)

    ax.clear()

    

    ax.barh(dff['province'], dff['value'], color=[normal_colors[p] for p in dff['province']],

                edgecolor =([dark_colors[x] for x in dff['province']]), linewidth = '6')

    dx = dff['value'].max() / 200





    for i, (value, name) in enumerate(zip(dff['value'], dff['province'])):

        ax.text(value + dx, 

                i + (num_of_elements / 50), '    ' + name,

                size = 32,

                ha = 'left',

                va = 'center',

                fontdict = {'fontname': 'Trebuchet MS'})



        ax.text(value + dx,

                i - (num_of_elements / 50), 

                f'    {value:,.0f}', 

                size = 32, 

                ha = 'left', 

                va = 'center')



    time_unit_displayed = re.sub(r'\^(.*)', r'', str(current_date))

    ax.text(1.0, 

            1.1, 

            time_unit_displayed,

            transform = ax.transAxes, 

            color = '#666666',

            size = 32,

            ha = 'right', 

            weight = 'bold', 

            fontdict = {'fontname': 'Trebuchet MS'})



    ax.text(-0.005, 

            1.05, 

            'Confirmed', 

            transform = ax.transAxes, 

            size = 32, 

            color = '#666666')



    ax.text(-0.005, 

            1.1, 

            'Confirmed from 2020-01-20 to 2020-03-22', 

            transform = ax.transAxes,

            size = 32, 

            weight = 'bold', 

            ha = 'left')



    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis = 'x', colors = '#666666', labelsize = 28)

    ax.set_yticks([])

    ax.set_axisbelow(True)

    ax.margins(0, 0.01)

    ax.grid(which = 'major', axis = 'x', linestyle = '-')



    plt.locator_params(axis = 'x', nbins = 4)

    plt.box(False)

    plt.subplots_adjust(left = 0.075, right = 0.75, top = 0.825, bottom = 0.05, wspace = 0.2, hspace = 0.2)

    

    ax.text(1, 

            0, 

            'by @youhanlee; credit @Korea Centers for Disease Control and Prevention @jihookim', 

            transform=ax.transAxes, 

            color='#777777', 

            ha='right', 

            size=32,

            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)    

draw_barchart('2020-03-20')
fig, ax = plt.subplots(figsize = (36, 20))

animator = animation.FuncAnimation(fig, draw_barchart, frames=fnames_list)

HTML(animator.to_jshtml())
#Writer = animation.writers['ffmpeg']

#writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1600)

#animator.save("corona_province_kr_new.mp4", writer=writer)