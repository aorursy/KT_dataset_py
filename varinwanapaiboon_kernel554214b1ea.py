# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# ดึงข้อมูลจากไฟล์

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt     # for visualisation

import seaborn as sns     # for visualisation

from wordcloud import WordCloud    # for create word cloud

import random    # for use in random color in word cloud



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# ข้อมูลที่มีอยู่ในdata set 

ramen_data = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')

ramen_data.head()
ramen_data.info()
ramen_data.replace([np.inf, -np.inf], np.nan)

ramen_data.isnull().any()

ramen_data['Style'].value_counts()
ramen_data['Top Ten'].value_counts()
ramen_data['Stars'].value_counts()
def length_Style(x):

    if pd.isnull(x):

        return 'Pack'

    else:

        return x

ramen_data['Style'] = ramen_data['Style'].apply(length_Style)
def length_Top_Ten(x):

    if pd.isnull(x):

        return '\n'

    else:

        return x

ramen_data['Top Ten'] = ramen_data['Top Ten'].apply(length_Top_Ten)
def length_Stars(x):

    if pd.isnull(x) or 'Unrated':

        return 3.75

    else:

        return x

ramen_data['Stars'] = ramen_data['Stars'].apply(length_Stars)
ramen_data['Brand'] = ramen_data['Brand'].str.lower()

ramen_data.head()
# นับจำนวนแบรนด์ที่ได้รับการีวิว

ramen_brand = ramen_data.groupby(['Brand','Country']).agg({'Review #':'count'})

ramen_brand = ramen_brand.reset_index() 

ramen_brand = ramen_brand.sort_values('Review #', ascending = False)

# นับแบรนด์จากแต่ละประเทศที่ได้รับการีวิว

ramen_coun = ramen_brand.groupby('Country').agg({'Brand':'count'}).reset_index()

ramen_coun = ramen_coun.rename(columns = {'Brand':'Amount of brand'})

ramen_coun = ramen_coun.sort_values(['Amount of brand', 'Country'], ascending = [False, True])

# ดู 10 ประเทศแรกที่มีราเม็งมากที่สุด

ramen_coun.head(10)
# แผนภูมิแท่งของจำนวนแบรนด์ราเม็งในแต่ละประเทศที่ได้รับการรีวิว

plt.figure(figsize=(15, 5))

plt.bar('Country', 'Amount of brand', data = ramen_coun, color = 'gold')

plt.title( 'The amount of ramen brands in each country', fontsize=14)

plt.ylabel('Number of brands')

plt.xticks(rotation = 90)

plt.show()
# นำเสนอความหลากหลายของแต่ละประเทศที่ได้รับการรีวิว

ramen_variety = ramen_data.groupby(['Country']).agg({'Variety':'count'})

ramen_variety = ramen_variety.reset_index() 

ramen_variety = ramen_variety.sort_values(['Variety','Country'], ascending = [False, True])

ramen_variety = ramen_variety.rename(columns = {'Variety': 'Country variety'})



# แผนภูมิแท่งของจำนวนผลิตภัณฑ์ราเม็งในแต่ละประเทศที่ได้รับการรีวิว

plt.figure(figsize=(15, 5))

plt.bar('Country', 'Country variety', data = ramen_variety, color = 'peru')

plt.title( 'The amount of ramen product in each country', fontsize=14)

plt.ylabel('Number of product')

plt.xticks(rotation = 90)

plt.show()
# Count number of style in each country

ramen_style = ramen_data.groupby(['Country','Style']).agg({'Variety':'count'})

ramen_style = ramen_style.reset_index()

ramen_style.head()
# Find the unique ramen styles

style_name = sorted(ramen_style['Style'].unique())

print(style_name)
# Not every styles were produce in every countries,thus, those styles were not present in the table

# Create the index of every styles in each country and add count number 0

# by create dummie column, merge and fill the NaN with 0

pattern = pd.DataFrame({'dummie' : [0]*266}, \

                       index = pd.MultiIndex.from_product([ramen_coun['Country'], style_name], \

                       names = ['Country', 'Style']))

ramen_style = pd.merge(ramen_style, pattern, how='outer', on=['Country', 'Style'])

ramen_style = ramen_style[['Country', 'Style', 'Variety']].fillna(0)



# Merge ramen_style with ramen_variety to be able to sort with the amount of brand

ramen_style = pd.merge(ramen_style, ramen_variety, how = 'left', on = 'Country')

ramen_style =ramen_style.sort_values(['Country variety','Country', 'Style'], ascending = [False,True, True])

# Create stack bar chart

plt.figure(figsize=(15, 5))

bottom_bar = [0]*38 # for identify the bottom of the bar graph in each style

bar_color = ['chocolate', 'yellowgreen', 'orange', 'forestgreen', 'peru', 'gold', 'saddlebrown']



# Use for loop for plot bar chart and stack the amount of ramen in each ramen style

for i in range(len(style_name)):

    plt.bar('Country', 'Variety', data = ramen_style[ramen_style['Style'] == style_name[i]], \

            bottom = bottom_bar, color = bar_color[i])

    # change the bottom_bar to the the amount of current style for the next loop

    bottom_bar = list(np.add(bottom_bar, ramen_style[ramen_style['Style'] == style_name[i]]['Variety']))



plt.title( 'The amount of ramen style in each country', fontsize=14)

plt.ylabel('Number of ramen')

plt.xticks(rotation = 90)

plt.legend(style_name)

plt.show()
# Create percentage stack bar chart of countries which have more than or equal to 50 products reviewed

# Select only countries which have more than or equal to 50 products reviewed

ramen_per = ramen_style[ramen_style['Country variety'] >= 50].reset_index()



# Create percentage column in ramen_style

ramen_per['Percentage'] = ramen_per['Variety'] * 100 / ramen_per['Country variety']



# Create percentage stack bar chart

plt.figure(figsize=(14, 5))

bottom_bar = [0]*12 # for identify the bottom of the bar graph in each style

for i in range(len(style_name)):

    plt.bar('Country', 'Percentage', data = ramen_per[ramen_per['Style'] == style_name[i]], \

            bottom = bottom_bar, color = bar_color[i])

    bottom_bar = list(np.add(bottom_bar, ramen_per[ramen_per['Style'] == style_name[i]]['Percentage']))



plt.title('The percentage of ramen style in countries which have more than or equal to 50 products reviewed', \

          fontsize=14)

plt.ylabel('Per cent')

plt.xticks(rotation = 90)

plt.legend(style_name,bbox_to_anchor=(1.1, 1))    # move legend box to the right of the graph

plt.show()
ramen_data.describe(include='all')
# Convert 'Stars' column to int

ramen_data['Stars'] = pd.to_numeric(ramen_data['Stars'], errors = 'coerce')

ramen_data.describe(include='all')
# Group ramen_data by Country and Brand column 

# and calculate the mean and median of Stars that each brand received

ramen_stars = ramen_data.groupby(['Country','Brand']).agg({'Stars': ['mean', 'median'], 'Review #': 'count'})

ramen_stars = ramen_stars.reset_index()

ramen_stars.columns = ['Country','Brand','Mean Stars', 'Median Stars', 'Review#']

ramen_stars = ramen_stars.sort_values('Median Stars', ascending = False)



# Create new column for label

ramen_stars['Country Brand'] = ramen_stars['Brand'] + ' (' + ramen_stars['Country'] + ')'

ramen_stars.head()
# ดูแบรนด์ 5 อันดับแรกที่มีดาวเฉลี่ยสูงสุด (เรียงลำดับตามค่าเฉลี่ย)

ramen_stars_re = ramen_stars[ramen_stars['Review#'] >= 10].reset_index()

ramen_stars_re = ramen_stars_re.sort_values('Mean Stars', ascending = False)

ramen_stars_re.head()
# ดูแบรนด์ 5 อันดับแรกที่มีดาวเฉลี่ยต่ำสุด (เรียงตามค่าเฉลี่ย)

ramen_stars_re.tail()
# Create box plot with mean

# Sort by median of the stars for the order in box plot

ramen_stars_re = ramen_stars_re.sort_values('Median Stars', ascending = False)



# Create boxplot

ramen_box = ramen_data[['Country','Brand','Stars']].reset_index()

ramen_box['Country Brand'] = ramen_box['Brand'] + ' (' + ramen_box['Country'] + ')'



# Select only brand in country that in ramen_stars_re

ramen_box = ramen_box[ramen_box['Country Brand'].isin(ramen_stars_re['Country Brand'])]



# Create boxplot

fig, ax = plt.subplots(figsize=(5, 20))

sns.boxplot(x = 'Stars', y = 'Country Brand', data = ramen_box, color = 'yellow',\

            order = ramen_stars_re['Country Brand'], showmeans = True,\

            meanprops = {'marker': 'o','markerfacecolor': 'saddlebrown', 'markeredgecolor': 'saddlebrown'})

ax.xaxis.tick_top()

ax.xaxis.set_label_position('top') 

plt.title( 'The distribution of the stars in each brand (mean display as brown circles)', \

          fontsize=14)

plt.show()
# จัดอันดับราเมนตามคอลัมน์ดาว

ramen_sort = ramen_data.sort_values('Stars').dropna(subset = ['Stars'])



# แบ่งออกเป็น 100 อันดับแรกและต่ำสุด 100

ramen_top = ramen_sort.head(100)

ramen_bottom = ramen_sort.tail(100)

ramen_bottom.head()
categori = ['Style', 'Country']

model_data = pd.get_dummies(ramen_data.copy(), columns=categori,drop_first=True)
# กำหนด X และ y

X = model_data.drop(columns=['Brand', 'Review #','Top Ten','Variety'],axis=1)

y = model_data['Brand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

ramen = RandomForestClassifier(n_estimators=100)

ramen.fit(X_train,y_train)
ramen_data.info()
ramen_data.replace([np.inf, -np.inf], np.nan)

ramen_data.isnull().any()
predictions = ramen.predict(X_test)
print(classification_report(y_test,predictions))