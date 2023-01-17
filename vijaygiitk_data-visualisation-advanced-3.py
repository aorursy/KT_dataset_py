import numpy as np  # useful for many scientific computing in Python

import pandas as pd # primary data structure library

from PIL import Image # converting images into arrays
!conda install -c anaconda xlrd --yes
df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',

                       sheet_name='Canada by Citizenship',

                       skiprows=range(20),

                       skipfooter=2)



print('Data downloaded and read into a dataframe!')
# clean up the dataset to remove unnecessary columns (eg. REG) 

df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)



# let's rename the columns so that they make sense

df_can.rename (columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace = True)



# for sake of consistency, let's also make all column labels of type string

df_can.columns = list(map(str, df_can.columns))



# set the country name as index - useful for quickly looking up countries using .loc method

df_can.set_index('Country', inplace = True)



# add total column

df_can['Total'] =  df_can.sum (axis = 1)



# years that we will be using in this lesson - useful for plotting later on

years = list(map(str, range(1980, 2014)))

print ('data dimensions:', df_can.shape)
%matplotlib inline



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches # needed for waffle Charts



mpl.style.use('ggplot') # optional: for ggplot-like style



# check for latest version of Matplotlib

print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# let's create a new dataframe for these three countries 

df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]



# let's take a look at our dataframe

df_dsn
# compute the proportion of each category with respect to the total

total_values = sum(df_dsn['Total'])

category_proportions = [(float(value) / total_values) for value in df_dsn['Total']]



# print out proportions

for i, proportion in enumerate(category_proportions):

    print (df_dsn.index.values[i] + ': ' + str(proportion))
width = 40 # width of chart

height = 10 # height of chart



total_num_tiles = width * height # total number of tiles



print ('Total number of tiles is ', total_num_tiles)
# compute the number of tiles for each catagory

tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]



# print out number of tiles per category

for i, tiles in enumerate(tiles_per_category):

    print (df_dsn.index.values[i] + ': ' + str(tiles))
# initialize the waffle chart as an empty matrix

waffle_chart = np.zeros((height, width))



# define indices to loop through waffle chart

category_index = 0

tile_index = 0



# populate the waffle chart

for col in range(width):

    for row in range(height):

        tile_index += 1



        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...

        if tile_index > sum(tiles_per_category[0:category_index]):

            # ...proceed to the next category

            category_index += 1       

            

        # set the class value to an integer, which increases with class

        waffle_chart[row, col] = category_index

        

print ('Waffle chart populated!')
waffle_chart
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()



# get the axis

ax = plt.gca()



# set minor ticks

ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

# add gridlines based on minor ticks

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



plt.xticks([])

plt.yticks([])
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()



# get the axis

ax = plt.gca()



# set minor ticks

ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

# add gridlines based on minor ticks

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



plt.xticks([])

plt.yticks([])



# compute cumulative sum of individual categories to match color schemes between chart and legend

values_cumsum = np.cumsum(df_dsn['Total'])

total_values = values_cumsum[len(values_cumsum) - 1]



# create legend

legend_handles = []

for i, category in enumerate(df_dsn.index.values):

    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'

    color_val = colormap(float(values_cumsum[i])/total_values)

    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))



# add legend to chart

plt.legend(handles=legend_handles,

           loc='lower center', 

           ncol=len(df_dsn.index.values),

           bbox_to_anchor=(0., -0.2, 0.95, .1)

          )
def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):



    # compute the proportion of each category with respect to the total

    total_values = sum(values)

    category_proportions = [(float(value) / total_values) for value in values]



    # compute the total number of tiles

    total_num_tiles = width * height # total number of tiles

    print ('Total number of tiles is', total_num_tiles)

    

    # compute the number of tiles for each catagory

    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]



    # print out number of tiles per category

    for i, tiles in enumerate(tiles_per_category):

        print (df_dsn.index.values[i] + ': ' + str(tiles))

    

    # initialize the waffle chart as an empty matrix

    waffle_chart = np.zeros((height, width))



    # define indices to loop through waffle chart

    category_index = 0

    tile_index = 0



    # populate the waffle chart

    for col in range(width):

        for row in range(height):

            tile_index += 1



            # if the number of tiles populated for the current category 

            # is equal to its corresponding allocated tiles...

            if tile_index > sum(tiles_per_category[0:category_index]):

                # ...proceed to the next category

                category_index += 1       

            

            # set the class value to an integer, which increases with class

            waffle_chart[row, col] = category_index

    

    # instantiate a new figure object

    fig = plt.figure()



    # use matshow to display the waffle chart

    colormap = plt.cm.coolwarm

    plt.matshow(waffle_chart, cmap=colormap)

    plt.colorbar()



    # get the axis

    ax = plt.gca()



    # set minor ticks

    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

    # add dridlines based on minor ticks

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



    plt.xticks([])

    plt.yticks([])



    # compute cumulative sum of individual categories to match color schemes between chart and legend

    values_cumsum = np.cumsum(values)

    total_values = values_cumsum[len(values_cumsum) - 1]



    # create legend

    legend_handles = []

    for i, category in enumerate(categories):

        if value_sign == '%':

            label_str = category + ' (' + str(values[i]) + value_sign + ')'

        else:

            label_str = category + ' (' + value_sign + str(values[i]) + ')'

            

        color_val = colormap(float(values_cumsum[i])/total_values)

        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))



    # add legend to chart

    plt.legend(

        handles=legend_handles,

        loc='lower center', 

        ncol=len(categories),

        bbox_to_anchor=(0., -0.2, 0.95, .1)

    )
width = 40 # width of chart

height = 10 # height of chart



categories = df_dsn.index.values # categories

values = df_dsn['Total'] # correponding values of categories



colormap = plt.cm.coolwarm # color map class
create_waffle_chart(categories, values, height, width, colormap)
# install wordcloud

!conda install -c conda-forge wordcloud==1.4.1 --yes



# import package and its set of stopwords

from wordcloud import WordCloud, STOPWORDS



print ('Wordcloud is installed and imported!')
# download file and save as alice_novel.txt

!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/alice_novel.txt



# open the file and read it into a variable alice_novel

alice_novel = open('alice_novel.txt', 'r').read()

    

print ('File downloaded and saved!')
stopwords = set(STOPWORDS)
# instantiate a word cloud object

alice_wc = WordCloud(

    background_color='white',

    max_words=2000,

    stopwords=stopwords

)



# generate the word cloud

alice_wc.generate(alice_novel)
# display the word cloud

plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



# display the cloud

plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
stopwords.add('said') # add the words said to stopwords



# re-generate the word cloud

alice_wc.generate(alice_novel)



# display the cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
# download image

!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Images/alice_mask.png

    

# save mask to alice_mask

alice_mask = np.array(Image.open('alice_mask.png'))

    

print('Image downloaded and saved!')
fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')

plt.axis('off')

plt.show()
# instantiate a word cloud object

alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)



# generate the word cloud

alice_wc.generate(alice_novel)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
df_can.head()
total_immigration = df_can['Total'].sum()

total_immigration
max_words = 90

word_string = ''

for country in df_can.index.values:

    # check if country's name is a single-word name

    if len(country.split(' ')) == 1:

        repeat_num_times = int(df_can.loc[country, 'Total']/float(total_immigration)*max_words)

        word_string = word_string + ((country + ' ') * repeat_num_times)

                                     

# display the generated text

word_string
# create the word cloud

wordcloud = WordCloud(background_color='white').generate(word_string)



print('Word cloud created!')
# display the cloud

fig = plt.figure()

fig.set_figwidth(14)

fig.set_figheight(18)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# import library

import seaborn as sns



print('Seaborn imported!')
# we can use the sum() method to get the total population per year

df_tot = pd.DataFrame(df_can[years].sum(axis=0))



# change the years to type float (useful for regression later on)

df_tot.index = map(float, df_tot.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot.reset_index(inplace=True)



# rename columns

df_tot.columns = ['year', 'total']



# view the final dataframe

df_tot.head()
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot)
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot, color='green')
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.figure(figsize=(15, 10))

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.figure(figsize=(15, 10))

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})



ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels

ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)

sns.set_style('ticks') # change background to white background



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)

sns.set_style('whitegrid')



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
# create df_countries dataframe

df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()



# create df_total by summing across three countries for each year

df_total = pd.DataFrame(df_countries.sum(axis=1))



# reset index in place

df_total.reset_index(inplace=True)



# rename columns

df_total.columns = ['year', 'total']



# change column year from string to int to create scatter plot

df_total['year'] = df_total['year'].astype(int)



# define figure size

plt.figure(figsize=(15, 10))



# define background style and font size

sns.set(font_scale=1.5)

sns.set_style('whitegrid')



# generate plot and add title and axes labels

ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigrationn from Denmark, Sweden, and Norway to Canada from 1980 - 2013')