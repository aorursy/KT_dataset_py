# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import math

import PIL

from PIL import Image

from PIL import ImageStat
# read the csv files

train = pd.read_csv('/kaggle/input/pollutionvision/train_data.csv')

test = pd.read_csv('/kaggle/input/pollutionvision/test_data.csv')

sample = pd.read_csv('/kaggle/input/pollutionvision/sample.csv')
# check with the train data

train.head(10)



# meaning of the colmunn names

## Temp(C) = ambient temperature

## Pressure(Kpa) = air pressure

## Rel. Humidity = relative humidity

## Errors = error that the air measurement equipment has during sampling (0=no)

## Alarm Triggered = instrumental warning shown during sampling (0=no)

## Dilution factor = an instrumental parameter (should close to 1)

## Dead time = an instrumental parameter (ideally close to 0)

## Median, Mean, Geo. Mean, Mode, and Geo. St. Dev. = parameters describe particle sizes (can be ignored)

## Total Conc.(#/cm³) = an output variable from the instrument that should not be used

## image_file = visual information of the traffic condition, corresponding to an image in the "frames" directory

## wind_speed = wind velocity during sampling

## Distance_to_road = distance between camera and road

## camera angle = angle of incidence between the camera and the road

## Elevation = elevation between the camera and the breathing zone

## Total(#/cm³) = total measured particle number concentration (dependent variable)
train.shape
# clean up the train data

## the first column titled"unnamed" is superfluous,delete.

## the columns titiled "Median, Mean, Geo. Mean, Mode, and Geo. St. Dev."are not needed,delete.

## the column titiled "Total Conc.(#/cm³)" is an output variable and should not be used,delete.

train = train[['Temp(C)','Pressure(kPa)','Rel. Humidity','Errors','Alarm Triggered','Dilution Factor','Dead Time','Image_file','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation','Total']]
train = train[train['Errors'] == 0].reset_index(drop=True)

## Errors = error that the air measurement equipment has during sampling (0=no)

## only reserve data with no error (value = 0)



train = train[train['Alarm Triggered'] == 0].reset_index(drop=True)

## Alarm Triggered = instrumental warning shown during sampling (0=no)

## only reserve data with warning (value = 0)



train = train[train['Dilution Factor'] == 1].reset_index(drop=True)

## Dilution factor = an instrumental parameter (should close to 1)

## only reserve data with ideal value (value = 1)



train = train[train['Dead Time'] <= 0.005].reset_index(drop=True)

## Dead time = an instrumental parameter (ideally close to 0)

## only reserve data with ideal value (value <= 0.005)
# check with the new train data

train.head(10)
train.describe()
# go on cleaning up the train data

## in the last step, we only reserve data with no error, no warning, and ideal dilution factor.

## that is why the std for columns "Errors""Alarm Triggered"and "Dilution Factor" is 0.

## the std for the column "Rel. Humidity" is also 0. Thus, I delected these mentioned 4 columns.

train = train[['Temp(C)','Pressure(kPa)','Dead Time','Image_file','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation','Total']]
# check with the new train data

train.head(10)
train.shape

# after neccessary data cleaning, the train data size:

# decreases from (64961 rows, 20 columns) to (60091 rows, 9 columns)
# The current train data has 9 columns. 

# 7 external factors: the first 8 columns except "Image_file" column)

factor_list = ['Temp(C)', 'Pressure(kPa)', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation']



# Measured total particle concentration: the last column "Total"

# We are concerned with the measured total particle concentration.
# plot the (factor,concentration) diagram

for factor in factor_list:

    plt.scatter(train[factor],train['Total'])

    plt.xlabel(factor)

    plt.ylabel('total particle concentration')

    plt.show()
# Observe the histogram

train.hist(figsize=(15, 20), bins=25, xlabelsize=5, ylabelsize=5);
# Observation 1

## The measured total partical concentration has two peaks: one is close to 100, the other close to 300.



# Observation 2

## The values for factors such as "Distance_to_Road","Camera_Angle"and "Elevation"are relatively fixed.

## For example, the elevation between the camera and the breathing zone is either 0, or 0.2, or 0.5, or 0.6.

## while the weather data such as "Temp(C)","Pressure(Kpa)", and "Wind_speed" has a large range of values.

## Thus, next I will explore the relationship between weather and particle concentration.
train_weather = train[['Temp(C)','Pressure(kPa)','Wind_Speed','Total']]

train_weather.head(10)
train_weather.corr()
# visualize the correlation between weather data and paticle concentration

sns.heatmap(train_weather.corr(),cmap = 'bwr')
# When deep=True (default), a new object will be created with a copy of the calling object’s data and indices. 

# Modifications to the data or indices of the copy will not be reflected in the original object

train_image = train.copy(deep=True)

train_image
# extract the date and the number of the images from the "Image_file" columns

train_image['Date'] = train_image['Image_file'].str[5:9]

train_image['Image_Number'] = train_image['Image_file'].str.split('_').str[1]

train_image['Number'] = train_image['Image_Number'].str.split('.').str[0]
# replace the "Image_file" column with "Date" and "Number" column whose expression is more digital

train_image = train_image[['Date','Number','Temp(C)','Pressure(kPa)','Dead Time','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation','Total']]

train_image = train_image.sort_values(by=['Date','Number'])

train_image
# extract the train data on date 06/08 directly

images_on_0608 = train_image.loc[train_image['Date']=='0608']

images_on_0608
# extract the train data on different dates

for date, frame in train_image.groupby('Date'):

    globals()['images_on_' +str(date)] = frame
# apply the algorithm in last step and test if the result is the same

# compare the result below and the result from direct extraction above, the algorithm is fine.

images_on_0608
images_on_0608.describe()

# from the dataframe, we can see the average particle concentration is 123.040055.
total_concentration_0608= images_on_0608['Total']

plt.plot(total_concentration_0608,'o')

plt.xlabel('No. of images')

plt.ylabel('Particle concentration values')

plt.title('06/08/2020')

plt.show()

# from the graph, we can see the particle concentration values mainly lie in [100,150] on 06/08.

# it is consistent with the result that the average particle concentration is 123.040055 on 06/08.
# try on extracting the train data on other dates

# guarantee the accuracy of the algorithm

# the train data on any date can be obtained now

images_on_0807
images_on_0807.describe()

# from the dataframe, we can see the average particle concentration is 250.503498.
total_concentration_0807 = images_on_0807['Total']

plt.plot(total_concentration_0807,'o')

plt.xlabel('No. of images')

plt.ylabel('Particle concentration values')

plt.title('08/07/2020')

plt.show()

# from the graph, we can see the particle concentration values mainly lie in [200,300] on 08/07.

# it is consistent with the result that the average particle concentration is 250.503498 on 08/07.
# summarize the date-based train data properties in one data frame

date_based_train = train_image.groupby('Date').agg([np.mean,np.std,np.min,np.max])

date_based_train
# extract the total particle concentration data based on different dates

date_based_train_total = date_based_train['Total']

date_based_train_total
# visualize the total particle concentration on different dates

total_concentration_mean = date_based_train_total['mean']

total_concentration_min = date_based_train_total['amin']

total_concentration_max = date_based_train_total['amax']

total_concentration_mean.plot(xlabel='Date',ylabel='Total Particle Concentration',title='Date-based Particle Concentration',legend='mean')

total_concentration_min.plot(legend='min')

total_concentration_max.plot(legend='max')
# take one image form the image files as an example & extract its image features

# first of all, read the randomly selected image "video06182020_1271.jpg"

img1 = PIL.Image.open('/kaggle/input/pollutionvision/frames/frames/video06182020_1271.jpg').convert('RGB')
# show "img1"

plt.imshow(img1)
# export RGB & Luminance for img

imnp = np.array(img1)

r = imnp[:, :, 0]

g = imnp[:, :, 1]

b = imnp[:, :, 2]

r_avg = r.mean()/225

g_avg = g.mean()/225

b_avg = b.mean()/225

print(r_avg,g_avg,b_avg)
# export Luminance for img

img2 = img1.convert('L')

lum_avg = ImageStat.Stat(img2).mean[0]

print(lum_avg)
# export Blueness for img (as defined in midterm)

blue = b_avg/lum_avg

print(blue)