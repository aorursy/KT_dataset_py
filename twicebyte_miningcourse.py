# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math as math # linear algebra

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

temperatures = pd.read_csv("../input/GlobalTemperatures.csv")

print(temperatures.head())
x = pd.to_datetime(temperatures.dt[:], format='%Y-%m-%d', errors='ignore').dt.year;

y = temperatures.LandAverageTemperature[:];

data = pd.concat([x,y], axis=1);

data = data.groupby(['dt']).mean()
plt.plot(data, 'ko')

plt.show()
x = pd.to_datetime(temperatures.dt[:], format='%Y-%m-%d', errors='ignore').dt.year;

y = temperatures.LandAverageTemperature[:];

s = ((y-y.mean())*(y-y.mean())).sum()/y.size

s = math.sqrt(s)

mi = y.mean()-s

ma = y.mean()+s

print (mi)

print (ma)

for (i,item) in enumerate(y):

    if (item<mi) or (item>ma):

        y[i]=(2*y[i+1]+y[i+2])/3;        

data = pd.concat([x,y], axis=1);

data = data.groupby(['dt']).mean()
plt.plot(data, 'ko')

plt.show()
tempByCity = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv")

print(tempByCity.head())
print(tempByCity[tempByCity.Country=="United States"])
plt.figure()

x = pd.to_datetime(tempByCity.dt[:], format='%Y-%m-%d', errors='ignore').dt.year

plt.subplot(221) 

plt.title('Before 1900')

plt.boxplot([tempByCity.AverageTemperature[x<1900][tempByCity.Country=="Russia"][tempByCity.City=="Moscow"].dropna().values,

             tempByCity.AverageTemperature[x<1900][tempByCity.Country=="United States"][tempByCity.City=="New York"].dropna().values,

            tempByCity.AverageTemperature[x<1900][tempByCity.Country=="Russia"][tempByCity.City=="Saint Petersburg"].dropna().values]

            ,0,'kd',1,1)

plt.axis([0,4,-25, 35])

plt.xticks([1, 2, 3], ['MOW', 'NYC', 'LED'])

plt.subplot(222) 

plt.title('1900-2000')

plt.boxplot([tempByCity.AverageTemperature[x>=1900][x<2000][tempByCity.Country=="Russia"][tempByCity.City=="Moscow"].dropna().values,

             tempByCity.AverageTemperature[x>=1900][x<2000][tempByCity.Country=="United States"][tempByCity.City=="New York"].dropna().values,

            tempByCity.AverageTemperature[x>=1900][x<2000][tempByCity.Country=="Russia"][tempByCity.City=="Saint Petersburg"].dropna().values]

            ,0,'kd',1,1)

plt.axis([0,4,-25, 35])

plt.xticks([1, 2, 3], ['MOW', 'NYC', 'LED'])

plt.subplot(223) 

plt.title('2000-2008')

plt.boxplot([tempByCity.AverageTemperature[x>=2000][x<2008][tempByCity.Country=="Russia"][tempByCity.City=="Moscow"].dropna().values,

             tempByCity.AverageTemperature[x>=2000][x<2008][tempByCity.Country=="United States"][tempByCity.City=="New York"].dropna().values,

            tempByCity.AverageTemperature[x>=2000][x<2008][tempByCity.Country=="Russia"][tempByCity.City=="Saint Petersburg"].dropna().values]

            ,0,'kd',1,1)

plt.axis([0,4,-25, 35])

plt.xticks([1, 2, 3], ['MOW', 'NYC', 'LED'])

plt.subplot(224) 

plt.title('After 2008')

plt.boxplot([tempByCity.AverageTemperature[x>=2008][tempByCity.Country=="Russia"][tempByCity.City=="Moscow"].dropna().values,

             tempByCity.AverageTemperature[x>=2008][tempByCity.Country=="United States"][tempByCity.City=="New York"].dropna().values,

            tempByCity.AverageTemperature[x>=2008][tempByCity.Country=="Russia"][tempByCity.City=="Saint Petersburg"].dropna().values]

            ,0,'kd',1,1)

plt.axis([0,4,-25, 35])

plt.xticks([1, 2, 3], ['MOW', 'NYC', 'LED'])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4,

                    wspace=0.3)
from pylab import *



def cface(ax, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18):

    # x1 = height  of upper face

    # x2 = overlap of lower face

    # x3 = half of vertical size of face

    # x4 = width of upper face

    # x5 = width of lower face

    # x6 = length of nose

    # x7 = vertical position of mouth

    # x8 = curvature of mouth

    # x9 = width of mouth

    # x10 = vertical position of eyes

    # x11 = separation of eyes

    # x12 = slant of eyes

    # x13 = eccentricity of eyes

    # x14 = size of eyes

    # x15 = position of pupils

    # x16 = vertical position of eyebrows

    # x17 = slant of eyebrows

    # x18 = size of eyebrows

    

    # transform some values so that input between 0,1 yields variety of output

    x3 = 1.9*(x3-.5)

    x4 = (x4+.25)

    x5 = (x5+.2)

    x6 = .3*(x6+.01)

    x8 = 5*(x8+.001)

    x11 /= 5

    x12 = 2*(x12-.5)

    x13 += .05

    x14 += .1

    x15 = 1.5*(x15-.5)

    x16 = .25*x16

    x17 = 1.0*(x17-.5)

    x18 = .5*(x18+.1)



    # top of face, in box with l=-x4, r=x4, t=x1, b=x3

    e = mpl.patches.Ellipse( (0,(x1+x3)/2), 2*x4, (x1-x3), fc='white', linewidth=2)

    ax.add_artist(e)



    # bottom of face, in box with l=-x5, r=x5, b=-x1, t=x2+x3

    e = mpl.patches.Ellipse( (0,(-x1+x2+x3)/2), 2*x5, (x1+x2+x3), fc='white', linewidth=2)

    ax.add_artist(e)



    # cover overlaps

    e = mpl.patches.Ellipse( (0,(x1+x3)/2), 2*x4, (x1-x3), fc='white', ec='none')

    ax.add_artist(e)

    e = mpl.patches.Ellipse( (0,(-x1+x2+x3)/2), 2*x5, (x1+x2+x3), fc='white', ec='none')

    ax.add_artist(e)

    

    # draw nose

    plot([0,0], [-x6/2, x6/2], 'k')

    

    # draw mouth

    p = mpl.patches.Arc( (0,-x7+.5/x8), 1/x8, 1/x8, theta1=270-180/pi*arctan(x8*x9), theta2=270+180/pi*arctan(x8*x9))

    ax.add_artist(p)

    

    # draw eyes

    p = mpl.patches.Ellipse( (-x11-x14/2,x10), x14, x13*x14, angle=-180/pi*x12, facecolor='grey')

    ax.add_artist(p)

    p = mpl.patches.Ellipse( (-x11-x14/2,x10+x15*x13*x14/4), x14/2, x13*x14/2, angle=-180/pi*x12, facecolor='lightskyblue')

    ax.add_artist(p)

    p = mpl.patches.Ellipse( (x11+x14/2,x10), x14, x13*x14, angle=180/pi*x12, facecolor='grey')

    ax.add_artist(p)

    p = mpl.patches.Ellipse( (x11+x14/2,x10+x15*x13*x14/4), x14/2, x13*x14/2, angle=180/pi*x12, facecolor='lightskyblue')

    ax.add_artist(p)

    # draw pupils

    p = mpl.patches.Ellipse( (-x11-x14/2, x10+x15*x13*x14/2), .05, .05, facecolor='black')

    ax.add_artist(p)

    p = mpl.patches.Ellipse( (x11+x14/2, x10+x15*x13*x14/2), .05, .05, facecolor='black')

    ax.add_artist(p)

    

    # draw eyebrows

    plot([-x11-x14/2-x14*x18/2,-x11-x14/2+x14*x18/2],[x10+x13*x14*(x16+x17),x10+x13*x14*(x16-x17)],'k')

    plot([x11+x14/2+x14*x18/2,x11+x14/2-x14*x18/2],[x10+x13*x14*(x16+x17),x10+x13*x14*(x16-x17)],'k')



fig = figure(figsize=(11,11))



COUNTRIES = ["Russia","United States","Egypt"]

CITIES = ["Moscow","New York","Cairo"]

maxmean = tempByCity.AverageTemperature.dropna().groupby([tempByCity.Country, tempByCity.City]).mean().max()

minmean = tempByCity.AverageTemperature.dropna().groupby([tempByCity.Country, tempByCity.City]).mean().min()

maxmax = tempByCity.AverageTemperature.dropna().max()

minmin = tempByCity.AverageTemperature.dropna().min()



for i in range(3):

    me = (tempByCity.AverageTemperature[tempByCity.Country==COUNTRIES[i]][tempByCity.City==CITIES[i]].dropna().mean())

    mi = (tempByCity.AverageTemperature[tempByCity.Country==COUNTRIES[i]][tempByCity.City==CITIES[i]].dropna().max())

    ma = (tempByCity.AverageTemperature[tempByCity.Country==COUNTRIES[i]][tempByCity.City==CITIES[i]].dropna().min())

    ax = fig.add_subplot(1,3,i+1,aspect='equal')

    cface(ax, 

          1, #1

          1, #2

          1, #3

          1, #4

          1, #5

          (me-mi)/(ma-mi), #6

          0.5, #7

          (me-minmean)/(maxmean-minmean), #8

          (me-minmean)/(maxmean-minmean), #9

          0.5, #10

          0.5, #11

          0.5, #12

          1.5*(ma - mi)/(maxmax-minmin), #13

          0.1, #14

          0.5, #15

          -3, #16

          0.5, #17

          2) #18

    ax.axis([-1.2,1.2,-1.2,1.2])

    ax.set_xticks([])

    ax.set_yticks([])



fig.subplots_adjust(hspace=0, wspace=0)