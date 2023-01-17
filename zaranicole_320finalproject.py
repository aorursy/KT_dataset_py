# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, array_to_img

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.w

tf.__version__
df_attr = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')

df_attr
df_attr.shape
import seaborn as sns



image_src = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'

image_id = '121410.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]
image_id = '121123.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]
image_id = '122001.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]
image_id = '121399.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]
image_id = '114511.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]
image_id = '121420.jpg'

img = load_img(image_src+image_id)

plt.grid(False)

plt.imshow(img)

df_attr.loc[df_attr['image_id'] == image_id]

# Young or Old?

sns.set(style="darkgrid")

plt.title('Young or Old?')

sns.countplot(y='Young', data=df_attr, color="b")

plt.show()
# Attractive

plt.title('Attractive or Not')

sns.countplot(y='Attractive', data=df_attr, color="g")

plt.show()
# Gender

plt.title('Female or Male')

sns.countplot(y='Male', data=df_attr, color="c")

plt.show()
# Gender

plt.title('Black Hair')

sns.countplot(y='Black_Hair', data=df_attr, color="r")

plt.show()
import seaborn as sns





sns.set(style="darkgrid")

plt.title('Attractive?')

sns.countplot(x='Attractive', data=df_attr, color="b")



att = 0; 

notAtt = 0; 



eyebrowsA = 0; 

bagsA = 0; 

baldA = 0; 

bangsA = 0; 

lipsA = 0; 

noseA = 0; 

blackhairA = 0; 

sideburnsA = 0; 

smileA = 0; 

straighthair = 0; 

wavyhair = 0; 

earrings = 0; 

hat = 0; 

lipstick = 0; 

necklace = 0; 

necktie = 0; 

young = 0; 



eyebrowsN = 0; 

bagsN = 0; 

baldN = 0; 

bangsN = 0; 

lipsN = 0; 

noseN = 0; 

blackhairN = 0; 

sideburnsN = 0; 

smileN = 0; 

straighthairN = 0; 

wavyhairN = 0; 

earringsN = 0; 

hatN = 0; 

lipstickN = 0; 

necklaceN = 0; 

necktieN = 0; 

youngN = 0; 



for index, row in df_attr.iterrows(): 

    # index => celebrity, row => attributes 

    if (row["Attractive"] == 1):

        att += 1 

        if (row['Arched_Eyebrows'] == 1): 

            eyebrowsA += 1 

        if (row['Bags_Under_Eyes'] == 1): 

            bagsA += 1 

        if (row['Bald'] == 1): 

            baldA += 1 

        if (row['Bangs'] == 1): 

            bangsA += 1 

        if (row['Big_Lips'] == 1): 

            lipsA += 1 

        if (row['Big_Nose'] == 1): 

            noseA += 1 

        if (row['Black_Hair'] == 1): 

            blackhairA += 1 

        if (row['Sideburns'] == 1): 

            sideburnsA += 1 

        if (row['Smiling'] == 1): 

            smileA += 1 

        if (row['Straight_Hair'] == 1): 

            straighthair += 1 

        if (row['Wavy_Hair'] == 1): 

            wavyhair += 1 

        if (row['Wearing_Earrings'] == 1): 

            earrings += 1 

        if (row['Wearing_Hat'] == 1): 

            hat += 1 

        if (row['Wearing_Lipstick'] == 1): 

            lipstick += 1 

        if (row['Wearing_Necklace'] == 1): 

            necklace += 1 

        if (row['Wearing_Necktie'] == 1): 

            necktie += 1 

        if (row['Young'] == 1): 

            young += 1 

    else: 

        if (row['Arched_Eyebrows'] == 1): 

            eyebrowsN += 1 

        if (row['Bags_Under_Eyes'] == 1): 

            bagsN += 1 

        if (row['Bald'] == 1): 

            baldN += 1 

        if (row['Bangs'] == 1): 

            bangsN += 1 

        if (row['Big_Lips'] == 1): 

            lipsN += 1 

        if (row['Big_Nose'] == 1): 

            noseN += 1 

        if (row['Black_Hair'] == 1): 

            blackhairN += 1 

        if (row['Sideburns'] == 1): 

            sideburnsN += 1 

        if (row['Smiling'] == 1): 

            smileN += 1 

        if (row['Straight_Hair'] == 1): 

            straighthairN += 1 

        if (row['Wavy_Hair'] == 1): 

            wavyhairN += 1 

        if (row['Wearing_Earrings'] == 1): 

            earringsN += 1 

        if (row['Wearing_Hat'] == 1): 

            hatN += 1 

        if (row['Wearing_Lipstick'] == 1): 

            lipstickN += 1 

        if (row['Wearing_Necklace'] == 1): 

            necklaceN += 1 

        if (row['Wearing_Necktie'] == 1): 

            necktieN += 1 

        if (row['Young'] == 1): 

            youngN += 1 



notAtt = 202599- att; 



# attractive ratios

eyebrowsA = eyebrowsA / att

bagsA = bagsA / att

baldA = baldA / att

bangsA = bangsA / att

lipsA = lipsA / att

noseA = noseA / att

blackhairA = blackhairA / att

sideburnsA = sideburnsA / att

smileA = smileA / att

straighthair = straighthair / att

wavyhair = wavyhair / att

earrings = earrings / att

hat = hat / att

lipstick = lipstick / att 

necklace = necklace / att

necktie = necktie / att

young = young / att 



# not attractive ratios 

eyebrowsN = eyebrowsN / notAtt

bagsN = bagsN / notAtt

baldN = baldN / notAtt

bangsN = bangsN / notAtt 

lipsN = lipsN / notAtt

noseN = noseN / notAtt

blackhairN = blackhairN / notAtt

sideburnsN = sideburnsN / notAtt

smileN = smileN / notAtt

straighthairN = straighthairN / notAtt

wavyhairN = wavyhairN / notAtt

earringsN = earringsN / notAtt

hatN = hatN / notAtt

lipstickN = lipstickN / notAtt 

necklaceN = necklaceN / notAtt

necktieN = necktieN / notAtt

youngN = youngN / notAtt 



# plot graph 

N = 17

ind = np.arange(N)  # the x locations for the groups

width = 0.25       # the width of the bars



fig = plt.figure()

ax = fig.add_subplot(111)



yvalsA = [eyebrowsA, bagsA, baldA, bangsA, lipsA, noseA, blackhairA, sideburnsA, smileA, straighthair, wavyhair, earrings, hat, lipstick, necklace, necktie, young]

#yvalsA = [eyebrowsA, bagsA, baldA, bangsA]

rects1 = ax.bar(ind, yvalsA, width, color='r')

#yvalsN = [1-eyebrowsA, 1-bagsA, 1-baldA, 1-bangsA, 1-lipsA, 1-noseA, 1-blackhairA, 1-sideburnsA, 1-smileA, 1-straighthair, 1-wavyhair, 1-earrings, 1-hat, 1-lipstick, 1-necklace, 1-necktie, 1-young]

yvalsN = [eyebrowsN, bagsN, baldN, bangsN, lipsN, noseN, blackhairN, sideburnsN, smileN, straighthairN, wavyhairN, earringsN, hatN, lipstickN, necklaceN, necktieN, youngN]

rects2 = ax.bar(ind+width, yvalsN, width, color='g') 



# Big_Lips	Big_Nose	Black_Hair	...	Sideburns	Smiling	Straight_Hair	Wavy_Hair	Wearing_Earrings	Wearing_Hat	Wearing_Lipstick	Wearing_Necklace	Wearing_Necktie	Young



ax.set_ylabel('% of Celebs with Each Feature') 

ax.set_xlabel('Feature')

ax.set_xticks(ind+width)

ax.set_xticklabels( ('ArchedEyebrows', 'BagsUnderEyes', 'Bald', 'Bangs', 'BigLips', 'BigNose', 'BlackHair', 'Sideburns', 'Smiling', 'StraightHair', 'WavyHair', 'Earrings', 'WearingHat', 'Lipstick', 'Necklace', 'Necktie', 'Young') ) 

ax.legend( (rects1[0], rects2[0]), ('Attractive', 'Not Attractive') )



# ax = sns.countplot(x="Column", data=ds)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()



def autolabel(rects):

    for rect in rects:

        h = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2) 







df_partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')

df_partition.head()
df_partition['partition']=df_partition['partition'].replace(2,1)
df_partition['partition'].value_counts().sort_index()
# need 690 more in 0

# replace partition value from 162771 to 162771+690 with 0

df_partition.loc[162770]
df_partition.at[162769, 'partition'] = 1
i=162770;

while i>=162770-690:

    df_partition.at[i, 'partition'] = 1

    # print(i)

    i-=1
#df_partition.loc[162770]

df_partition.loc[162773]
df_partition['partition'].value_counts().sort_index()