import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr

from PIL import Image
df = pd.read_json('../input/train.json')

df.info()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import sent_tokenize
def description_sentiment(sentences):

    analyzer = SentimentIntensityAnalyzer()

    result = []

    for sentence in sentences:

        vs = analyzer.polarity_scores(sentence)

        result.append(vs)

    return pd.DataFrame(result).mean()



sdf = df.sample(5000)

sdf['description_tokens'] = sdf['description'].apply(sent_tokenize)

sdf = pd.concat([sdf,sdf['description_tokens'].apply(description_sentiment)],axis=1)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True,figsize=(8,16))

sns.violinplot(x="interest_level",y="compound",data=sdf,ax=ax1,order =['low','medium','high'])

sns.violinplot(x="interest_level",y="neg",data=sdf,ax=ax2)

sns.violinplot(x="interest_level",y="pos",data=sdf,ax=ax3)

sns.violinplot(x="interest_level",y="neu",data=sdf,ax=ax4)
# Get available images

from subprocess import check_output

images = [int(x) for x in check_output(["ls", "../input/images_sample"]).decode("utf8").strip().split('\n')]



# Read the train set and choose those which have images only

df = df[df.listing_id.isin(images)]

print(df.shape)



# Add number of images

df['n_images'] = df.apply(lambda x: len(x['photos']), axis=1)
# this is what we are after

check_output(["ls", "../input/images_sample/6812223"]).decode("utf8").strip().split('\n')
#function to process one image

def process_image(path):

    path = '../input/images_sample/'+path[0:7]+'/'+path

    im = np.array(Image.open(path))



    #get dims

    width = im.shape[1]

    height = im.shape[0]

    

    #flatten image

    im = im.transpose(2,0,1).reshape(3,-1)

   

    

    #brightness is simple, assign 1 if zero to avoid divide

    brg = np.amax(im,axis=0)

    brg[brg==0] = 1

    

    #hue, same, assign 1 if zero, not working atm due to arccos

    denom = np.sqrt((im[0]-im[1])**2-(im[0]-im[2])*(im[1]-im[2]))

    denom[denom==0] = 1

    #hue = np.arccos(0.5*(2*im[0]-im[1]-im[2])/denom)

    

    #saturation

    sat = (brg - np.amin(im,axis=0))/brg

    

    #return mean values

    return width,height,np.mean(brg),np.mean(sat)
#second helper function - process a row of a dataset

#return mean of each property for all images

def process_row(row):

    images = check_output(["ls", "../input/images_sample/"+str(row.listing_id)]).decode("utf8").strip().split('\n')

    res = np.array([process_image(x) for x in images])

    res = np.mean(res,axis=0)

    row['img_width'] = res[0]

    row['img_height'] = res[1]

    row['img_brightness'] = res[2]

    row['img_saturation'] = res[3]

    return row
#Now we can process the dataset

df = df.apply(lambda row: process_row(row),axis=1)
#Some plots

d = df[['img_width','n_images','img_height','img_brightness','img_saturation','interest_level']]

sns.pairplot(d, hue="interest_level",size=1.5)