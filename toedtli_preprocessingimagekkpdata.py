import sklearn

import skimage

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from glob import glob

import os.path

from skimage import data, io

from matplotlib import pyplot as plt



print('loaded modules')

%matplotlib inline
! ls /kaggle/input/

!pwd
!ls /kaggle/input/daan-kreuz-kreis-plus/
!pwd
!tar -xvf /kaggle/input/daan-kreuz-kreis-plus/train.tar.gz # Trainingsbilder

!tar -xvf /kaggle/input/daan-kreuz-kreis-plus/data.tar.gz #Testbilder
!ls Bilder
%mkdir -p ../output

!ls /kaggle/working/Bilder/
path = '/kaggle/working/Bilder'

trainpath= os.path.join(path,'train')

testpath= os.path.join(path,'test')



outputpath = '../output'
y_train = pd.read_csv(os.path.join(path,'target_info_train.txt'),index_col='id')
#teste imread



im = skimage.io.imread(os.path.join(trainpath,r'0001-u010.png'))

#im = skimage.io.imread('Bilder/test/0887-u026.png')

im.shape
# target_info_train.txt enthält label-Werte 0,1,2

glob(os.path.join(trainpath,'*.png'))[:10]

with open(os.path.join(path,'target_info_train.txt'),'r') as fh:

    for i in range(10):

        line = fh.readline()

        print(line.strip())
plt.imshow(im[:,:],cmap='gray')

plt.title('Offenbar haben die Bilder Graustufen und Farbkanäle')

plt.colorbar();
mins = []

maxes=[]

imdict = dict()

#for fn in glob('Bilder/train/*.png')+glob('Bilder/test/*.png'):

for fn in glob(os.path.join(trainpath,'*'))+glob(os.path.join(testpath,'*')):

    im = skimage.io.imread(fn)

    imdict[fn]=im

len(imdict)
list(imdict.keys())[:5]
shapes = []

for fn,im in imdict.items():

    shapes.append(im.shape)

pd.Series(shapes).unique()[:10] #zeige nur die ersten 10 unterschiedlichen shapes.
seen4=0

for fn,im in imdict.items():

    if len(im.shape)>2:

        if im.shape[2]==4:

            seen4+=1

            if seen4==5: #5. Bild mit 4 Ebenen

                plt.subplot(2,2,1)

                plt.imshow(im[:,:,0])

                plt.subplot(2,2,2)

                plt.imshow(im[:,:,1])

                plt.subplot(2,2,3)

                plt.imshow(im[:,:,2])

                plt.subplot(2,2,4)

                plt.imshow(im[:,:,3])

                plt.suptitle(fn)

print('done.')
im.shape
grayimdict = dict()



for fn,im in imdict.items():

    if len(im.shape)!=2:

        assert len(im.shape)<=3,'unerwarteter Fall'

        im=np.mean(im,axis=2)

    grayimdict[fn]=im
plt.imshow(grayimdict['/kaggle/working/Bilder/train/0507-u029.png'],cmap='gray')

plt.colorbar()
plt.imshow(grayimdict['/kaggle/working/Bilder/train/0446-u032.png'],cmap='gray')

plt.colorbar();
for fn,im in grayimdict.items():

    mins.append(np.min(im))

    maxes.append(np.max(im))
ax1 = plt.subplot(1,2,1)

plt.hist(mins,bins=[0,256,1000,10000,100000]);plt.title('pixel minima')

ax1.set_xscale("log", nonposx='clip')

ax2 = plt.subplot(1,2,2)

plt.hist(maxes,bins=[0,256,1000,10000,100000]);plt.title('pixel maxima')

ax2.set_xscale("log", nonposx='clip')

#ax2.set_yscale("log", nonposy='clip')
scaledimdict = {}

for fn,im in grayimdict.items():

    imscaled = (im - np.min(im))/(np.max(im)-np.min(im))

    scaledimdict[fn]=imscaled
list(scaledimdict.keys())[:5]
from skimage import data, io

from matplotlib import pyplot as plt



io.imshow(scaledimdict['/kaggle/working/Bilder/train/0129-u021.png'])

plt.show()
scaledimdictflipped = scaledimdict.copy()

perc_list = []

for fn,im in scaledimdict.items():

    curr_perc = np.percentile(im.ravel(),50)

    if curr_perc > 0.5: #Falls die Mehrheit der Pixel hell ist, flippe schwarz <-> weiss       

        scaledimdictflipped[fn] = 1-im

        new_perc = np.percentile(scaledimdictflipped[fn].ravel(),50)

    perc_list.append(curr_perc)
plt.hist(perc_list);
import skimage.transform

resizedimdict=dict()

for fn,im in scaledimdictflipped.items():

    im = skimage.transform.resize(im,(15,15),

              #Diese zwei Optionen sind nötig, um keine Warung mehr zu erhalten:

              anti_aliasing=True, 

              mode='reflect')

    resizedimdict[fn]=im
plt.imshow(im,cmap='gray')

plt.colorbar();
pixelList = []

fnamelist = []

for fn,im in resizedimdict.items():

    pixelList.append(im.ravel().tolist())

    fnamelist.append(fn)

pixelArr = np.array(pixelList)

plt.imshow(pixelArr)

plt.gca().set_aspect('auto');plt.xlabel('Pixelindex');plt.ylabel('Bildindex')



pixelArr.shape #Anz. Bilder x Anz Pixel (15x15)
#Verteilung aller Pixelwerte: 

plt.hist(pixelArr.ravel(),bins=50);
import pandas as pd

cols = ['pix'+str(i) for i in range(15**2)]

df = pd.DataFrame(pixelArr,columns=cols,index=fnamelist)

df.shape
AnzProblemPixel = ((df>0.05) & (df < 0.1)).sum(axis=1)

AnzProblemPixel.hist(bins=20),plt.title('Verteilung der Anzahl "Problempixel"');
problemBilder = df[AnzProblemPixel>20]
#z.B. 

testbild = problemBilder.sample()

plt.imshow(resizedimdict[testbild.index[0]],cmap='gray');
#Wende den Schwellwert an: 

dfbin = df.copy()

dfbin[df>0.1]=1

dfbin[df<=0.1]=0

dfbin = dfbin.astype('int')
#Überprüfe qualitativ, indem wir alle Bilder in einem zusammenfassen. Oder auch einfach, weil's hübsch aussieht...

plt.imshow(dfbin,cmap='gray')

plt.gca().set_aspect('auto');plt.xlabel('Pixelindex');plt.ylabel('Bildindex')
im1 = dfbin.sample(1).values.reshape(15,15);plt.subplot(1,3,1);plt.imshow(im1);

im2 = dfbin.sample(1).values.reshape(15,15);plt.subplot(1,3,2);plt.imshow(im2);

im3 = dfbin.sample(1).values.reshape(15,15);plt.subplot(1,3,3);plt.imshow(im3);
def ticksoff():

    plt.xticks([]),plt.yticks([])

plt.figure(1,figsize=(15,10))

im1 = dfbin.iloc[1].values.reshape(15,15);plt.subplot(1,3,2);plt.imshow(im1);ticksoff()

im2 = dfbin.iloc[2].values.reshape(15,15);plt.subplot(1,3,1);plt.imshow(im2);ticksoff()

im3 = dfbin.iloc[4].values.reshape(15,15);plt.subplot(1,3,3);plt.imshow(im3);ticksoff()
dfy = pd.read_csv('/kaggle/working/Bilder/target_info_train.txt',index_col='id')

dfy.sort_index(inplace=True) #wohl nicht nötig...



dfy.tail()
#dfbin beginnt mit training-Bildern, dann Testbildern

display(dfbin.head())

display(dfbin.tail())
#extrahiere den Dateinamen aus dem Index:

image_index = dfbin.index

from os.path import basename

import re

#re.sub?

dfbin2 = dfbin.reset_index(inplace=False);



basenames = pd.Series([os.path.basename(fn) for fn in image_index])

dfbin2['basename'] = basenames

image_index = basenames.replace(to_replace='-u\d*.png',value='',regex=True).map(lambda x:int(x))

image_index[:5]
dfbin2.rename(columns={'index':'Dateiname'},inplace=True)

dfbin2['image_index']=image_index

dfbin2 = dfbin2.set_index('image_index',inplace=False).sort_index()

dfbin2.head()
np.all(dfbin2.index==np.arange(1430)), np.all(dfy.index==np.arange(715))
dfy.index
dfbin2.tail(2)
#inner join:

df_train = pd.merge(dfbin2, dfy, left_index=True, right_index=True)

display(df_train.head(2))

df_train.tail(2)
dftemp = dfbin2.join(dfy)

df_test = dftemp[dftemp.target.isnull()]

display(df_test.head(2))

df_test.tail(2)
df_train.target.unique()
#Check! nehme 5 zufällige Bilder: stimmen die Label mit den Bildern überein?

sample = df_train.sample(5)

sample.index
sample = df_train.sample(5)

d={0:'Kreuz',1:'Kreis',2:'Plus'}

for iframe,index in enumerate(sample.index):

    plt.subplot(1,5,iframe+1)

    im = sample.loc[index].iloc[1:-2].values.reshape(15,15).astype('int')

    target = sample.loc[index].target

    plt.imshow(im)

    plt.title('{0}'.format(d[target]))
if 'Dateiname' in df_train.columns: del df_train['Dateiname']

if 'Dateiname' in df_test.columns: del df_test['Dateiname']

for coln in df_test.columns:print(coln)
df_train.set_index('basename',inplace=True)

df_train.head(5)
del df_test['target']
df_test.set_index('basename',inplace=True)

df_test.head()
if 'target' in df_test.columns:

    del df_test['target']



df_train.to_csv('KreuzKreisPlus_train.csv',index=False)

df_test.to_csv('KreuzKreisPlus_test.csv',index=False)
!ls .

!pwd
#funktioniert bei mir nicht

from IPython.display import FileLink,FileLinks

#FileLink('/kaggle/output/KreuzKreisPlus_train.csv')

FileLinks('/kaggle/output/')

#!head {outputpath}/*
from IPython.display import Javascript

def download_file(fn):

    js_download = """

    var csv = '%s';



    var filename = '{fn}';

    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });

    if (navigator.msSaveBlob) { // IE 10+

    navigator.msSaveBlob(blob, filename);

    } else {

    var link = document.createElement("a");

    if (link.download !== undefined) { // feature detection

    // Browsers that support HTML5 download attribute

    var url = URL.createObjectURL(blob);

    link.setAttribute("href", url);

    link.setAttribute("download", filename);

    link.style.visibility = 'hidden';

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    }

    }

    """

    return Javascript(js_download)

#download_file('KreuzKreisPlus_train.csv')

#download_file('KreuzKreisPlus_test.csv')
!ls {outputpath}

!ls /kaggle/output
Xtrain = df_train.values[:,:-1]
from sklearn.decomposition import PCA

Xtrain
clf = PCA(n_components=5)

clf.fit_transform(Xtrain,None)
plt.figure(1,figsize=(15,20))

plt.subplot(2,3,1),plt.imshow(clf.components_[0].reshape(-1,15));

plt.subplot(2,3,2),plt.imshow(clf.components_[1].reshape(-1,15));

plt.subplot(2,3,3),plt.imshow(clf.components_[2].reshape(-1,15));

plt.subplot(2,3,4),plt.imshow(clf.components_[3].reshape(-1,15));

plt.subplot(2,3,5),plt.imshow(clf.components_[4].reshape(-1,15));