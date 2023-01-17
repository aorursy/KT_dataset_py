from IPython.display import YouTubeVideo

YouTubeVideo('QdwuHKwOLRU', width=800, height=300)

!pip install tensorflow_io
import pandas as pd 

import os

import matplotlib.pyplot as plt 

import numpy as np 

import tensorflow as tf

import tensorflow_io as tfio 

import matplotlib.image as mpimg

import seaborn as sns 

import pydicom

sns.set_palette("bright")
df=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
(sns.countplot(df.Sex)).figure.savefig("output1.png")

fig, axs = plt.subplots(ncols=3)

fig.set_size_inches(19,6)

sns.countplot(df['SmokingStatus'],ax=axs[0])

sns.countplot(df['SmokingStatus'][df['Sex']=="Male"],ax=axs[1])

sns.countplot(df['SmokingStatus'][df['Sex']=="Female"],ax=axs[2])

fig.savefig("output2.jpeg")
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.distplot(df.Age,kde=False,bins=80,color="k")

fig.savefig("output3.jpeg")
def create_last_scan_df():

    last_test=pd.DataFrame()    

    for i in df.Patient.unique():

        last_test=last_test.append((df[df['Patient']=="{}".format(i)][-1:]))

    last_test=last_test.drop("Patient",axis=1)

    last_test=last_test.drop("Weeks",axis=1)

    return last_test

dd=create_last_scan_df()

dd=dd.reset_index(drop=True)

dd.head()
def create_baseline():

    first_scan=pd.DataFrame()    

    for i in df.Patient.unique():

        first_scan=first_scan.append((df[df['Patient']=="{}".format(i)][:1]))

    first_scan=first_scan.drop("Patient",axis=1)

    first_scan=first_scan.drop("Weeks",axis=1)

    return first_scan

fc=create_baseline()

fc=fc.reset_index(drop=True)

fc.head()
(sns.pairplot(df,hue="SmokingStatus",height=4)).savefig("output4.jpeg")

sns.pairplot(fc,hue="SmokingStatus",height=4).savefig("output5.jpeg")
(sns.pairplot(dd,hue="Sex",height=4)).savefig("output6.jpeg")
(sns.pairplot(dd,hue="SmokingStatus",height=4)).savefig("output7.jpeg")

(sns.pairplot(fc,hue="Sex",height=4)).savefig("output8.jpeg")

ex_smoker_male=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Male") & (df["SmokingStatus"]=="Ex-smoker"),'Patient'][:1].values[0]))]

ex_smoker_female=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Female") & (df["SmokingStatus"]=="Ex-smoker"),'Patient'][:1].values[0]))]

non_smoker_male=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Male") & (df["SmokingStatus"]=="Never smoked"),'Patient'][:1].values[0]))]

non_smoker_female=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Female") & (df["SmokingStatus"]=="Never smoked"),'Patient'][:1].values[0]))]

current_smoker_male=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Male") & (df["SmokingStatus"]=="Currently smokes"),'Patient'][:1].values[0]))]

current_smoker_female=df.loc[df['Patient']=='{}'.format((df.loc[(df["Sex"]=="Female") & (df["SmokingStatus"]=="Currently smokes"),'Patient'][:1].values[0]))]
fig, ax = plt.subplots(nrows=2)

fig.set_size_inches(22, 8.27)

sns.lineplot(x='Weeks',y='Percent',data=df,ax=ax[0])

sns.lineplot(x='Weeks',y='FVC',data=df,ax=ax[1])

fig.savefig("weeksvsfvc.jpeg")

males=df[df["Sex"]=="Male"]

females=df[df["Sex"]=="Female"]
fig, ax = plt.subplots(nrows=4)

fig.set_size_inches(22, 22)

sns.lineplot(x='Weeks',y='FVC',data=males,ax=ax[0]).set_title("MALES FVC TREND")

sns.lineplot(x='Weeks',y='FVC',data=females,ax=ax[1]).set_title("FEMALE FVC TREND")

sns.lineplot(x='Weeks',y='Percent',data=males,ax=ax[2]).set_title("MALES PERCENT TREND")

sns.lineplot(x='Weeks',y='Percent',data=females,ax=ax[3]).set_title("FEMALE PERCENT TREND")

fig.savefig("mvffvctrend.jpeg")

smoker=df[df["SmokingStatus"]=="Ex-smoker"]

never_smoked=df[df["SmokingStatus"]=="Never smoked"]

current_smoker=df[df["SmokingStatus"]=="Currently smokes"]
fig, ax = plt.subplots(nrows=6)

fig.set_size_inches(22, 35)

sns.lineplot(x='Weeks',y='FVC',data=smoker,ax=ax[0]).set_title("EX SMOKER FVC TREND")

sns.lineplot(x='Weeks',y='FVC',data=never_smoked,ax=ax[1]).set_title("NON SMOKER FVC TREND")

sns.lineplot(x='Weeks',y='FVC',data=current_smoker,ax=ax[2]).set_title("SMOKER FVC TREND")

sns.lineplot(x='Weeks',y='Percent',data=smoker,ax=ax[3]).set_title("EX SMOKER PERCENT  TREND")

sns.lineplot(x='Weeks',y='Percent',data=never_smoked,ax=ax[4]).set_title("NON SMOKER PERCENT TREND")

sns.lineplot(x='Weeks',y='Percent',data=current_smoker,ax=ax[5]).set_title("SMOKER PERCENT TREND")

fig.savefig("weeksvpercent.jpeg")

fig, ax = plt.subplots()

fig.set_size_inches(22,5)

sns.lineplot(x=ex_smoker_male["Weeks"], y=ex_smoker_male["FVC"],label='ex_smoker_male')

sns.lineplot(x=ex_smoker_female["Weeks"], y=ex_smoker_female["FVC"],label='ex_smoker_female')

sns.lineplot(x=non_smoker_male["Weeks"], y=non_smoker_male["FVC"],label='non_smoker_male')

sns.lineplot(x=non_smoker_female["Weeks"], y=non_smoker_female["FVC"],label='non_smoker_female')

sns.lineplot(x=current_smoker_male["Weeks"], y=current_smoker_male["FVC"],label='current_smoker_male')

sns.lineplot(x=current_smoker_female["Weeks"], y=current_smoker_female["FVC"],label='current_smoker_female')

fig.savefig("smoker_current_fvc.jpeg")

fig, ax = plt.subplots()

fig.set_size_inches(22,5)

sns.lineplot(x=ex_smoker_male["Weeks"], y=ex_smoker_male["Percent"],label='ex_smoker_male')

sns.lineplot(x=ex_smoker_female["Weeks"], y=ex_smoker_female["Percent"],label='ex_smoker_female')

sns.lineplot(x=non_smoker_male["Weeks"], y=non_smoker_male["Percent"],label='non_smoker_male')

sns.lineplot(x=non_smoker_female["Weeks"], y=non_smoker_female["Percent"],label='non_smoker_female')

sns.lineplot(x=current_smoker_male["Weeks"], y=current_smoker_male["Percent"],label='current_smoker_male')

sns.lineplot(x=current_smoker_female["Weeks"], y=current_smoker_female["Percent"],label='current_smoker_female')

fig.savefig("sdad.jpeg")

files=[]

for dirname, _, filenames in os.walk('../input/osic-pulmonary-fibrosis-progression/train'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))
def decode_image(image_path):

    image_bytes = tf.io.read_file(image_path)

    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

    image=np.squeeze(image.numpy())

    return image 
def show_scan(image):

    img = decode_image(image)

    patient_name=str(image).split('/')[1]

    fig, ax = plt.subplots()

    im=ax.imshow(img,cmap='twilight_r')

    plt.axis('off')

    plt.title("Baseline CT Scan of Patient {}".format(patient_name))

    fig.set_size_inches(9,9)

    plt.show()
show_scan(files[3])
def choose_patient(ID):

    images_x=[]

    for i in files:

        name=str(i)

        if str(ID) in name:

            images_x.append(i)

    return sorted(images_x)



def generate_images(images):

    for x,i in enumerate(images):

        image=decode_image(i)   

        fname=str(x)+".png"

        plt.imsave(fname,image,cmap='twilight_shifted')    

        

def make_progressive_video():

    os.system("ffmpeg  -r 30 -i %d.png -vcodec mpeg4 -y -vb 400M patient_ct_scan_progression.mp4")
patient_x=choose_patient(df['Patient'][288])

generate_images(patient_x)

make_progressive_video()
!rm -r *.png
!ffmpeg -i patient_ct_scan_progression.mp4 out.gif