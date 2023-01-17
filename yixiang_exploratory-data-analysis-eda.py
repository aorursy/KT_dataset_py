import pandas as pd 

import numpy as np

import pydicom

import matplotlib.pyplot as plt 

import re

import random
# data paths 

# (put img_paths into df table so can be sorted and join with patient_paths id)



# img_train_paths

img_train_paths_df = pd.DataFrame(columns=['id', 'img_path', 'path_num'])



img_train_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'



for folder in os.listdir(img_train_path):

    id_folder = str(folder)

    for file in os.listdir(img_train_path + folder):

        path_file = img_train_path + str(folder) + '/' + file

        path_num = int(str(file).replace('.dcm', '')) # cast to int

        img_train_paths_df = img_train_paths_df.append(

            {'id': id_folder, 'img_path': path_file, 'path_num': path_num}, ignore_index=True)



img_train_paths_df = img_train_paths_df.sort_values(

        by=['id', 'path_num'], axis=0) # IMPT: sort by id then path_num for sequential cross-section scan img for each id

img_train_paths_df = img_train_paths_df.drop(columns=['path_num'], axis=1) # drop path_num used for sort

img_train_paths_df = img_train_paths_df.reset_index(drop=True)

img_train_paths_df
# img_test_paths

img_test_paths_df = pd.DataFrame(columns=['id', 'img_path', 'path_num'])



img_test_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/test/'



for folder in os.listdir(img_test_path):

    id_folder = str(folder)

    for file in os.listdir(img_test_path + folder):

        path_file = img_test_path + str(folder) + '/' + file

        path_num = int(str(file).replace('.dcm', '')) # cast to int

        img_test_paths_df = img_test_paths_df.append(

            {'id': id_folder, 'img_path': path_file, 'path_num': path_num}, ignore_index=True)



img_test_paths_df = img_test_paths_df.sort_values(

        by=['id', 'path_num'], axis=0) # IMPT: sort by id then path_num for sequential cross-section scan img for each id

img_test_paths_df = img_test_paths_df.drop(columns=['path_num'], axis=1) # drop path_num used for sort

img_test_paths_df = img_test_paths_df.reset_index(drop=True)

img_test_paths_df
# patient train test paths

patient_train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

patient_test_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')



# IMPT: get first week of patient data only because CT scan images done on first week

patient_train_df = patient_train_df.groupby('Weeks').first().reset_index()

patient_test_df = patient_test_df.groupby('Weeks').first().reset_index()



patient_train_df = patient_train_df.drop(columns=['Weeks']) # drop Weeks

patient_test_df = patient_test_df.drop(columns=['Weeks'])



# rename Patient to 'id' for join

rename_columns = patient_train_df.columns.tolist()

rename_columns[0] = 'id'



patient_train_df.columns = rename_columns

patient_test_df.columns = rename_columns



patient_test_df
# change id string to float (not int as it will cause overflow) for faster join

img_train_paths_df['id'] = img_train_paths_df['id'].replace('ID', '', regex=True).astype(float)

img_test_paths_df['id'] = img_test_paths_df['id'].replace('ID', '', regex=True).astype(float)

patient_train_df['id'] = patient_train_df['id'].replace('ID', '', regex=True).astype(float)

patient_test_df['id'] = patient_test_df['id'].replace('ID', '', regex=True).astype(float)



patient_test_df
# left join subset patient_train_df with img_train_paths_df by id

X_train_df = patient_train_df.merge(img_train_paths_df, on=['id'], how='left') # merge joins on non-string columns vs join which throws error 

X_test_df = patient_test_df.merge(img_test_paths_df, on=['id'], how='left')



X_test_df
# plot lung cross-section scan of following categories:

# (1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'

# (2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'

# (3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'

# (4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'



# (1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'

category_1 = X_train_df[(X_train_df['Percent'] >= 70) & (X_train_df['SmokingStatus'] == 'Never smoked')] 



img_plt_df_1 = category_1

plt_index_1 = random.randint(0, img_plt_df_1.shape[0])

img_plt_df_1 = img_plt_df_1[img_plt_df_1['id'] == img_plt_df_1.iloc[plt_index_1]['id']]

img_plt_df_1
# (2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'

category_2 = X_train_df[(X_train_df['Percent'] >= 70) & (X_train_df['SmokingStatus'] == 'Ex-smoker')] 



img_plt_df_2 = category_2

plt_index_2 = random.randint(0, img_plt_df_2.shape[0])

img_plt_df_2 = img_plt_df_2[img_plt_df_2['id'] == img_plt_df_2.iloc[plt_index_2]['id']]

img_plt_df_2
# (3) Not healthy and smoke: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'

category_3 = X_train_df[(X_train_df['Percent'] <= 50) & (X_train_df['SmokingStatus'] == 'Ex-smoker')] 



img_plt_df_3 = category_3

plt_index_3 = random.randint(0, img_plt_df_3.shape[0])

img_plt_df_3 = img_plt_df_3[img_plt_df_3['id'] == img_plt_df_3.iloc[plt_index_3]['id']]

img_plt_df_3
# (4) Not healthy and smoke: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'

category_4 = X_train_df[(X_train_df['Percent'] <= 50) & (X_train_df['SmokingStatus'] == 'Never smoked')] 



img_plt_df_4 = category_4

plt_index_4 = random.randint(0, img_plt_df_4.shape[0])

img_plt_df_4 = img_plt_df_4[img_plt_df_4['id'] == img_plt_df_4.iloc[plt_index_4]['id']]

img_plt_df_4
# plot img_df



def plot_img_df(img_plt_df, title):

    f, plots = plt.subplots(2, 2, figsize=(15,15)) # set figsize to clear (15, 15)



    # get four cross sections: 1/4, 1/3, 1/2, 1/1.5

    ix_1 = int(img_plt_df.shape[0]/4)

    ix_2 = int(img_plt_df.shape[0]/3)

    ix_3 = int(img_plt_df.shape[0]/2)

    ix_4 = int(img_plt_df.shape[0]/1.5)



    img_1 = pydicom.read_file(img_plt_df['img_path'].values[ix_1])

    img_2 = pydicom.read_file(img_plt_df['img_path'].values[ix_2])

    img_3 = pydicom.read_file(img_plt_df['img_path'].values[ix_3])

    img_4 = pydicom.read_file(img_plt_df['img_path'].values[ix_4])



    img_1_arr = img_1.pixel_array

    img_2_arr = img_2.pixel_array

    img_3_arr = img_3.pixel_array

    img_4_arr = img_4.pixel_array



    plots[0, 0].set_title('1/4 from top', fontsize=15)

    plots[0, 1].set_title('1/3 from top', fontsize=15)

    plots[1, 0].set_title('1/2 from top', fontsize=15)

    plots[1, 1].set_title('1/1.5 from top', fontsize=15)



    print(title)



    plots[0, 0].imshow(img_1_arr, cmap='bone')

    plots[0, 1].imshow(img_2_arr, cmap='bone')

    plots[1, 0].imshow(img_3_arr, cmap='bone')

    plots[1, 1].imshow(img_4_arr, cmap='bone')
# (1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'

plot_img_df(img_plt_df_1, 

            title="(1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'")
# (2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'

plot_img_df(img_plt_df_2, 

            title="(2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'")
# (3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'

plot_img_df(img_plt_df_3, 

            title="(3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'")
# (4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'

plot_img_df(img_plt_df_4, 

            title="(4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'")
# make video of lung cross-section scan of patient

import matplotlib.animation as animation

from IPython.display import HTML

    

def create_img_video(img_plt_df):

    fig = plt.figure(figsize=(7, 7))



    imgs_anim = []

    for index,row in img_plt_df.iterrows():

        img = pydicom.read_file(row['img_path'])

        img_arr = img.pixel_array

        img_anim = plt.imshow(img_arr, animated=True, cmap=plt.cm.bone)

        plt.axis('off')

        imgs_anim.append([img_anim])



    anim = animation.ArtistAnimation(fig, imgs_anim, interval=25, blit=False, repeat_delay=1000)

    

    return anim
# (1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'

anim = create_img_video(img_plt_df_1) 

title = "(1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'"

print(title)

HTML(anim.to_html5_video())
# (2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'

anim = create_img_video(img_plt_df_2) 

title = "(2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'"

print(title)

HTML(anim.to_html5_video())
# (3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'

# 

# (as you can see, not healthy lung has larger areas of aveoli strands (NOT bronchi strands) for large vertical areas i.e. for longer video 

#  seconds vs healthy lung as the unhealhty aveolis are inflammed or enlarged and more spread out vs healthy)



anim = create_img_video(img_plt_df_3) 

title = "(3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'"

print(title)

HTML(anim.to_html5_video())
# (4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'

anim = create_img_video(img_plt_df_3) 

title = "(4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'"

print(title)

HTML(anim.to_html5_video())
# plot histogram of above categories of interest:

# (1) Healthy and never smoked (benchmark): a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Never smoked'

# (2) Healthy and smoked: a patient with FVC percentage='high' >= 70 (i.e. no pulmonary fibrosis), and SmokingStatus='Ex-smoker'

# (3) Not healthy and smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Ex-smoker'

# (4) Not healthy and never smoked: a patient with FVC percentage='low' <= 50 (i.e. have pulmonary fibrosis), and SmokingStatus='Never smoked'



category_df = X_train_df



category_df.loc[category_1.index,'category'] = '(1)' # IMPT: use 'category'.index not img_plt_df

category_df.loc[category_2.index,'category'] = '(2)'

category_df.loc[category_3.index,'category'] = '(3)'

category_df.loc[category_4.index,'category'] = '(4)'

category_df.loc[pd.isnull(category_df['category']) == True, 'category'] = 'Others'



plt.hist(category_df['category'])
# get % counts 



print("Category")

category_df['category'].value_counts() / category_df.shape[0] * 100
# plot histogram of variables



f, plots = plt.subplots(3, 2, figsize=(15, 15))



plots[0, 0].set_title('FVC', fontsize=15)

plots[0, 1].set_title('Percent', fontsize=15)

plots[1, 0].set_title('Age', fontsize=15)

plots[1, 1].set_title('Sex', fontsize=15)

plots[2, 0].set_title('SmokingStatus', fontsize=15)



plots[0, 0].hist(X_train_df['FVC'])

plots[0, 1].hist(X_train_df['Percent'])

plots[1, 0].hist(X_train_df['Age'])

plots[1, 1].hist(X_train_df['Sex'])

plots[2, 0].hist(X_train_df['SmokingStatus'])
# get % counts of discrete variables



print("Sex")

X_train_df['Sex'].value_counts() / X_train_df.shape[0] * 100
# get % counts of discrete variables



print("SmokingStatus")

X_train_df['SmokingStatus'].value_counts() / X_train_df.shape[0] * 100
# print mode

from scipy import stats



print("Mode FVC: ", stats.mode(X_train_df['FVC']))

print("Mode Percent: ", stats.mode(X_train_df['Percent']))

print("Mode Age: ", stats.mode(X_train_df['Age']))

print("Mode Sex: ", stats.mode(X_train_df['Sex']))

print("Mode SmokingStatus: ", stats.mode(X_train_df['SmokingStatus']))