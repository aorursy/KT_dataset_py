import numpy as np
import pandas as pd
import missingno as msn 
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pylab as plt
%pylab inline

PATH = '../input/'
label = {
"0" : "Nucleoplasm", 
"1" : "Nuclear membrane",   
"2" : "Nucleoli", 
"3" : "Nucleoli fibrillar center",   
"4" : "Nuclear speckles",   
"5" : "Nuclear bodies",   
"6" : "Endoplasmic reticulum",   
"7" : "Golgi apparatus",   
"8" : "Peroxisomes",   
"9" : "Endosomes",   
"10" : "Lysosomes",   
"11" : "Intermediate filaments",   
"12" : "Actin filaments",   
"13" : "Focal adhesion sites",  
"14" : "Microtubules",   
"15" : "Microtubule ends",   
"16" : "Cytokinetic bridge",   
"17" : "Mitotic spindle",   
"18" : "Microtubule organizing center",   
"19" : "Centrosome",   
"20" : "Lipid droplets",   
"21" : "Plasma membrane",   
"22" : "Cell junctions",   
"23" : "Mitochondria",   
"24" : "Aggresome",   
"25" : "Cytosol",   
"26" : "Cytoplasmic bodies",   
"27" : "Rods & rings",  
}

reversed_label = dict()
for i in label.keys():
    reversed_label[label[i]] = i
    
labeled_columns = [i for i in reversed_label.keys()]
df_train_labels = pd.read_csv(PATH+'train.csv', sep=',')
display(df_train_labels.describe(include='all'))

display(df_train_labels.sample(5))

for i in range(25):
    loca = [t for t in df_train_labels["Target"].value_counts()[:25].index]
    loca_name = []
    for n in loca:
        temp = [label[j] for j in n.split(' ')]
        loca_name.append(temp)
    counts = df_train_labels["Target"].value_counts()[i]  
        
    print("{} occurs {} times in data.".format((", ").join(loca_name[i]), counts))
for j in label.values():
    df_train_labels[j] = 0
def fill_rows(row):
    #function which fills dataframe based on target label
    for i in row["Target"].split(" "):
        name = label[i]
        row.loc[name] = 1
    return row
        
df_train_labels = df_train_labels.apply(fill_rows, axis=1)

df_train_labels.head()
df_train_labels[labeled_columns].sum(0).sort_values(ascending=False)
plt.figure(figsize=(7,7))
ax = plt.subplot()
plt.title("Fig1. # annotations for image")
plt.xlabel("# annot")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(False)
sns.countplot(df_train_labels.sum(1))
plt.figure(figsize=(10,10))
plt.title("Fig2. Correlation of label")
sns.heatmap(df_train_labels[labeled_columns].corr(), cmap="PiYG", linewidths=.05,
           linecolor='b',square=True)
plt.figure(figsize=(25,15))
for i, loca in enumerate(labeled_columns):
    if i==3:
        plt.title("How likely is the co-labeling for a certain compartment")
    ax = plt.subplot(5,6,i+1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(False)
    only = df_train_labels[(df_train_labels[loca] == 1) & 
                   (df_train_labels.sum(1) == 1)].shape[0]
    co_labeled = df_train_labels[(df_train_labels[loca] == 1) & 
                   (df_train_labels.sum(1) > 1)].shape[0]

    plt.title(loca)
    plt.bar(x=[0,1], height=[only,co_labeled], tick_label=["only", "colabeled"], color=["purple", "green"])
plt.figure(figsize=(7,7))
plt.title("Fig3. Barplot of label counts")
sns.barplot(x=df_train_labels[labeled_columns].sum(0).sort_values(ascending=False).index,
           y=df_train_labels[labeled_columns].sum(0).sort_values(ascending=False))
plt.xticks(rotation=90);
import os, sys
import cv2
import gc
import random
def show_random_img(noi = 1):
    #Function  that shows a given number of random imgs
    
    colors = ["_blue.png", "_green.png", "_red.png", "_yellow.png"]
    cmaps =["Blues", "Greens", "Oranges", "Reds"]
    
    rnd_imgs = []
    for i in range(noi):
        rnd = random.randint(0,len(os.listdir(PATH+"train")))
        rnd_imgs.append(os.listdir(PATH+"train")[rnd].split("_")[0])
    
    for rnd_img in rnd_imgs:
        targets = (df_train_labels["Target"][df_train_labels["Id"].str.contains(rnd_img)])
        for i in targets.iteritems():
            l = [label[j] for j in i[1].split(" ")]

        plt.figure(figsize=(20,10))
        for j,color in enumerate(colors):
            plt.subplot(1,4,j+1)
            if j == 0:
                plt.title("Nucleus")
            if j == 1:
                plt.title(l)
            if j == 2:
                plt.title("Microtubules")
            if j == 3:
                plt.title("ER")
            plt.grid(False)
            img = cv2.imread(PATH+"train/"+rnd_img+color, 0)
            plt.imshow(img, cmap=cmaps[j])

show_random_img(4)
rnd = random.randint(0,len(os.listdir(PATH+"train")))
rnd_img = os.listdir(PATH+"train")[rnd].split("_")[0]

nuc = cv2.imread(PATH+"train/"+rnd_img+"_blue.png", 0)
nuc = cv2.cvtColor(nuc, cv2.COLOR_GRAY2BGR)
poi = cv2.imread(PATH+"train/"+rnd_img+"_green.png", 0)
poi = cv2.cvtColor(poi, cv2.COLOR_GRAY2BGR)
mt = cv2.imread(PATH+"train/"+rnd_img+"_yellow.png", 0)
mt = cv2.cvtColor(mt, cv2.COLOR_GRAY2BGR)
er = cv2.imread(PATH+"train/"+rnd_img+"_red.png", 0)
er = cv2.cvtColor(er, cv2.COLOR_GRAY2BGR)

nuc[:,:,:2] = 0
poi[:,:,0] = 0
mt[:,:,2] = 0
er[:,:,1:] = 0

img3 = (0.25*(nuc/255) + 0.25*(poi/255) + 0.25*(mt/255) + 0.25 *(er/255))
plt.figure(figsize=(8,8))
plt.grid(False)
plt.imshow(3*img3)
plt.figure(figsize=(10,10))
plt.title("Nucleus and microtubule")
plt.grid(False)
nuc_mt = (0.5*(nuc/255) + 0.5*(mt/255))
plt.imshow(3*nuc_mt)

plt.figure(figsize=(10,10))
plt.title("Nucleus and ER")
plt.grid(False)
nuc_er = (0.5*(nuc/255) + 0.5*(er/255))
plt.imshow(3*nuc_er)

plt.figure(figsize=(10,10))
plt.title("Nucleus, microtubule and ER")
plt.grid(False)
nuc_er_mt = (0.33*(nuc/255) + 0.33*(er/255) + 0.33*(mt/255))
plt.imshow(3*nuc_er_mt)
rnd = random.randint(0,len(os.listdir(PATH+"train")))
rnd_img = os.listdir(PATH+"train")[rnd].split("_")[0]
nuc = cv2.imread(PATH+"train/"+rnd_img+"_blue.png", 0)
mt = cv2.imread(PATH+"train/"+rnd_img+"_yellow.png", 0)
er = cv2.imread(PATH+"train/"+rnd_img+"_red.png", 0)
composit = cv2.add(nuc, mt, er)
composit = cv2.resize(composit, (256,256))

plt.figure()
plt.grid(False)
plt.title("original")
plt.imshow(composit, cmap="gray")



#test opencv threshold methods
methods = [
("THRESH_BINARY", cv2.THRESH_BINARY),
("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
("THRESH_TRUNC", cv2.THRESH_TRUNC),
("THRESH_TOZERO", cv2.THRESH_TOZERO),
("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV),
("THRESH_OTSU", cv2.THRESH_OTSU)]

minval = 0
maxval = 255

plt.figure(figsize=(10,10))
for i, (Name, Method) in enumerate(methods):
    plt.subplot(3, 3, i+1)
    plt.grid(False)
    plt.title(Name)
    (T, thresh) = cv2.threshold(composit, minval,maxval, type=Method)
    plt.imshow(thresh, cmap="gray")

blur = cv2.GaussianBlur(composit,(5,5),0)
plt.imshow(blur)
t, thresh = cv2.threshold(blur, minval,maxval,cv2.THRESH_OTSU)
plt.imshow(thresh)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
plt.figure()
plt.title("Thresholding with Closing")
plt.grid(False)
plt.imshow(closing, cmap="gray")

nuc = cv2.imread(PATH+"train/"+rnd_img+"_blue.png", 0)
nuc = cv2.resize(nuc, (256,256))
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.grid(False)
plt.title("Nuclear staining")
plt.imshow(nuc, cmap='gray')

t, thresh = cv2.threshold(nuc, 0,255,cv2.THRESH_OTSU)

plt.subplot(132)
plt.grid(False)
plt.title("OTSU thresholding\nof nuclear staining")
plt.imshow(thresh)

kernel = np.ones((4,4),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

plt.subplot(133)
plt.grid(False)
plt.title("OTSU thresholding\nafter closing operation")
plt.imshow(closing)

im, contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

test = cv2.cvtColor(nuc, cv2.COLOR_GRAY2BGR)

t=cv2.drawContours(composit, contours, -1, (255,255,0), 1)

plt.figure(figsize=(10,10))
plt.grid(False)
plt.title("Nuclear contours drawn in composit image")
plt.imshow(t)
rnd_imgs = []
noi = 50 
tot_area = []

for i in range(noi):
    rnd = random.randint(0,len(os.listdir(PATH+"train")))
    rnd_imgs.append(os.listdir(PATH+"train")[rnd].split("_")[0])
    
for img_idx in rnd_imgs:
    
    nuc = cv2.imread(PATH+"train/"+img_idx+"_blue.png", 0)
    nuc = cv2.resize(nuc, (256,256))
    
    t, thresh = cv2.threshold(nuc, 0,255,cv2.THRESH_OTSU)
    
    kernel = np.ones((4,4),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    im, contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = [cv2.contourArea(cnt) for cnt in contours]
    tot_area.append(area)
check = [0,5,10,19,25,48]
plt.figure()
ax = plt.subplot()
median_areas = [np.median(area) for area in tot_area]
plt.bar(range(len(median_areas)), (median_areas))
for i in check:
    ax.patches[i].set_facecolor('r')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.title("Median area of segmented nuclei of 50 random images")
plt.xlabel("# img")
plt.ylabel("Area (pxl)")

check = [0,5,10,19,25,48]

plt.figure(figsize=(10,10))
for i,idx in enumerate(check):
    temp = cv2.imread(PATH+"train/"+rnd_imgs[idx]+"_blue.png", 0)
    plt.subplot(2,3,i+1)
    plt.grid(False)
    plt.title("Image {}".format(idx))
    plt.imshow(temp,cmap='gray_r')