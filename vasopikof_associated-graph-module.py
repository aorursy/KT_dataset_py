import numpy as np 

import math

import matplotlib.pyplot as plt

import matplotlib.patches as patches



from matplotlib.patches import Circle





import json



import os

for dirname, _, filenames in os.walk('/kaggle/input/ldataset/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
with open('/kaggle/input/mnist-coefficient-visalization/two_raw_weights/neuron1_layer1.json') as f:

   data = json.load(f)

print (data['size'])
plt.imshow(np.array(data['data']).reshape(data['size']['width'],data['size']['height']))
#display(np.array(data['data']).reshape(data['size']['width'],data['size']['height'])[:,:])
pix = np.array(data['data']).reshape(data['size']['width'],data['size']['height'])

pix = np.round(pix,1)

pixs = 1-pix

pixs = np.round(pixs,1)



xticklabels=np.arange(0,data['size']['width'])

yticklabels=np.arange(0,data['size']['height'])
plt.imshow(pix, cmap='gray', vmin=0, vmax=1)

plt.show()
plt.imshow(pixs, cmap='gray', vmin=0, vmax=1)

plt.show()
fig, ax = plt.subplots(figsize=(15,15))



ax.imshow(pixs, cmap='gray', vmin=0, vmax=1)



# draw gridlines

ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

ax.set_xticks(np.arange(-.5, data['size']['width'], 1));

ax.set_yticks(np.arange(-.5, data['size']['height'], 1));



ax.set_xticklabels(xticklabels)

ax.set_yticklabels(yticklabels)





for i in range(len(yticklabels)):

    for j in range(len(xticklabels)):

        text = ax.text(j, i, pix[i, j],

                       ha="center", va="center", color="w")



plt.show()

tmpx = np.ones_like(np.arange(28*27).reshape(28,27)).astype(float)

tmpy = np.ones_like(np.arange(27*28).reshape(27,28)).astype(float)



for i in range(28):

    for j in range(27):

        if i <27 or j<27:

            tmpx[i,j] = (min(pix[i,j], pix[i,j+1])*10)

        

for i in range(27):

    for j in range(28):

        if i <27 or j<27:

            tmpy[i,j] = (min(pix[i,j], pix[i+1,j])*10)



tmpx=tmpx.astype(int)

tmpy=tmpy.astype(int)

            

print(tmpx.shape)

print(tmpy.shape)
class linkGen:

    def __init__(self, degree, origin, target):

        self.degree = degree

        self.origin = origin

        self.target = target

    def generate(self):

        if self.degree == 1:

            return [patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **arrowBlack)]

        if self.degree == 2:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **arrowBlack)

            ]

        if self.degree == 3:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **arrowBlack)

            ]    

        if self.degree == 4:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.15", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.15", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **arrowBlack)

            ] 

        if self.degree == 5:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **arrowBlack)

            ] 

        if self.degree == 6:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.1", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.3", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.1", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.3", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **arrowWhite)

            ]

        if self.degree == 7:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.2", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.2", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **arrowWhite)

            ]

        if self.degree == 8:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.08", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.08", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **arrowWhite)

            ]

        if self.degree == 9:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.1", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.1", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.45", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.45", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **arrowWhite)



            ]

        if self.degree == 10:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.05", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.05", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.2", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.2", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.35", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.35", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.55", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.55", **arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.7", **arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.7", **arrowWhite)



            ]

        else:

            return []

        
class assocG:

    def __init__(self, name, w,h,data):

        self.name = name

        self.data = data

        self.w = w

        self.h = h



    def load(self):

        self.assocW = np.round((np.array(self.data)).reshape(self.w,self.h),1)

        self.render = np.round((1-np.array(self.data)).reshape(self.w,self.h),1)

        self.xticklabels=np.arange(0,self.w)

        self.yticklabels=np.arange(0,self.h)

        

    def calLinks(self):

        self.wLink = np.ones_like(np.arange((self.h)*(self.w-1)).reshape(self.h,self.w-1)).astype(float)

        self.hLink = np.ones_like(np.arange((self.h-1)*(self.w)).reshape(self.h-1,self.w)).astype(float)



        for i in range(self.h):

            for j in range(self.w-1):

                if i <self.h-1 or j<self.w-1:

                    self.wLink[i,j] = (min(self.assocW[i,j], self.assocW[i,j+1])*10)



        for i in range(self.h-1):

            for j in range(self.w):

                if i <self.h-1 or j<self.w-1:

                    self.hLink[i,j] = (min(self.assocW[i,j], self.assocW[i+1,j])*10)



        self.wLink=self.wLink.astype(int)

        self.hLink=self.hLink.astype(int)

    

    def plot(self,fig,ax):

        self.ax = ax

        self.fig= fig

        

        self.ax.imshow(self.render, cmap='gray', vmin=0, vmax=1)



        # draw gridlines

        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

        self.ax.set_xticks(np.arange(-.5, self.w, 1));

        self.ax.set_yticks(np.arange(-.5, self.h, 1));



        self.ax.set_xticklabels(self.xticklabels)

        self.ax.set_yticklabels(self.yticklabels)

        #style="Simple,tail_width=0.1,head_width=10,head_length=8"

        style="-"

        kw = dict(arrowstyle=style, color="k")





        for i in range(len(self.yticklabels)):

            for j in range(len(self.xticklabels)):

                text = ax.text(j, i, self.assocW[i,j],ha="center", va="center", color="w")



                if (i<self.h-1 and j < self.w-1):

                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()

                    for arx in arrowg:

                        plt.gca().add_patch(arx)

                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()

                    for ary in arrowg:

                        plt.gca().add_patch(ary)

                elif(j == self.w-1 and i < self.h-1):                    

                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()

                    for ary in arrowg:

                        plt.gca().add_patch(ary)

                elif(i == self.h-1 and j < self.w-1):

                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()

                    for arx in arrowg:

                        plt.gca().add_patch(arx)

        
fig, ax = plt.subplots(figsize=(15,15))



ax.imshow(pix, cmap='gray', vmin=0, vmax=1)



# draw gridlines

ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

ax.set_xticks(np.arange(-.5, data['size']['width'], 1));

ax.set_yticks(np.arange(-.5, data['size']['height'], 1));



ax.set_xticklabels(xticklabels)

ax.set_yticklabels(yticklabels)



#style="Simple,tail_width=0.1,head_width=10,head_length=8"

style="-"

arrowBlack = dict(arrowstyle=style, color="k")

arrowWhite = dict(arrowstyle=style, color="w")



arrowg = linkGen(1,(0,1),(0,2)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(2,(0,2),(0,3)).generate()

for i in arrowg:

    plt.gca().add_patch(i)

    

arrowg = linkGen(3,(0,3),(0,4)).generate()

for i in arrowg:

    plt.gca().add_patch(i)

    

arrowg = linkGen(4,(0,4),(0,5)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(5,(0,5),(0,6)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(6,(0,6),(0,7)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(7,(0,7),(0,8)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(8,(0,8),(0,9)).generate()

for i in arrowg:

    plt.gca().add_patch(i)

    

arrowg = linkGen(9,(0,9),(0,10)).generate()

for i in arrowg:

    plt.gca().add_patch(i)



arrowg = linkGen(10,(0,10),(0,11)).generate()

for i in arrowg:

    plt.gca().add_patch(i)

    

plt.show()

l1sample = assocG("l1sample", data['size']['width'],data['size']['height'],data['data'])

l1sample.load()

l1sample.calLinks()
len(l1sample.wLink)

print(l1sample.wLink[27,26])
fig, ax = plt.subplots(figsize=(15,15))

l1sample.plot(fig,ax)

plt.show()
plt.clf()

with open('/kaggle/input/mnist-coefficient-visalization/two_raw_weights/neuron1_layer2.json') as f:

   l2_input = json.load(f)



figs, axs = plt.subplots(figsize=(15,15))



l2sample = assocG("l2sample", l2_input['size']['width'],l2_input['size']['height'],l2_input['data'])

l2sample.load()

l2sample.calLinks()



l2sample.plot(figs,axs)



plt.tight_layout()

plt.show()
class linkGen:

    def __init__(self, degree, origin, target):

        self.degree = degree

        self.origin = origin

        self.target = target

    def generate(self):

        self.style="-"

        self.arrowBlack = dict(arrowstyle=self.style, color="k")

        self.arrowWhite = dict(arrowstyle=self.style, color="w")

        

        if self.degree == 1:

            return [patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **self.arrowBlack)]

        if self.degree == 2:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **self.arrowBlack)

            ]

        if self.degree == 3:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **self.arrowBlack)

            ]    

        if self.degree == 4:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.15", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.15", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **self.arrowBlack)

            ] 

        if self.degree == 5:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **self.arrowBlack),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **self.arrowBlack)

            ] 

        if self.degree == 6:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.1", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.3", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.5", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.1", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.3", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.5", **self.arrowWhite)

            ]

        if self.degree == 7:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **self.arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.2", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.2", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **self.arrowWhite)

            ]

        if self.degree == 8:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.08", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.08", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.4", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.4", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **self.arrowWhite)

            ]

        if self.degree == 9:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.1", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.1", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.25", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.25", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.45", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.45", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.6", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.6", **self.arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=0", **self.arrowWhite)



            ]

        if self.degree == 10:

            return [

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.05", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.05", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.2", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.2", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.35", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.35", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.55", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.55", **self.arrowWhite),                

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=.7", **self.arrowWhite),

                patches.FancyArrowPatch(self.origin,self.target,connectionstyle="arc3,rad=-.7", **self.arrowWhite)



            ]

        else:

            return []

        

class assocG_w_dots:

    def __init__(self, name, w,h,data):

        self.name = name

        self.data = data

        self.w = w

        self.h = h



    def load(self):

        self.assocW = np.round((np.array(self.data)).reshape(self.h,self.w),1)

        self.render = np.round((1-np.array(self.data)).reshape(self.h,self.w),1)

        self.xticklabels=np.arange(0,self.w)

        self.yticklabels=np.arange(0,self.h)

        

    def calLinks(self):

        self.wLink = np.ones_like(np.arange((self.h)*(self.w-1)).reshape(self.h,self.w-1)).astype(float)

        self.hLink = np.ones_like(np.arange((self.h-1)*(self.w)).reshape(self.h-1,self.w)).astype(float)



        for i in range(self.h):

            for j in range(self.w-1):

                if i <self.h-1 or j<self.w-1:

                    self.wLink[i,j] = (min(self.assocW[i,j], self.assocW[i,j+1])*10)



        for i in range(self.h-1):

            for j in range(self.w):

                if i <self.h-1 or j<self.w-1:

                    self.hLink[i,j] = (min(self.assocW[i,j], self.assocW[i+1,j])*10)



        self.wLink=self.wLink.astype(int)

        self.hLink=self.hLink.astype(int)

    

    def save(self,format='png',dpi=30):

        self.format = format

        self.dpi=dpi

        self.fig.savefig(self.name+"."+self.format, format=self.format, dpi=self.dpi)

    

    def plot(self,fig,ax,showWeight):

        

        

        self.ax = ax

        self.fig= fig

        

        self.ax.imshow(self.render, cmap='gray', vmin=0, vmax=1)

        self.ax.set_title(self.name)

        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

        self.ax.set_xticks(np.arange(-.5, self.w, 1));

        self.ax.set_yticks(np.arange(-.5, self.h, 1));



        self.ax.set_xticklabels(self.xticklabels)

        self.ax.set_yticklabels(self.yticklabels)

        #style="Simple,tail_width=0.1,head_width=10,head_length=8"



        self.style="-"

        self.arrowBlack = dict(arrowstyle=self.style, color="k")

        self.arrowWhite = dict(arrowstyle=self.style, color="w")

       





        for i in range(len(self.yticklabels)):

            for j in range(len(self.xticklabels)):

                #changed here

                if showWeight:

                    cl = "black" if self.assocW[i,j] <= 0.5 else "white" 

                    text = self.ax.text(j-0.2, i, self.assocW[i,j],ha="center", va="center", color=cl, size='xx-large')

                

                if (i<self.h-1 and j < self.w-1):

                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()

                    for arx in arrowg:

                        self.ax.add_patch(arx)

                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()

                    for ary in arrowg:

                        self.ax.add_patch(ary)

                elif(j == self.w-1 and i < self.h-1):                    

                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()

                    for ary in arrowg:

                        self.ax.add_patch(ary)

                elif(i == self.h-1 and j < self.w-1):

                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()

                    for arx in arrowg:

                        self.ax.add_patch(arx)



        #changed here

        for i in range(len(self.yticklabels)):

            for j in range(len(self.xticklabels)):



                cl = "black" if self.assocW[i,j] <= 0.5 else "white" 

                self.ax.add_patch(Circle((j, i), 0.03, color = cl))



plt.clf()

with open('/kaggle/input/mnist-coefficient-visalization/two_raw_weights/neuron1_layer2.json') as f:

   l2_input = json.load(f)



figs, axs = plt.subplots(figsize=(15,15))



l2sample = assocG_w_dots("l2sample", l2_input['size']['width'],l2_input['size']['height'],l2_input['data'])

l2sample.load()

l2sample.calLinks()



#showWeight

l2sample.plot(figs,axs,True)

#l2sample.plot(figs,axs,False)



#l2sample.save()



plt.tight_layout()

plt.show()

plt.clf()

with open('/kaggle/input/mnist-coefficient-visalization/two_raw_weights/neuron1_layer1.json') as f:

   l1_input = json.load(f)



figs, axs = plt.subplots(figsize=(15,15))



l1final = assocG_w_dots("l1final", l1_input['size']['width'],l1_input['size']['height'],l1_input['data'])

l1final.load()

l1final.calLinks()



l1final.plot(figs,axs,False)



#l1final.save(format="jpg")



plt.tight_layout()

plt.show()