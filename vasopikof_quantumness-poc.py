import math
import json
import copy
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import seaborn as sns

import numpy as np 
from scipy import stats
from scipy import signal
import pandas as pd

from sklearn import preprocessing
from skimage.transform import resize,downscale_local_mean
import cv2
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
def quant(jab,model,layer):

    flux = pd.DataFrame(np.full(model['0']['jabBar'].shape[0],jab))
    lss = 1
    for l in range(layer+1):
        mos={}
        mos['jabBar'] = pd.DataFrame(model[str(l)]['jabBar'])
        mos['jabSigma'] = pd.DataFrame(model[str(l)]['jabSigma'])
        lss = lss * (math.e**((-1*(flux-mos['jabBar'])**2)/((2*(mos['jabSigma']**2))))).prod().prod()
    return lss

class qnnNetwork:
    def __init__(self,qnn,numLayers):
        self.qnnLayer=[]
        self.qnnLayer.append(qnn)
        self.kernelSize = 5
        self.Models = {}
        self.Layers = numLayers
        for a in range(1,self.Layers+1):
            layerQnn = []
            bound = float(len(self.qnnLayer[0].spin))
            run =0
            for wn in self.qnnLayer[0].spin:
                tmp = self.qnnLayer[0].spin[wn]
                proto = {}
                proto['size'] = {"height":tmp.h,"width":tmp.w}
                proto['data'] = cv2.blur(np.array(tmp.data).reshape(tmp.h,tmp.w),(self.kernelSize+a,self.kernelSize+a)).flatten()
                proto['name'] = tmp.name
                layerQnn.append(proto)
                run = run+1
                update_progress(run / bound)
            self.qnnLayer.append(qnnPlatform('data',layerQnn))
            update_progress(1)
    
    def initParams(self,ref):
        self.Models[ref] = {}
        for l in range(self.Layers+1):
            self.Models[ref][str(l)] = {                
                'jabBar' : self.qnnLayer[l].JabFrames[self.qnnLayer[l].JabFrames.columns[0]].values.flatten(),
                'jabSigma' : np.ones(self.qnnLayer[l].JabFrames.shape[0])

            } 
    def makeClassCandidate(self,ref,cand,confi=1):
        self.Models[ref] = {}
        for l in range(self.Layers+1):
            self.Models[ref][str(l)] = {                
                'jabBar' : self.qnnLayer[l].JabFrames[cand].mean(axis=1).values.flatten(),
                'jabSigma' : (self.qnnLayer[l].JabFrames[cand].std(axis=1)*confi).values.flatten(),

            } 
        
    def makeClassRep(self,ref,query,confi=1):
        self.Models[ref] = {}
        for l in range(self.Layers+1):
            self.Models[ref][str(l)] = {                
                'jabBar' : self.qnnLayer[l].JabFrames.filter(like=query).mean(axis=1).values.flatten(),
                'jabSigma' : (self.qnnLayer[l].JabFrames.filter(like=query).std(axis=1)*confi).values.flatten(),

            } 
    
    def functorOptz(self,ref,inp):

        lastLayer = self.qnnLayer[self.Layers].JabFrames.filter(items=[inp])
        
        dfv = pd.concat([lastLayer.T]*lastLayer.shape[0],ignore_index=True)
        inps = []
        
        for l in range(self.Layers+1):#testNetwork.Layers+1):
            
            jBar = pd.DataFrame(self.Models[ref][str(l)]['jabBar']).values.flatten()
            jSigma = pd.DataFrame(self.Models[ref][str(l)]['jabSigma']).values.flatten()

            dfv1 = math.e**((-1*(dfv.T-jBar)**2)/(2*((jSigma)**2)))
            dfv1 = dfv1.T
            inps.append(dfv1)              
        
        self.multilayer = pd.concat(inps,ignore_index=True)
        return self.multilayer.prod().agg('sum')
    
    def functor(self,ref,inp):
        
        funt = 0

        lastLayer = self.qnnLayer[self.Layers].JabFrames[inp]

        for i in range(lastLayer.shape[0]):

            jab_s = 1
            for l in range(self.Layers+1):#testNetwork.Layers+1):
                dfv = lastLayer
                for in1 in np.arange(lastLayer.shape[0]):
                    tmp = math.e**((-1*(dfv.loc[i]-self.Models[ref][str(l)]['jabBar'][in1])**2)/(2*(self.Models[ref][str(l)]['jabSigma'][in1])**2))                                 
                    jab_s = jab_s * tmp

            update_progress(i / lastLayer.shape[0])
            funt = funt+ (jab_s)
        update_progress(1)
        return funt
    def groupEstimate(self,ref):
        inpList = self.qnnLayer[self.Layers].JabFrames.columns
        run=1
        result = {}
        for inp in inpList:
            tmp = {inp: self.functorOptz(ref,inp)}
            result.update(tmp)
            run = run+1
            update_progress(run / len(inpList))
        update_progress(1)
        return pd.DataFrame.from_dict(result,orient='index')
        
            
class qnnPlatform:
    
    def __init__(self,channel,feed):
        self.spin = {}
        self.dataFrame = {}
        if channel == 'file':
            for dirname, _, filenames in os.walk(feed):
                for filename in filenames:
                    #print(os.path.join(dirname, filename))
                    with open(os.path.join(dirname, filename)) as f:
                       inputs = json.load(f)

                    tmp = qnn(filename[:-5], inputs)
                    #tmp.calLinks()
                    self.spin[filename[:-5]] = tmp
                    self.dataFrame.update({tmp.name:tmp.allJab})
        if channel == 'data':
            run =0
            bound = len(feed)
            for i in feed:
                tmp = qnn(i['name'], i)
                #tmp.calLinks()
                self.spin[i['name']] = tmp
                self.dataFrame.update({tmp.name:tmp.allJab})
                update_progress(run / bound)
                run = run+1
            update_progress(1)
        
        #print(self.dataFrame)
        self.JabFrames = pd.DataFrame.from_dict(self.dataFrame)
        self.Models ={}
        self.groupEstimated={}
        
    def loadModel(self,ref,jabBar,jabSigma):
        self.ref = ref
        params = {
            'jabBar' : jabBar,
            'jabSigma' : jabSigma
        }
        self.Models[ref] = params
        
    def addTesting(self,test):
        self.JabFrames=self.JabFrames.join(test)

    def groupEstimator(self,ref):
        
        self.df = self.JabFrames

        self.df = math.e**((-1*(self.df.T-self.Models[ref]['jabBar'])**2)/(2*(self.Models[ref]['jabSigma']**2)))
        self.df = self.df.T
        self.df = self.df.append((self.df.prod(min_count=1)**2).rename(r'probability $|A|^2$'))
        #dfv                
        self.groupEstimated[ref] = self.df.iloc[-1]
        #return self.df.iloc[-1]
    def groupPlot(self,ref,logPlot,**kwargs):
        if len(kwargs)>0:
            if 'topN' in kwargs: 
                self.groupEstimated[ref].sort_values(ascending=kwargs['topASC']).iloc[0:kwargs['topN']].plot(kind="bar",logy=logPlot,grid=True,legend=True,ylim=(0,1.1))
        else:
            self.groupEstimated[ref].plot(kind="bar",logy=logPlot,grid=True,legend=True, ylim=(0,1.1))
    def candidatePlot(self,ref,logPlot,candidates):
        self.groupEstimated[ref].loc[candidates].plot(kind="bar",logy=logPlot,grid=True,legend=True, ylim=(0,1.1))    
    def getTopEstimated(self,ref,**kwargs):
        return self.groupEstimated[ref].sort_values(ascending=kwargs['topASC']).iloc[0:kwargs['topN']]
        
class qnn:
    def __init__(self, name, data):
        self.name = name
        self.data = data['data']
        self.w = data['size']['width']
        self.h = data['size']['height']
                
        self.Models = {
            
        }
        self.assocW = np.round((np.array(self.data)).reshape(self.h,self.w),1)
        self.render = np.round((1-np.array(self.data)).reshape(self.h,self.w),1)
        self.xticklabels=np.arange(0,self.w)
        self.yticklabels=np.arange(0,self.h)
        self.hardLimit = 0
        self.calLinks()

    def load(self,ref,model):        
        self.ref = ref    
        self.Models[ref] = model        
        
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
        
        
        self.allwJab=list((self.wLink/2).flatten())
        self.allhJab=list((self.hLink/2).flatten())
        self.allwJab.extend(self.allhJab)
        
        
        
        self.allJab = np.array(self.allwJab)
        #print('screen allJab',self.allJab[self.allJab!=0.0])
        self.allActive=list(np.argwhere(self.allJab).flatten())
        #display(self.allActive)

        self.mean=np.mean(self.allJab[self.allJab!=0])
        self.var=(np.std(self.allJab[self.allJab!=0]))**2
        #print('mean',self.mean)
        #print('var',self.var)
    
    def save(self,format='png',dpi=30):
        self.format = format
        self.dpi=dpi
        self.fig.savefig(self.name+"."+self.format, format=self.format, dpi=self.dpi)
    
    def plot(self,fig,ax,showWeight,showPrediction):
        
        
        self.ax = ax
        self.ax.clear()
        self.fig= fig
        
        self.ax.imshow(self.render, cmap='gray', vmin=0, vmax=1)
        if showPrediction:
            self.ax.set_title(self.name + ": "+str(np.round(self.prediction,3)))
        else:
            self.ax.set_title(self.name)
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        self.ax.set_xticks(np.arange(-.5, self.w, 1));
        self.ax.set_yticks(np.arange(-.5, self.h, 1));

        self.ax.set_xticklabels(self.xticklabels)
        self.ax.set_yticklabels(self.yticklabels)

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

        for i in range(len(self.yticklabels)):
            for j in range(len(self.xticklabels)):

                cl = "black" if self.assocW[i,j] <= 0.5 else "white" 
                self.ax.add_patch(Circle((j, i), 0.03, color = cl))
    
    
    def train(self,ref):
        shapeReg = ((self.allJab > 0.5).astype(np.int_))
        xi = 1-shapeReg    
        
        self.A=1
        for a in self.allActive:
            self.power= self.safeDivide((-1*(self.allJab[a]-self.mean)**2),(2*self.var))
            #print('power',self.power)
            #print('e^exp',(math.e**(self.power)))
            #print('shapeReg',shapeReg[a])
            eachItem = (shapeReg[a]*(2*self.allJab[a]+1))*(math.e**(self.power))*(math.e**(-1j*xi[a]*self.allJab[a]))
            #print('eachItem',eachItem)
            self.A*=eachItem

        estimation = self.A 
        prediction = abs(estimation)**2
        delta = 1 / math.sqrt(prediction)
        
        
        model = {
            "xi":xi,
            "shapeReg": shapeReg,
            "delta": delta
        }
        self.Models[ref] = model
        return model
    
    def predict(self,ref):
        
        self.A=1
        for a in self.allActive:
            self.power= self.safeDivide((-1*(self.allJab[a]-self.mean)**2),(2*self.var))
            #print('power',self.power)
            #print('e^exp',(math.e**(self.power)))
            #print('shapeReg',shapeReg[a])
            eachItem = (self.Models[ref]['shapeReg'][a]*(2*self.allJab[a]+1))*(math.e**(self.power))*(math.e**(-1j*self.Models[ref]['xi'][a]*self.allJab[a]))
            #print('eachItem',eachItem)
            self.A*=eachItem

        self.estimation = self.A * self.Models[ref]["delta"]
        self.prediction = abs(self.estimation)**2
def safeDivide(x,y):
    if y == 0:
        return 0
    else:
        return x/y
    
def qEstimator(jab, **kwargs):
    #function inputs (jab, jabBar, jabSigma)
    
    return safeDivide(1,(math.sqrt(2*math.pi*kwargs['jabSigma'])))*(math.e**safeDivide((-1*((jab-kwargs['jabBar'])**2)),(2*(kwargs['jabSigma']**2))))
import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
toyP1 = []
count = 0
for i in np.arange(0,1.1,0.1):
    # Serialization
    numpyData = {
        "size":{
            "height": 1,
            "width": 3            
        },
        "data": [np.round(i,1),np.round(1-i,1),np.round(1-i,1)],
        "name": "toyP1_heavyRight_"+str(count)
    }
    count+=1
    toyP1.append(numpyData)

count = 0
for i in np.arange(0,1.1,0.1):
    # Serialization
    numpyData = {
        "size":{
            "height": 1,
            "width": 3,
        },
        "data": [np.round(1-i,1),np.round(1-i,1),np.round(i,1)],
        "name": "toyP1_heavyLeft_"+str(count)
    }
    count+=1
    toyP1.append(numpyData)
toyP1datasets = qnnPlatform('data',toyP1)
plt.clf()

figs, axs = plt.subplots(figsize=(15,15))

toyP1datasets.spin['toyP1_heavyRight_0'].plot(figs,axs,False,False)
plt.clf()

figs, axs = plt.subplots(figsize=(15,15))

toyP1datasets.spin['toyP1_heavyRight_10'].plot(figs,axs,False,False)
plt.clf()

figs, axs = plt.subplots(figsize=(15,15))

toyP1datasets.spin['toyP1_heavyLeft_0'].plot(figs,axs,False,False)
toyP1datasets.JabFrames
toyP1datasets.JabFrames.filter(like='heavyLeft')
pos_perfectHeavyRight = toyP1datasets.JabFrames['toyP1_heavyRight_0']
pos_perfectHeavyLeft = toyP1datasets.JabFrames['toyP1_heavyLeft_0']

pos_semiHeavyRight = toyP1datasets.JabFrames.filter(like='heavyRight').loc[0:3]
pos_semiHeavyLeft = toyP1datasets.JabFrames.filter(like='heavyLeft').loc[0:3]


testingSets=0
del testingSets
testingSets = copy.copy(toyP1datasets)


testingSets.loadModel('perfectHeavyRight',pos_perfectHeavyRight,(pos_perfectHeavyRight*1))
testingSets.loadModel('perfectHeavyLeft',pos_perfectHeavyRight,(pos_perfectHeavyRight*1))
testingSets.loadModel('semiHeavyRight',pos_semiHeavyRight.mean(axis=1),(pos_semiHeavyRight.std(axis=1)*1))
testingSets.loadModel('semiHeavyLeft',pos_semiHeavyLeft.mean(axis=1),(pos_semiHeavyLeft.std(axis=1)*1))


testingSets.groupEstimator('perfectHeavyRight')
testingSets.groupEstimator('perfectHeavyLeft')
testingSets.groupEstimator('semiHeavyRight')
testingSets.groupEstimator('semiHeavyLeft')
testingSets.groupPlot('perfectHeavyRight',False,topN=10,topASC=False)
testingSets.groupPlot('semiHeavyRight',False,topN=10,topASC=False)
testingSets.groupPlot('perfectHeavyLeft',False,topN=10,topASC=False)
testingSets.groupPlot('semiHeavyLeft',False,topN=10,topASC=False)
fig= plt.figure(figsize=(20,10))
plotTarget = testingSets.getTopEstimated('perfectHeavyRight',topN=11,topASC=False)
plotCols = plotTarget.index
run = 0
for i in range(4):
    for j in range(4):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((4,4), (i,j))    
        ax.set_title(plotCols[run]+"\n"+str(np.round(plotTarget[plotCols[run]],5)))
        ref = ax.imshow(testingSets.spin[plotCols[run]].assocW.reshape(1,3), origin='upper',cmap='gray',vmax=1,vmin=0)
        fig.colorbar(ref,ax=ax)
        run += 1
plt.tight_layout()
plt.show()
p1network = qnnNetwork(toyP1datasets,1)
p1network.initParams('0')
plt.clf()

figs, axs = plt.subplots(figsize=(15,15))
p1network.qnnLayer[0].spin['toyP1_heavyRight_0'].plot(figs,axs,False,False)

plt.clf()

figs, axs = plt.subplots(figsize=(15,15))
p1network.qnnLayer[1].spin['toyP1_heavyRight_0'].plot(figs,axs,False,False)
funt = 0

lastLayer = p1network.qnnLayer[1].JabFrames['toyP1_heavyRight_0']


#testNetwork.qnnLayer[0].JabFrames
for i in range(lastLayer.shape[0]):
    #sum self.qnnLayer[self.Layers].JabFrames[spin]
    jab_s = 1
    for l in range(1+1):#testNetwork.Layers+1):
        dfv = lastLayer
        for in1 in np.arange(lastLayer.shape[0]):
            print(dfv.loc[i],p1network.Models['0'][str(l)]['jabBar'][in1],p1network.Models['0'][str(l)]['jabSigma'][in1])
            tmp = math.e**((-1*(dfv.loc[i]-p1network.Models['0'][str(l)]['jabBar'][in1])**2)/(2*(p1network.Models['0'][str(l)]['jabSigma'][in1]**2)))             
            print(tmp)
            jab_s = jab_s * tmp
        #print(jab_s)
    print("hjab: "+str(jab_s))
    funt = funt+ (jab_s)
print(funt)
%time p1network.functor('0','toyP1_heavyRight_0')
ta = p1network.qnnLayer[1].JabFrames.filter(items=['toyP1_heavyRight_0'])
lastLayer = ta
jab_s = 1


dfv = pd.concat([lastLayer.T]*lastLayer.shape[0],ignore_index=True)
dfv
jBar = pd.DataFrame(p1network.Models['0']['0']['jabBar']).values.flatten()
jSigma = pd.DataFrame(p1network.Models['0']['0']['jabSigma']).values.flatten()

dfv1 = math.e**((-1*(dfv.T-jBar)**2)/(2*((jSigma)**2)))
dfv1 = dfv1.T

dfv1                

jBar = pd.DataFrame(p1network.Models['0']['1']['jabBar']).values.flatten()
jSigma = pd.DataFrame(p1network.Models['0']['1']['jabSigma']).values.flatten()

dfv2 = math.e**((-1*(dfv.T-jBar)**2)/(2*((jSigma)**2)))
dfv2 = dfv2.T

dfv2                

pd.concat([dfv1,dfv2],ignore_index=True).prod()

pd.concat([dfv1,dfv2],ignore_index=True).prod().agg('sum')
%time p1network.functorOptz('0','toyP1_heavyRight_0')
p1network.makeClassCandidate('0',['toyP1_heavyRight_1','toyP1_heavyRight_2','toyP1_heavyRight_3','toyP1_heavyRight_4'],5)
%time p1network.functorOptz('0','toyP1_heavyRight_1')
toyP2 = []
count = 0
for i in np.arange(0,1.1,0.1):
    # Serialization
    numpyData = {
        "size":{
            "width": 4,
            "height": 4
        },
        "data": [np.round(i,1),np.round(1-i,1),np.round(1-i,1),np.round(i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(1-i,1),np.round(i,1),np.round(1-i,1),np.round(1-i,1),np.round(i,1)],
        "name": "toyP2_"+str(count)
    }
    count+=1
    toyP2.append(numpyData)
toyP2datasets = qnnPlatform('data',toyP2)
p2Network = qnnNetwork(toyP2datasets,1)
fig= plt.figure(figsize=(20,10))

plotCols = toyP2datasets.JabFrames.columns
run = 0
for i in range(4):
    for j in range(4):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((4,4), (i,j))    
        ax.set_title(plotCols[run])
        #ref = ax.imshow((toyP4datasets.spin[plotCols[run]].assocW.reshape(4,4)), origin='upper',cmap='gray',vmax=1,vmin=0)
        p2Network.qnnLayer[0].spin[plotCols[run]].plot(fig,ax,False,False)
        #fig.colorbar(ref,ax=ax)
        run += 1
plt.tight_layout()
plt.show()
fig= plt.figure(figsize=(20,10))

plotCols = toyP2datasets.JabFrames.columns
run = 0
for i in range(4):
    for j in range(4):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((4,4), (i,j))    
        ax.set_title(plotCols[run]+"_hidden")
        #ref = ax.imshow((toyP4datasets.spin[plotCols[run]].assocW.reshape(4,4)), origin='upper',cmap='gray',vmax=1,vmin=0)
        p2Network.qnnLayer[1].spin[plotCols[run]].plot(fig,ax,False,False)
        #fig.colorbar(ref,ax=ax)
        run += 1
plt.tight_layout()
plt.show()
p2Network.makeClassCandidate('0',['toyP2_1','toyP2_2','toyP2_3'],6)
%time p2Network.functorOptz('0','toyP2_10')
#data
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#create duplicates
train_data = train.copy()
test_data = test.copy()
target = train_data['label']
train_data.drop('label',axis=1,inplace=True)
X = train_data.to_numpy()
y = target.to_numpy()
test_data = test_data.to_numpy()
X = X.astype('float32')
test_data = test_data.astype('float32')
y = y.astype('float32')
#normalizing
X = X/255
test_data = test_data/255
tmp_n = X[np.where(y==0)]

minist_class_0 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_0_"+str(i)
    }
    minist_class_0.append(numpyData)

MINIST0datasets = qnnPlatform('data',minist_class_0)
dNetwork = qnnNetwork(MINIST0datasets,1)
dNetwork.initParams('0')
#dNetwork.makeClassCandidate('0',['MNIST_class_0_3522','MNIST_class_0_2812','MNIST_class_0_3654','MNIST_class_0_401'],2)
dNetwork.makeClassRep('0','MNIST_class_0',50)
%time dNetwork.functorOptz('0','MNIST_class_0_22')
#mlp = dNetwork.groupEstimate('0')
#mlp.columns=['pos']
#mlp.sort_values(by=['pos'],ascending=False)
secondCandiTop = ['MNIST_class_0_3793','MNIST_class_0_801','MNIST_class_0_3449','MNIST_class_0_1272']
secondCandiBottom = ['MNIST_class_0_3889', 'MNIST_class_0_127','MNIST_class_0_1967', 'MNIST_class_0_796']

pos = MINIST0datasets.JabFrames
candidate = ['MNIST_class_0_3522','MNIST_class_0_2812','MNIST_class_0_3654','MNIST_class_0_401']#candidate = MINIST0datasets.JabFrames.columns[-5:]

testingSets=0
del testingSets
testingSets = copy.copy(MINIST0datasets)


testingSets.loadModel('0',pos.mean(axis=1),((pos.std(axis=1)*50)))

testingSets.groupEstimator('0')

testingSets.groupPlot('0',False,topN=5,topASC=False)



testingSets.candidatePlot('0',False,candidate)
fig= plt.figure(figsize=(20,10))
#fig.suptitle('Plot '+r'|A|^{2} ' + 'with equation 20', fontsize=16) 
plotCols = candidate
run = 0
for i in range(2):
    for j in range(2):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((2,2), (i,j))
        
        #display(x[jabSigmaMax],y[jabBarMax])
        ax.set_title(plotCols[run])
        #ref = ax.imshow(zScaled[run], extent=[0, 5, 0, 5], origin='lower',cmap='coolwarm')
        ref = ax.imshow(testingSets.spin[plotCols[run]].assocW.reshape(28,28), origin='upper',cmap='gray')
        fig.colorbar(ref,ax=ax)#,shrink=0.5)
        run += 1
plt.tight_layout()
plt.show()
testingSets.candidatePlot('0',False,secondCandiTop)
fig= plt.figure(figsize=(20,10))
#fig.suptitle('Plot '+r'|A|^{2} ' + 'with equation 20', fontsize=16) 
plotCols = secondCandiTop
run = 0
for i in range(2):
    for j in range(2):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((2,2), (i,j))
        
        #display(x[jabSigmaMax],y[jabBarMax])
        ax.set_title(plotCols[run])#+"\n"+str(mlp.loc[plotCols[run]].values))
        #ref = ax.imshow(zScaled[run], extent=[0, 5, 0, 5], origin='lower',cmap='coolwarm')
        ref = ax.imshow(testingSets.spin[plotCols[run]].assocW.reshape(28,28), origin='upper',cmap='gray')
        fig.colorbar(ref,ax=ax)#,shrink=0.5)
        run += 1
plt.tight_layout()
plt.show()
testingSets.candidatePlot('0',False,secondCandiBottom)
fig= plt.figure(figsize=(20,10))
#fig.suptitle('Plot '+r'|A|^{2} ' + 'with equation 20', fontsize=16) 
plotCols = secondCandiBottom
run = 0
for i in range(2):
    for j in range(2):
        if run > len(plotCols)-1:
            break
        ax = plt.subplot2grid((2,2), (i,j))
        
        #display(x[jabSigmaMax],y[jabBarMax])
        ax.set_title(plotCols[run])#+"\n"+str(mlp.loc[plotCols[run]].values))
        #ref = ax.imshow(zScaled[run], extent=[0, 5, 0, 5], origin='lower',cmap='coolwarm')
        ref = ax.imshow(testingSets.spin[plotCols[run]].assocW.reshape(28,28), origin='upper',cmap='gray')
        fig.colorbar(ref,ax=ax)#,shrink=0.5)
        run += 1
plt.tight_layout()
plt.show()
tmp_n = X[np.where(y==1)]

minist_class_1 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_1_"+str(i)
    }
    minist_class_1.append(numpyData)

tmp_n = X[np.where(y==2)]

minist_class_2 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_2_"+str(i)
    }
    minist_class_2.append(numpyData)

tmp_n = X[np.where(y==3)]

minist_class_3 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_3_"+str(i)
    }
    minist_class_3.append(numpyData)

tmp_n = X[np.where(y==4)]

minist_class_4 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_4_"+str(i)
    }
    minist_class_4.append(numpyData)

tmp_n = X[np.where(y==5)]

minist_class_5 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_5_"+str(i)
    }
    minist_class_5.append(numpyData)

tmp_n = X[np.where(y==6)]

minist_class_6 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_6_"+str(i)
    }
    minist_class_6.append(numpyData)

tmp_n = X[np.where(y==7)]

minist_class_7 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_7_"+str(i)
    }
    minist_class_7.append(numpyData)

tmp_n = X[np.where(y==8)]

minist_class_8 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_8_"+str(i)
    }
    minist_class_8.append(numpyData)

minist_class_9 = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_9_"+str(i)
    }
    minist_class_9.append(numpyData)
#ministDataset = minist_class_0+minist_class_1+minist_class_2+minist_class_3+minist_class_4+minist_class_5+minist_class_6+minist_class_7+minist_class_8+minist_class_9
#pd.DataFrame(ministDataset)
MINIST0datasets = qnnPlatform('data',minist_class_0)
%time m0Network = qnnNetwork(MINIST0datasets,1)
m0Network.makeClassRep('0','MNIST_class_0',50)
rep0=m0Network.Models['0']
m0Network.functorOptz('0','MNIST_class_0_1')
m0Network.Models['1']=rep1
m0Network.Models
m0Network.functorOptz('1','MNIST_class_0_1')
m0Network.Models['2'] = rep2
m0Network.functorOptz('2','MNIST_class_0_1')
MINIST1datasets = qnnPlatform('data',minist_class_1)
m1Network = qnnNetwork(MINIST1datasets,1)
m1Network.makeClassRep('1','MNIST_class_1',50)
rep1=m1Network.Models['1']
m1Network.Models['0'] = rep0
m1Network.functorOptz('1','MNIST_class_1_1')
m1Network.functorOptz('0','MNIST_class_1_1')
m1Network.Models['2'] = rep2
m1Network.functorOptz('2','MNIST_class_1_1')
MINIST2datasets = qnnPlatform('data',minist_class_2)
%time m2Network = qnnNetwork(MINIST2datasets,1)
m2Network.makeClassRep('2','MNIST_class_2',50)
rep2 = m2Network.Models['2']
MINIST3datasets = qnnPlatform('data',minist_class_3)
%time m3Network = qnnNetwork(MINIST3datasets,1)
m3Network.makeClassRep('3','MNIST_class_3',50)
m3Network
# MINIST4datasets = qnnPlatform('data',minist_class_4)
# %time m4Network = qnnNetwork(MINIST4datasets,1)
# MINIST5datasets = qnnPlatform('data',minist_class_5)
# %time m5Network = qnnNetwork(MINIST5datasets,1)
# MINIST6datasets = qnnPlatform('data',minist_class_6)
# %time m6Network = qnnNetwork(MINIST6datasets,1)
# MINIST7datasets = qnnPlatform('data',minist_class_7)
# %time m7Network = qnnNetwork(MINIST7datasets,1)
# MINIST8datasets = qnnPlatform('data',minist_class_8)
# %time m8Network = qnnNetwork(MINIST8datasets,1)
# MINIST9datasets = qnnPlatform('data',minist_class_9)
# %time m9Network = qnnNetwork(MINIST9datasets,1)
tmp_n = test_data

minist_class_test = []
for i in range(len(tmp_n)):
    # Serialization
    numpyData = {
        "size":{
            "width": 28,
            "height": 28
        },
        "data": tmp_n[i],
        "name": "MNIST_class_test_"+str(i)
    }
    minist_class_test.append(numpyData)
