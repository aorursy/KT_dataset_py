import csv
import numpy as np
import matplotlib.pyplot as plt
def loadCSV(fn):
    with open(fn,'r') as csf:
        lines=csv.reader(csf)
        data=list(lines)
        for i in range(len(data)):
            data[i]=[float(x) for x in data[i]]
    return np.array(data)
           
def norm(X):
    mi=np.min(X,axis=0)
    mx=np.max(X,axis=0)
    rng=mx-mi
    norm_X=1-((mx-X)/rng)
    return norm_X
def logfn(theta,X):
    return 1/(1+np.exp(-np.dot(X,theta.T)))
def log_grad(theta,X,y):
    fc=logfn(theta,X)-y.reshape(X.shape[0],-1)
    fl=np.dot(fc.T,X)
    return fl   
def cost(theta,X,y):
    log_fn=logfn(theta,X)
    y=np.squeeze(y)
    s1=y*np.log(log_fn)
    s2=(1-y)*np.log(1-log_fn)
    fi=-(s1+s2)
    return np.mean(fi)
def grad_desc(X,y,theta,lr=0.05,conv_change=0.001):
    cos=cost(theta,X,y)
    chgcos=1
    noi=1
    while chgcos>conv_change and noi<500:
        oldcos=cos
        theta=theta-(lr*log_grad(theta,X,y))/len(y)
        cos=cost(theta,X,y)
        chgcos=oldcos-cos
        noi+=1
    return theta,noi
def pred(theta,X):
    prob=logfn(theta,X)
    value=np.where(prob>=0.5,1,0)
    return np.squeeze(value)
data=loadCSV('../input/logistic.csv')
X=norm(data[:,:-1])
X=np.hstack((np.matrix(np.ones(X.shape[0])).T,X))
y=data[:,-1]
theta=np.matrix(np.zeros(X.shape[1]))
theta,noi=grad_desc(X,y,theta)
print("estimated regression coefficients:",theta)
print("no of iterations:",noi)
ypred=pred(theta,X)
print("correctly predicted labels",np.sum(y==ypred))