from numpy import *
ptsp=[]
theta1=0
theta2=0
thetal=[]
def predict(x):
    y= 1.322*x + 7.991
    return y
def error(b,m,pts):
    tr=0
    for i in range(0,len(pts)):
        x=pts[i,0]
        y=pts[i,1]
        tr+=(y-(m*x+b))**2
    return tr/float(len(pts))
def run():
    global theta1
    global theta2
    global ptsp
    pts=genfromtxt("../input/data.csv",delimiter=",")
    ptsp=pts
    import numpy as np
    xx=ptsp[:,0]
    yy=ptsp[:,1]
    mask=[]
    for i in range(0,len(xx)):
        if xx[i]<=35 or xx[i]>=65:
            mask.append(i)
    xx=np.delete(xx,mask)
    yy=np.delete(yy,mask)
    mask=[]
    for i in range(0,len(yy)):
        if yy[i]<=45 or yy[i]>=100:
            mask.append(i)
    xx=np.delete(xx,mask)
    yy=np.delete(yy,mask)
    ptsp=[[k,v] for k,v in zip(xx,yy)]
    ptsp=np.array(ptsp)
    pts=ptsp
    lr=0.0001
    ib=0
    im=0
    noi=100000
    print("starting gradient descent at b={0},m={1},error={2}".format(ib,im,error(ib,im,pts)))
    print("running....")
    b,m=gradesc_runner(pts,ib,im,lr,noi)
    print("after {0} iterations ,b={1},m={2},error={3}".format(noi,b,m,error(b,m,pts)))
    theta1=m
    theta2=b
def gradesc_runner(pts,ib,im,lr,noi):
    b=ib
    m=im
    global thetal
    for i in range(noi):
        b,m=gradesc(b,m,array(pts),lr)
        tp=[b,m,error(b,m,pts)]
        thetal.append(tp)
    return [b,m]
def gradesc(b,m,pts,lr):
    bg=0
    mg=0
    N=float(len(pts))
    for i in range(0,len(pts)):
        x=pts[i,0]
        y=pts[i,1]
        bg+=-(2/N)*(y-((m*x)+b))
        mg+=-(2/N)*x*(y-((m*x)+b))
    b=b-(lr*bg)
    m=m-(lr*mg)
    return b,m
run()
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(ptsp[:,0],ptsp[:,1],c='r')
plt.plot(ptsp[:,0],[predict(k) for k in ptsp[:,0]])
