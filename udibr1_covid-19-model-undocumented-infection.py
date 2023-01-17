import pandas as pd

import scipy.io as sio

import numpy as np
import os

os.listdir('../input')
cities = sio.loadmat('../input/cities.mat')

cities = cities['cities']

cities = [city[0][0] for city in cities]

len(cities),cities[170-1]
def lhsu(xmin,xmax,nsample):

    '''LHS from uniform distribution

    Input:

        xmin    : min of data (nvar)

        xmax    : max of data (nvar)

        nsample : no. of samples

    Output:

        s       : random sample (nsample,nvar)

    Budiman (2003)

    '''

    nvar=len(xmin)

    ran=np.random.uniform(size=(nsample,nvar))

    s=np.zeros((nsample,nvar),dtype=np.float64)

    for j in range(nvar):

        idx=np.random.permutation(nsample)+1

        P =(idx-ran[:,j])/nsample;

        s[:,j] = xmin[j] + P * (xmax[j]-xmin[j])

    return s
def initialize(pop,num_ens):

    # Initialize the metapopulation SEIRS model

    # load mobility int32(375, 375, 14)

    M = sio.loadmat('../input/M.mat')['M']

    num_loc=len(pop) # 375

    # num_var=5*num_loc+6;

    # S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

    # prior range

    Slow=1.0;Sup=1.0; # susceptible fraction

    Elow=0;Eup=0; # exposed

    Irlow=0;Irup=0; # documented infection

    Iulow=0;Iuup=0; # undocumented infection

    obslow=0;obsup=0; # reported case

    betalow=0.8;betaup=1.5; # transmission rate

    mulow=0.2;muup=1.0; # relative transmissibility

    thetalow=1;thetaup=1.75; # movement factor

    Zlow=2;Zup=5; # latency period

    alphalow=0.02;alphaup=1.0; # reporting rate

    Dlow=2;Dup=5; # infectious period

    # range of model state including variables and parameters

    xmin=[]

    xmax=[]

    for i in range(num_loc):

        xmin += [Slow*pop[i], Elow*pop[i], Irlow*pop[i], Iulow*pop[i], obslow]

        xmax +=[Sup*pop[i], Eup*pop[i], Irup*pop[i], Iuup*pop[i], obsup]

    xmin+=[betalow,mulow,thetalow,Zlow,alphalow,Dlow]

    xmax+=[betaup,muup,thetaup,Zup,alphaup,Dup]

    paramax=xmax[-6:]

    paramin=xmin[-6:]

    # seeding in Wuhan

    # Wuhan - 170

    seedid=170-1

    # E

    xmin[seedid*5+1]=0;xmax[seedid*5+1]=2000

    # Is

    xmin[seedid*5+2]=0;xmax[seedid*5+2]=0

    # Ia

    xmin[seedid*5+3]=0;xmax[seedid*5+3]=2000

    # Latin Hypercubic Sampling

    x=lhsu(xmin,xmax,num_ens)

    x=x.T

    x=np.round(x);

    # seeding in other cities

    C=M[:,seedid,0] # first day

    for i in range(num_loc):

        if i != seedid:

            # E

            Ewuhan=x[seedid*5+1,:]

            x[i*5+1,:]=np.round(C[i]*3.*Ewuhan/pop[seedid])

            # Ia

            Iawuhan=x[seedid*5+3,:];

            x[i*5+3,:]=np.round(C[i]*3.*Iawuhan/pop[seedid])

    return (x,np.array(paramax),np.array(paramin))
def SEIR(x,M,pop,ts,pop0):

    # the metapopulation SEIR model

    dt=1

    tmstep=1

    # integrate forward for one day

    num_loc=len(pop)

    num_ens=x.shape[1]

    #S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

    Sidx=slice(0,5*num_loc,5)

    Eidx=slice(1,5*num_loc,5)

    Isidx=slice(2,5*num_loc,5)

    Iaidx=slice(3,5*num_loc,5)

    obsidx=slice(4,5*num_loc,5)

    

    betaidx=5*num_loc

    muidx=5*num_loc+1

    thetaidx=5*num_loc+2

    Zidx=5*num_loc+3

    alphaidx=5*num_loc+4

    Didx=5*num_loc+5



    S=np.zeros((num_loc,num_ens,tmstep+1),dtype=np.float64)

    E=np.zeros((num_loc,num_ens,tmstep+1),dtype=np.float64)

    Is=np.zeros((num_loc,num_ens,tmstep+1),dtype=np.float64)

    Ia=np.zeros((num_loc,num_ens,tmstep+1),dtype=np.float64)

    Incidence=np.zeros((num_loc,num_ens,tmstep+1),dtype=np.float64)

    obs=np.zeros((num_loc,num_ens),dtype=np.float64)

    #initialize S,E,Is,Ia and parameters

    S[:,:,0]=x[Sidx,:]

    E[:,:,0]=x[Eidx,:]

    Is[:,:,0]=x[Isidx,:]

    Ia[:,:,0]=x[Iaidx,:]

    beta=x[betaidx,:]

    mu=x[muidx,:]

    theta=x[thetaidx,:]

    Z=x[Zidx,:]

    alpha=x[alphaidx,:]

    D=x[Didx,:]

    #start integration

    tcnt=-1

    for t in range(ts+dt,ts+tmstep,dt):

        tcnt=tcnt+1

        dt1=dt

        #first step

        ESenter=dt1*np.multiply(np.ones((num_loc,1))*theta,M[:,:,ts]*np.divide(S[:,:,tcnt],(pop-IS[:,:,tcnt])))

        ESleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(S[:,:,tcnt],pop-IS[:,:,tcnt])), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*S[:,:,tcnt])

        EEenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(E[:,:,tcnt],(pop-IS[:,:,tcnt])))

        EEleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(E[:,:,tcnt], pop-IS[:,:,tcnt])), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*E[:,:,tcnt])

        EIaenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Ia[:,:,tcnt],(pop-IS[:,:,tcnt])))

        EIaleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Ia[:,:,tcnt],(pop-IS[:,:,tcnt]))),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Ia[:,:,tcnt])



        Eexps=dt1*np.divide(np.multiply(np.multiply(np.ones((num_loc,1))*beta, S[:,:,tcnt]),IS[:,:,tcnt]),pop)

        Eexpa=dt1*np.divide(np.multiply(np.multiply(np.multiply(np.ones((num_loc,1))*mu,np.ones((num_loc,1))*beta),S[:,:,tcnt]),Ia[:,:,tcnt]),pop)

        Einfs=dt1*np.multiply(np.ones((num_loc,1))*alpha, np.divide(E[:,:,tcnt],(np.ones((num_loc,1))*Z)))

        Einfa=dt1*np.divide(np.multiply(np.ones((num_loc,1))*(1-alpha),E[:,:,tcnt]),(np.ones((num_loc,1))*Z))

        Erecs=dt1*np.divide(IS[:,:,tcnt],np.ones((num_loc,1))*D)

        Ereca=dt1*np.divide(Ia[:,:,tcnt],np.ones((num_loc,1))*D)



        ESenter=np.maximum(ESenter,0);ESleft=np.maximum(ESleft,0);

        EEenter=np.maximum(EEenter,0);EEleft=np.maximum(EEleft,0);

        EIaenter=np.maximum(EIaenter,0);EIaleft=np.maximum(EIaleft,0);

        Eexps=np.maximum(Eexps,0);Eexpa=np.maximum(Eexpa,0);

        Einfs=np.maximum(Einfs,0);Einfa=np.maximum(Einfa,0);

        Erecs=np.maximum(Erecs,0);Ereca=np.maximum(Ereca,0);



        ##########stochastic version

        ESenter=np.random.poisson(ESenter);ESleft=np.random.poisson(ESleft);

        EEenter=np.random.poisson(EEenter);EEleft=np.random.poisson(EEleft);

        EIaenter=np.random.poisson(EIaenter);EIaleft=np.random.poisson(EIaleft);

        Eexps=np.random.poisson(Eexps);

        Eexpa=np.random.poisson(Eexpa);

        Einfs=np.random.poisson(Einfs);

        Einfa=np.random.poisson(Einfa);

        Erecs=np.random.poisson(Erecs);

        Ereca=np.random.poisson(Ereca);



        sk1=-Eexps-Eexpa+ESenter-ESleft;

        ek1=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;

        isk1=Einfs-Erecs;

        iak1=Einfa-Ereca+EIaenter-EIaleft;

        ik1i=Einfs;

        #second step

        Ts1=S[:,:,tcnt]+sk1/2

        Te1=E[:,:,tcnt]+ek1/2

        Tis1=IS[:,:,tcnt]+isk1/2

        Tia1=Ia[:,:,tcnt]+iak1/2



        ESenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Ts1,(pop-Tis1)))

        ESleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta,np.divide(Ts1, (pop-Tis1))),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Ts1)

        EEenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Te1, (pop-Tis1)))

        EEleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Te1, (pop-Tis1))), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Te1)

        EIaenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.multiply(Tia1,(pop-Tis1)))

        EIaleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Tia1,(pop-Tis1))), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Tia1)



        Eexps=dt1*np.divide(np.multiply(np.multiply(np.ones((num_loc,1))*beta, Ts1), Tis1), pop)

        Eexpa=dt1*np.divide(np.multiply(np.multiply(np.multiply(np.ones((num_loc,1))*mu, np.ones((num_loc,1))*beta), Ts1), Tia1), pop)

        Einfs=dt1*np.divide(np.multiply(np.ones((num_loc,1))*alpha, Te1), np.ones((num_loc,1))*Z)

        Einfa=dt1*np.multiply(np.ones((num_loc,1))*(1-alpha), np.divide(Te1,np.ones((num_loc,1))*Z))

        Erecs=dt1*np.divide(Tis1, np.ones((num_loc,1))*D)

        Ereca=dt1*np.divide(Tia1, np.ones((num_loc,1))*D)



        ESenter=np.maximum(ESenter,0);ESleft=np.maximum(ESleft,0);

        EEenter=np.maximum(EEenter,0);EEleft=np.maximum(EEleft,0);

        EIaenter=np.maximum(EIaenter,0);EIaleft=np.maximum(EIaleft,0);

        Eexps=np.maximum(Eexps,0);Eexpa=np.maximum(Eexpa,0);

        Einfs=np.maximum(Einfs,0);Einfa=np.maximum(Einfa,0);

        Erecs=np.maximum(Erecs,0);Ereca=np.maximum(Ereca,0);



        ##########stochastic version

        ESenter=np.random.poisson(ESenter);ESleft=np.random.poisson(ESleft);

        EEenter=np.random.poisson(EEenter);EEleft=np.random.poisson(EEleft);

        EIaenter=np.random.poisson(EIaenter);EIaleft=np.random.poisson(EIaleft);

        Eexps=np.random.poisson(Eexps);

        Eexpa=np.random.poisson(Eexpa);

        Einfs=np.random.poisson(Einfs);

        Einfa=np.random.poisson(Einfa);

        Erecs=np.random.poisson(Erecs);

        Ereca=np.random.poisson(Ereca);



        sk2=-Eexps-Eexpa+ESenter-ESleft;

        ek2=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;

        isk2=Einfs-Erecs;

        iak2=Einfa-Ereca+EIaenter-EIaleft;

        ik2i=Einfs;



        #third step

        Ts2=S[:,:,tcnt]+sk2/2;

        Te2=E[:,:,tcnt]+ek2/2;

        Tis2=IS[:,:,tcnt]+isk2/2;

        Tia2=Ia[:,:,tcnt]+iak2/2;



        ESenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Ts2, (pop-Tis2)))

        ESleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta,np.divide(Ts2,pop-Tis2)),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))),dt1*Ts2)

        EEenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Te2,(pop-Tis2)))

        EEleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Te2,(pop-Tis2))),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))),dt1*Te2)

        EIaenter=dt1*np.multiply(np.ones((num_loc,1))*theta,M[:,:,ts]*np.divide(Tia2,(pop-Tis2)))

        EIaleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta,np.divide(Tia2,pop-Tis2)),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))),dt1*Tia2)



        Eexps=dt1*np.divide(np.multiply(np.multiply(np.ones((num_loc,1))*beta,Ts2),Tis2),pop)

        Eexpa=dt1*np.divide(np.multiply(np.multiply(np.multiply(np.ones((num_loc,1))*mu,np.ones((num_loc,1))*beta),Ts2),Tia2),pop)

        Einfs=dt1*np.multiply(np.ones((num_loc,1))*alpha, np.divide(Te2,np.ones((num_loc,1))*Z))

        Einfa=dt1*np.multiply(np.ones((num_loc,1))*(1-alpha),np.divide(Te2,np.ones((num_loc,1))*Z))

        Erecs=dt1*np.divide(Tis2,np.ones((num_loc,1))*D)

        Ereca=dt1*np.divide(Tia2, np.ones((num_loc,1))*D)



        ESenter=np.maximum(ESenter,0);ESleft=np.maximum(ESleft,0);

        EEenter=np.maximum(EEenter,0);EEleft=np.maximum(EEleft,0);

        EIaenter=np.maximum(EIaenter,0);EIaleft=np.maximum(EIaleft,0);

        Eexps=np.maximum(Eexps,0);Eexpa=np.maximum(Eexpa,0);

        Einfs=np.maximum(Einfs,0);Einfa=np.maximum(Einfa,0);

        Erecs=np.maximum(Erecs,0);Ereca=np.maximum(Ereca,0);



        ##########stochastic version

        ESenter=np.random.poisson(ESenter);ESleft=np.random.poisson(ESleft);

        EEenter=np.random.poisson(EEenter);EEleft=np.random.poisson(EEleft);

        EIaenter=np.random.poisson(EIaenter);EIaleft=np.random.poisson(EIaleft);

        Eexps=np.random.poisson(Eexps);

        Eexpa=np.random.poisson(Eexpa);

        Einfs=np.random.poisson(Einfs);

        Einfa=np.random.poisson(Einfa);

        Erecs=np.random.poisson(Erecs);

        Ereca=np.random.poisson(Ereca);



        sk3=-Eexps-Eexpa+ESenter-ESleft;

        ek3=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;

        isk3=Einfs-Erecs;

        iak3=Einfa-Ereca+EIaenter-EIaleft;

        ik3i=Einfs;



        #fourth step

        Ts3=S[:,:,tcnt]+sk3;

        Te3=E[:,:,tcnt]+ek3;

        Tis3=IS[:,:,tcnt]+isk3;

        Tia3=Ia[:,:,tcnt]+iak3;



        ESenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Ts3,(pop-Tis3)))

        ESleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Ts3,(pop-Tis3))), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Ts3)

        EEenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Te3,(pop-Tis3)))

        EEleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Te3,(pop-Tis3))), np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))), dt1*Te3)

        EIaenter=dt1*np.multiply(np.ones((num_loc,1))*theta, M[:,:,ts]*np.divide(Tia3,(pop-Tis3)))

        EIaleft=np.minimum(dt1*np.multiply(np.multiply(np.ones((num_loc,1))*theta, np.divide(Tia3, pop-Tis3)),np.sum(M[:,:,ts],0).T*np.ones((1,num_ens))),dt1*Tia3)



        Eexps=dt1*np.divide(np.multiply(np.multiply(np.ones((num_loc,1))*beta, Ts3),Tis3),pop)

        Eexpa=dt1*np.divide(np.multiply(np.multiply(np.multiply(np.ones((num_loc,1))*mu,np.ones((num_loc,1))*beta),Ts3),Tia3),pop)

        Einfs=dt1*np.multiply(np.ones((num_loc,1))*alpha,np.multiply(Te3, np.ones((num_loc,1))*Z))

        Einfa=dt1*np.divide(np.multiply(np.ones((num_loc,1))*(1-alpha),Te3), np.ones((num_loc,1))*Z)

        Erecs=dt1*np.divide(Tis3, np.ones((num_loc,1))*D)

        Ereca=dt1*np.divide(Tia3, np.ones((num_loc,1))*D)



        ESenter=np.maximum(ESenter,0);ESleft=np.maximum(ESleft,0);

        EEenter=np.maximum(EEenter,0);EEleft=np.maximum(EEleft,0);

        EIaenter=np.maximum(EIaenter,0);EIaleft=np.maximum(EIaleft,0);

        Eexps=np.maximum(Eexps,0);Eexpa=np.maximum(Eexpa,0);

        Einfs=np.maximum(Einfs,0);Einfa=np.maximum(Einfa,0);

        Erecs=np.maximum(Erecs,0);Ereca=np.maximum(Ereca,0);



        ##########stochastic version

        ESenter=np.random.poisson(ESenter);ESleft=np.random.poisson(ESleft);

        EEenter=np.random.poisson(EEenter);EEleft=np.random.poisson(EEleft);

        EIaenter=np.random.poisson(EIaenter);EIaleft=np.random.poisson(EIaleft);

        Eexps=np.random.poisson(Eexps);

        Eexpa=np.random.poisson(Eexpa);

        Einfs=np.random.poisson(Einfs);

        Einfa=np.random.poisson(Einfa);

        Erecs=np.random.poisson(Erecs);

        Ereca=np.random.poisson(Ereca);



        sk4=-Eexps-Eexpa+ESenter-ESleft;

        ek4=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;

        isk4=Einfs-Erecs;

        iak4=Einfa-Ereca+EIaenter-EIaleft;

        ik4i=Einfs;



        #####

        S[:,:,tcnt+1]=S[:,:,tcnt]+np.round(sk1/6+sk2/3+sk3/3+sk4/6);

        E[:,:,tcnt+1]=E[:,:,tcnt]+np.round(ek1/6+ek2/3+ek3/3+ek4/6);

        Is[:,:,tcnt+1]=IS[:,:,tcnt]+np.round(isk1/6+isk2/3+isk3/3+isk4/6);

        Ia[:,:,tcnt+1]=Ia[:,:,tcnt]+np.round(iak1/6+iak2/3+iak3/3+iak4/6);

        Incidence[:,:,tcnt+1]=np.round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);

        obs=Incidence[:,:,tcnt+1];



    ###update x

    x[Sidx,:]=S[:,:,tcnt+1];

    x[Eidx,:]=E[:,:,tcnt+1];

    x[Isidx,:]=Is[:,:,tcnt+1];

    x[Iaidx,:]=Ia[:,:,tcnt+1];

    x[obsidx,:]=obs;

    ###update pop

    pop=pop-np.sum(M[:,:,ts],0,keepdims=True).T*theta+np.sum(M[:,:,ts],1,keepdims=True)*theta;

    minfrac=0.6;

    pop[np.less(pop, minfrac*pop0)] = pop0[np.less(pop, minfrac*pop0)]*minfrac;

    return (x,pop)                                              
def checkbound_ini(x,pop):

    #S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

    betalow=0.8;betaup=1.5;#transmission rate

    mulow=0.2;muup=1.0;#relative transmissibility

    thetalow=1;thetaup=1.75;#movement factor

    Zlow=2;Zup=5;#latency period

    alphalow=0.02;alphaup=1.0;#reporting rate

    Dlow=2;Dup=5;#infectious period

    xmin=np.array([betalow,mulow,thetalow,Zlow,alphalow,Dlow])

    xmax=np.array([betaup,muup,thetaup,Zup,alphaup,Dup])

    num_loc=pop.shape[0]

    for i in range(num_loc):

        #S

        x[i*5+0,x[i*5+0,:]<0]=0;

        x[i*5+0,x[i*5+0,:]>pop[i,:]]=pop[i,x[i*5+0,:]>pop[i,:]];

        #E

        x[i*5+1,x[i*5+1,:]<0]=0;

        #Ir

        x[i*5+2,x[i*5+2,:]<0]=0;

        #Iu

        x[i*5+3,x[i*5+3,:]<0]=0;

        #obs

        x[i*5+4,x[i*5+4,:]<0]=0

    for i in range(6):

        temp=x[-6+i,:]

        index=np.logical_or(temp<xmin[i], temp>xmax[i])

        index_out=np.where(index>0)[0]

        index_in=np.where(index==0)[0]

        #redistribute out bound ensemble members

        x[-6+i,index_out]=np.random.choice(x[-6+i,index_in],len(index_out))

    return x



def checkbound(x,pop):

    #S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

    betalow=0.8;betaup=1.5;#transmission rate

    mulow=0.2;muup=1.0;#relative transmissibility

    thetalow=1;thetaup=1.75;#movement factor

    Zlow=2;Zup=5;#latency period

    alphalow=0.02;alphaup=1.0;#reporting rate

    Dlow=2;Dup=5;#infectious period

    xmin=np.array([betalow,mulow,thetalow,Zlow,alphalow,Dlow])

    xmax=np.array([betaup,muup,thetaup,Zup,alphaup,Dup])

    num_loc=pop.shape[0]

    for i in range(num_loc):

        #S

        x[i*5+0,x[i*5+0,:]<0]=0;

        x[i*5+0,x[i*5+0,:]>pop[i,:]]=pop[i,x[i*5+0,:]>pop[i,:]];

        #E

        x[i*5+1,x[i*5+1,:]<0]=0;

        #Ir

        x[i*5+2,x[i*5+2,:]<0]=0;

        #Iu

        x[i*5+3,x[i*5+3,:]<0]=0;

        #obs

        x[i*5+4,x[i*5+4,:]<0]=0

    for i in range(6):

        x[-6+i,x[-6+i,:]<xmin[i]]=xmin[i]*(1+0.1*np.random.random(np.sum(x[-6+i,:]<xmin[i])))

        x[-6+i,x[-6+i,:]>xmax[i]]=xmax[i]*(1-0.1*np.random.random(np.sum(x[-6+i,:]>xmax[i])))

    return x
M = sio.loadmat('../input/M.mat')['M']  # load mobility (375, 375, 14)

pop = sio.loadmat('../input/pop.mat')['pop']  # load population (375,)

Td=9;#average reporting delay

a=1.85;#shape parameter of gamma distribution

b=Td/a;#scale parameter of gamma distribution

rnds=np.ceil(np.random.gamma(a,b,(10000,1))) #pre-generate gamma random numbers

num_loc=M.shape[0] #number of locations

#observation operator: obs=Hx

H=np.zeros((num_loc,5*num_loc+6))

for i in range(num_loc):

    H[i,i*5+4]=1

incidence = sio.loadmat('../input/incidence.mat')['incidence'] #load observation (14, 375)

num_times=incidence.shape[0]

obs_truth=incidence.T

#set OEV

OEV=np.zeros((num_loc,num_times))

for l in range(num_loc):

    for t in range(num_times):

        OEV[l,t]=np.maximum(4,np.square(obs_truth[0,t])/4.)

num_ens=300 #number of ensemble

pop0=pop*np.ones((1,num_ens))

x,paramax,paramin = initialize(pop0,num_ens) #get parameter range

num_var=x.shape[0] #number of state variables

#IF setting

Iter=10;#number of iterations

num_para=paramax.shape[0] #number of parameters

theta=np.zeros((num_para,Iter+1)) #mean parameters at each iteration

para_post=np.zeros((num_para,num_ens,num_times,Iter)) #posterior parameters

sig=np.zeros((Iter)) #variance shrinking parameter

alp=0.9;#variance shrinking rate

SIG=np.square(paramax-paramin)/4 #initial covariance of parameters

Lambda=1.1 #inflation parameter to aviod divergence within each iteration
for n in range(Iter):

    sig[n]=alp**n

    # generate new ensemble members using multivariate normal distribution

    Sigma=np.diag(np.square(sig[n])*SIG)



    if (n==0):

        # first guess of state space

        x,_,_ = initialize(pop0,num_ens)

        para=x[-6:,:]

        theta[:,0]=np.mean(para,1) #mean parameter

    else:

        x,_,_ = initialize(pop0,num_ens)

        para=np.random.multivariate_normal(theta[:,n].T, Sigma, num_ens).T #generate parameters

        x[-6:,:]=para 



    #correct lower/upper bounds of the parameters

    x=checkbound_ini(x,pop0);

    #Begin looping through observations

    x_prior=np.zeros((num_var,num_ens,num_times)) #prior

    x_post=np.zeros((num_var,num_ens,num_times)) #posterior

    pop=pop0

    obs_temp=np.zeros((num_loc,num_ens,num_times)) # records of reported cases



    for t in range(num_times):

        print(n,t)

        #inflation

        x=np.mean(x,1,keepdims=True)*np.ones((1,num_ens))+Lambda*(x-np.mean(x,1,keepdims=True)*np.ones((1,num_ens)))

        x=checkbound(x,pop)

        #integrate forward

        x,pop = SEIR(x,M,pop,t,pop0)

        obs_cnt = np.matmul(H,x) # new infection

        # add reporting delay

        for k in range(num_ens):

            for l in range(num_loc):

                if obs_cnt[l,k]>0:

                    rnd=np.random.choice(rnds, obs_cnt[l,k])

                    for h in range(len(rnd)):

                        if (t+rnd[h]<=num_times):

                            obs_temp[l,k,t+rnd[h]] = obs_temp[l,k,t+rnd(h)]+1

        obs_ens=obs_temp[:,:,t] # observation at t

        x_prior[:,:,t] = x # set prior

        # loop through local observations

        for l in range(num_loc):

            # Get the variance of the ensemble

            obs_var = OEV[l,t]

            prior_var = np.var(obs_ens[l,:])

            post_var = prior_var*obs_var/(prior_var+obs_var);

            if prior_var==0: # if degenerate

                post_var=1e-3

                prior_var=1e-3

            prior_mean = np.mean(obs_ens[l,:])

            post_mean = post_var*(prior_mean/prior_var + obs_truth[l,t]/obs_var)

            #### Compute alpha and adjust distribution to conform to posterior moments

            alpha = np.square(obs_var/(obs_var+prior_var))

            dy = post_mean + alpha*(obs_ens[l,:]-prior_mean)-obs_ens[l,:];

            # Loop over each state variable (connected to location l)

            rr=np.zeros(num_var)

            neighbors=np.union1d(np.where(np.sum(M[:,l,:],1)>0)[0],np.where(np.sum(M[l,:,:],1)>0)[0]);

            neighbors=np.append(neighbors,l) # add location l

            for i in range(len(neighbors)):

                idx=neighbors[i];

                for j in range(5):

                    A=np.cov(x[idx*5+j,:],obs_ens[l,:])

                    rr[idx*5+j]=A[1,0]/prior_var

            for i in range(num_loc*5,num_loc*5+6):

                A=np.cov(x[i,:],obs_ens[l,:])

                rr[i]=A[1,0]/prior_var

            #Get the adjusted variable

            dx=np.matmul(rr[:,None],dy[None,:])

            x=x+dx

            #Corrections to DA produced aphysicalities

            x = checkbound(x,pop)

        x_post[:,:,t]=x

        para_post[:,:,t,n]=x[-6:,:]

    para=x_post[-6:,:,:num_times]

    temp=np.squeeze(np.mean(para,1));#average over ensemble members

    theta[:,n+1]=np.mean(temp,1);#average over time
parameters=theta[:,-1] # estimated parameters

print('|'+'|'.join(['%.3f'%p for p in parameters])+'|')