# import libraries

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt

import datetime
# Import the bitcoin dataset and encode the date

df =pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date

group = df.groupby('date')

Real_Price = group['Weighted_Price'].mean()
# some days are missing 

# I build a vector of daily prices for each day starting from 2014-12-01

# to 2019-01-07

# probably there is some really wasy way to do this in Pandas but I don't know it!

date=datetime.date(2014,12,1)

enddate=datetime.date(2019,1,7)

j=0

BC_Prices=[]

while date<enddate:

    while Real_Price.index[j]<date:

        j=j+1

        if Real_Price.index[j]==date:

            BC_Prices.append(Real_Price[j])

        else:

            BC_Prices.append(np.nan)

    date=date+datetime.timedelta(days=1)

BC_Prices=np.array(BC_Prices)

ndays=BC_Prices.shape[0]



# fill in missing values with local average

for i in range(ndays):

    if np.isnan(BC_Prices[i]):

        j=i

        while j<ndays and np.isnan(BC_Prices[j]):

            j=j+1

        if j<ndays:

            BC_Prices[i]=0.5*(BC_Prices[i-1]+BC_Prices[j])

        else:

            BC_Prices[i]=BC_Prices[i-1]
# plot daily prices 

plt.plot(BC_Prices)

plt.xlabel('day')

plt.ylabel('Bitcoin price')

plt.show()
Portfolio0=np.zeros((ndays,2))

Portfolio0[0,0]=1

Portfolio0[1,0]=1

a='c'

ntrades=0

for t in range(2,ndays):

    if BC_Prices[t-1]>BC_Prices[t-2]:

        Portfolio0[t-1,1]=Portfolio0[t-1,1]+Portfolio0[t-1,0]/BC_Prices[t-1]

        Portfolio0[t-1,0]=0

        if a=='c':

            ntrades=ntrades+1

            a='b'

    else:

        Portfolio0[t-1,0]=Portfolio0[t-1,0]+Portfolio0[t-1,1]*BC_Prices[t-1]

        Portfolio0[t-1,1]=0 

        if a=='b':

            ntrades=ntrades+1

            a='c'

    Portfolio0[t]=Portfolio0[t-1]



V0=Portfolio0[:,0]+Portfolio0[:,1]*BC_Prices
plt.semilogy(V0,label='previous day trend follower')

plt.semilogy(BC_Prices/BC_Prices[0],label='bitcoin investment')

plt.legend()

plt.xlabel('day')

plt.ylabel('portolio value')

plt.show()

print('number of trades = '+str(ntrades))
friction=np.linspace(0,0.02,100)

plt.semilogy(friction,V0[-1]*np.power(1-friction,ntrades),label='previous day trend follower')

plt.semilogy([0,0.02],[BC_Prices[-1]/BC_Prices[0],BC_Prices[-1]/BC_Prices[0]],label='bitcoin investment')

plt.legend()

plt.xlabel('friction rate')

plt.ylabel('final return')

plt.show()
UB=np.zeros(100)

TR=np.zeros(100)

for i in range(100):

    f=friction[i]

    c=1

    b=0

    tc=0

    tb=0

    for t in range(1,ndays):

        if (1-f)*b*BC_Prices[t]>c:

            c=(1-f)*b*BC_Prices[t]

            tc=tb+1

        if (1-f)*c/BC_Prices[t]>b:

            b=(1-f)*c/BC_Prices[t]

            tb=tc+1

    UB[i]=c

    TR[i]=tc
friction=np.linspace(0,0.02,100)

plt.semilogy(friction,V0[-1]*np.power(1-friction,ntrades),label='previous day trend follower')

plt.semilogy([0,0.02],[BC_Prices[-1]/BC_Prices[0],BC_Prices[-1]/BC_Prices[0]],label='bitcoin investment')

plt.semilogy(friction,UB,label='time traveler upper bound')

plt.legend()

plt.xlabel('friction rate')

plt.ylabel('final return')

plt.show()
plt.plot(friction,TR)

plt.xlabel('friction rate')

plt.ylabel('total number of trades made by time traveler')

plt.show()
from scipy.optimize import minimize



class MHH:



    def fit_HMM(self,Prices,nstarts=10):



        R=np.log(Prices[1:]/Prices[:-1])

        n=R.shape[0]

    

        bnds = ((None,None),(None,None),(None,None),(None,None),(None,0),(None, 0),(None, 0), (0, 1),(0, 1), (0, 1),(None,None))



        def HMM_NLL(x):

            sig=np.exp(x[0])

            MU=x[1:4]

            r0,r1,r2=np.exp(x[4:7])

            p0,p1,p2=x[7:10]

            beta=x[10]

            TP=np.array([[1-r0,r0*p0,r0*(1-p0)],[r1*p1,1-r1,r1*(1-p1)],[r2*p2,r2*(1-p2),1-r2]]).T

            P=np.zeros((n+1,3))

            P[0,:]=np.ones(3)/3

            S=np.zeros(n+1)

            rold=0

            for t in range(n):

                P[t+1]=np.matmul(TP,P[t])

                for j in range(3):

                    P[t+1,j]=P[t+1,j]*np.exp(-0.5*((R[t]-rold*beta-MU[j])/sig)**2)/sig

                rold=R[t]

                S[t+1]=max(P[t+1])

                P[t+1]=P[t+1]/S[t+1]

            nll=-np.sum(np.log(S[1:]))

            return nll



        best=np.inf

        for i in range(nstarts):

            mu0=np.random.rand()*0.001

            mu1=np.random.rand()*0.001

            mu2=-np.random.rand()*0.001

            r0=np.random.rand()

            r1=np.random.rand()

            r2=np.random.rand()

            p0=np.random.rand()

            p1=np.random.rand()

            p2=np.random.rand()

            sig=np.random.rand()*0.1

            beta=np.random.rand()*0.1

            x0=np.array([np.log(sig),mu0,mu1,mu2,np.log(r0),np.log(r1),np.log(r2),p0,p1,p2,beta])



            OPT = minimize(HMM_NLL, x0,bounds=bnds)



            if i==0:

                x=OPT.x    

                OPTbest=OPT



            if OPT.fun<best:

                best=OPT.fun

                x=OPT.x

                OPTbest=OPT



        self.sig=np.exp(x[0])

        self.MU=x[1:4]

        r0,r1,r2=np.exp(x[4:7])

        p0,p1,p2=x[7:10]

        self.TP=np.array([[1-r0,r0*p0,r0*(1-p0)],[r1*p1,1-r1,r1*(1-p1)],[r2*p2,r2*(1-p2),1-r2]]).T

        self.beta=x[10]

        self.x=x

        self.OPT=OPT

        

        # reorder so MU is increasing 

        ix=np.argsort(-self.MU)

        self.MU=self.MU[ix]

        self.TP=self.TP[np.ix_(ix,ix)]

        

    def get_hidden_state_probabilities(self,Prices):

            R=np.log(Prices[1:]/Prices[:-1])

            n=R.shape[0]

            P=np.zeros((n+1,3))

            P[0,:]=np.ones(3)/3

            rold=0

            for t in range(n):

                P[t+1]=np.matmul(self.TP,P[t])

                for j in range(3):

                    P[t+1,j]=P[t+1,j]*np.exp(-0.5*((R[t]-self.beta*rold-self.MU[j])/self.sig)**2)/self.sig

                rold=R[t]

                P[t+1]=P[t+1]/np.sum(P[t+1])

            return P

        

    def get_expected_abnormal_rates(self,Prices):

        P=self.get_hidden_state_probabilities(Prices)

        

        R=np.zeros(Prices.shape[0])

        R[1:]=np.log(Prices[1:]/Prices[:-1])

        

        lam,V=np.linalg.eig(self.TP)

        ix=np.argsort(lam)

        lam=lam[ix]

        V=V[:,ix]

        V[:,2]=V[:,2]/np.sum(V[:,2])

        VMU=np.matmul(V.T,self.MU)

        D=(1/(1-hmm.beta))*(lam[:2]/(1-lam[:2]))*VMU[:2]



        EAR=np.matmul(D,np.linalg.solve(V,P.T)[:2,:])+(1/(1-self.beta))*R

        

        return EAR
train_start=100

train_end=600

Prices=BC_Prices[train_start:train_end]



hmm=MHH()

hmm.fit_HMM(Prices)



LP=np.log(Prices)

pmin=np.min(LP)

pmax=np.max(LP)

LP=(LP-pmin)/(pmax-pmin)



P=hmm.get_hidden_state_probabilities(Prices)

plt.plot(range(train_start,train_end),P[:,0],label='h0 - upup')

plt.plot(range(train_start,train_end),P[:,1],label='h1 - up')

plt.plot(range(train_start,train_end),P[:,2],label='h1 - down')

plt.plot(range(train_start,train_end),LP*2+1)

plt.legend()

plt.xlabel('day')

plt.ylabel('probability of hidden state')

plt.show()



EAR=hmm.get_expected_abnormal_rates(Prices)

plt.plot(range(train_start,train_end),EAR)

plt.plot(range(train_start,train_end),(LP*2+1)*np.max(EAR))

plt.xlabel('day')

plt.ylabel('expected abnormal rates')

plt.show()



print('beta = '+str(hmm.beta))

print('\n')

print('MU =')

print(hmm.MU)

print('\n')

print('P =')

print(hmm.TP)
# back train buy/sell policies

res=200

X=np.linspace(-0.1,0.1,res)

def back_train_pol(mdl,Prices):

    P=mdl.get_hidden_state_probabilities(Prices)

    EAR=mdl.get_expected_abnormal_rates(Prices)

    n=P.shape[0]

    best=-np.inf

    R=np.zeros((res,res))

    T=np.zeros((res,res))

    for j in range(res):

        for i in range(j,res):

            buy=X[i] # buy when EAR>buy

            sell=X[j] # sell when EAR<sell

            a='c'

            pc=1

            pb=0

            ntrades=0

            for t in range(n):

                if a=='c' and EAR[t]>buy:

                    pb=pc/Prices[t]

                    pc=0

                    a='b'

                    ntrades=ntrades+1

                if a=='b' and EAR[t]<sell:

                    pc=pb*Prices[t]

                    pb=0

                    a='c'

                    ntrades=ntrades+1

            score=pc+pb*Prices[t]

            R[i,j]=score

            T[i,j]=ntrades

    return R,T
R,T=back_train_pol(hmm,Prices)

Back_train_Return=np.zeros(100)

Buy_Price=np.zeros(100)

Sell_Price=np.zeros(100)

for i in range(100):

    f=friction[i]

    FR=np.log(R)+T*np.log(1-f)

    ix=np.argmax(FR)

    i1=np.mod(ix,res)

    i0=int((ix-i1)/res)

    Back_train_Return[i]=np.exp(FR[i0,i1])

    Buy_Price[i]=X[i0]

    Sell_Price[i]=X[i1]



plt.plot(friction,Buy_Price,label='Buy treshold')

plt.plot(friction,Sell_Price,label='Sell treshold')

plt.xlabel('friction rate')

plt.ylabel('expected abnormal rate')

plt.legend()

plt.show()



plt.plot(friction,Back_train_Return,label='optimized buy/sell policy')

plt.plot([0,np.max(friction)],[Prices[-1]/Prices[0],Prices[-1]/Prices[0]],label='Bitcoin investment')

plt.ylim([0, 1.1*np.max(Back_train_Return)])

plt.xlabel('friction rate')

plt.ylabel('back training return')

plt.legend()

plt.show()
# back test

def back_test_pol(mdl,friction,Buy_Price,Sell_Price,Prices):

    P=mdl.get_hidden_state_probabilities(Prices)

    EAR=mdl.get_expected_abnormal_rates(Prices)

    n=P.shape[0]

    Portfolios=np.zeros((100,n,2))

    Value=np.zeros((100,n))

    for i in range(100):

        buy=Buy_Price[i]

        sell=Sell_Price[i]

        rate=1-friction[i]

        Portfolios[i,0,0]=1

        a='c'

        for t in range(n-1):

            if a=='c' and EAR[t]>buy:

                Portfolios[i,t,1]=rate*Portfolios[i,t,0]/Prices[t]

                Portfolios[i,t,0]=0

                a='b'

            if a=='b' and EAR[t]<sell:

                Portfolios[i,t,0]=rate*Portfolios[i,t,1]*Prices[t]

                Portfolios[i,t,1]=0

                a='c'

            Portfolios[i,t+1]=Portfolios[i,t]

        Value[i]=Portfolios[i,:,0]+Portfolios[i,:,1]*Prices

    return Portfolios,Value
Portfolios1,V1=back_test_pol(hmm,friction,Buy_Price,Sell_Price,BC_Prices)
ntrades_in_test_period0=np.sum(Portfolio0[train_end:-1]*Portfolio0[train_end+1:]>0)

plt.semilogy(friction,V0[-1]/V0[train_end]*np.power(1-friction,ntrades_in_test_period0),label='previous day trend follower')



plt.semilogy(friction,V1[:,-1]/V1[:,train_end],label='back-train optimized HMM buy/sell policy')

plt.semilogy([0,np.max(friction)],[BC_Prices[-1]/BC_Prices[train_end],BC_Prices[-1]/BC_Prices[train_end]],label='Bitcoin investment')

plt.legend()

plt.xlabel('friction rate')

plt.ylabel('back testing return')

plt.show()
i=25

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 0.5% market friction')

plt.legend()

plt.show()
i=50

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 1% market friction')

plt.legend()

plt.show()
i=75

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 1.5% market friction')

plt.legend()

plt.show()
s=0

r=0

n=5000

Rsampled=np.zeros(n)

for t in range(n):

    s=np.random.choice(range(3),p=hmm.TP[:,s])

    r=hmm.beta*r+hmm.MU[s]+np.random.randn()*hmm.sig

    Rsampled[t]=r

Psampled=np.exp(np.cumsum(Rsampled))

plt.semilogy(Psampled)

plt.ylabel('simulated price data sampled from HMM')

plt.xlabel('day')

plt.show()
R,T=back_train_pol(hmm,Psampled)

Back_train_Return=np.zeros(100)

Buy_Price=np.zeros(100)

Sell_Price=np.zeros(100)

for i in range(100):

    f=friction[i]

    FR=np.log(R)+T*np.log(1-f)

    ix=np.argmax(FR)

    i1=np.mod(ix,res)

    i0=int((ix-i1)/res)

    Back_train_Return[i]=np.exp(FR[i0,i1])

    Buy_Price[i]=X[i0]

    Sell_Price[i]=X[i1]



plt.plot(friction,Buy_Price,label='Buy treshold')

plt.plot(friction,Sell_Price,label='Sell treshold')

plt.xlabel('friction rate')

plt.ylabel('expected abnormal rate')

plt.legend()

plt.show()

    

plt.plot(friction,Back_train_Return,label='optimized buy/sell policy')

plt.plot([0,np.max(friction)],[Psampled[-1]/Psampled[0],Psampled[-1]/Psampled[0]],label='Bitcoin investment')

plt.ylim([0, 1.1*np.max(Back_train_Return)])

plt.xlabel('friction rate')

plt.ylabel('back training return')

plt.legend()

plt.show()
# back test

def back_test_pol(mdl,friction,Buy_Price,Sell_Price,Prices):

    P=mdl.get_hidden_state_probabilities(Prices)

    EAR=mdl.get_expected_abnormal_rates(Prices)

    n=P.shape[0]

    Portfolios=np.zeros((100,n,2))

    Value=np.zeros((100,n))

    for i in range(100):

        buy=Buy_Price[i]

        sell=Sell_Price[i]

        rate=1-friction[i]

        Portfolios[i,0,0]=1

        a='c'

        for t in range(n-1):

            if a=='c' and EAR[t]>buy:

                Portfolios[i,t,1]=rate*Portfolios[i,t,0]/Prices[t]

                Portfolios[i,t,0]=0

                a='b'

            if a=='b' and EAR[t]<sell:

                Portfolios[i,t,0]=rate*Portfolios[i,t,1]*Prices[t]

                Portfolios[i,t,1]=0

                a='c'

            Portfolios[i,t+1]=Portfolios[i,t]

        Value[i]=Portfolios[i,:,0]+Portfolios[i,:,1]*Prices

    return Portfolios,Value
Portfolios1,V1=back_test_pol(hmm,friction,Buy_Price,Sell_Price,BC_Prices)
ntrades_in_test_period0=np.sum(Portfolio0[train_end:-1]*Portfolio0[train_end+1:]>0)

plt.semilogy(friction,V0[-1]/V0[train_end]*np.power(1-friction,ntrades_in_test_period0),label='previous day trend follower')



plt.semilogy(friction,V1[:,-1]/V1[:,train_end],label='back-train optimized HMM buy/sell policy')

plt.semilogy([0,np.max(friction)],[BC_Prices[-1]/BC_Prices[train_end],BC_Prices[-1]/BC_Prices[train_end]],label='Bitcoin investment')

plt.legend()

plt.xlabel('friction rate')

plt.ylabel('back testing return')

plt.show()
i=25

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 0.5% market friction')

plt.legend()

plt.show()
i=50

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 1% market friction')

plt.legend()

plt.show()
i=75

plt.plot(range(train_end,ndays),V1[i,train_end:]/V1[i,train_end],label='HMM buy/sell policy')

plt.plot(range(train_end,ndays),BC_Prices[train_end:]/BC_Prices[train_end],label='Bitcoin investment')

plt.xlabel('day')

plt.ylabel('portfolio value')

plt.title('simulation with 1.5% market friction')

plt.legend()

plt.show()