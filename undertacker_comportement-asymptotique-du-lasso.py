import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
def build_design_matrix(n = 1000):
    """
    Build the matrix design for the breakpoint detection with LASSO.
    The design matrix is a triangular matrix with only one on and below the diagonal.
    
    :parameters:
    - n : (int, 1000) the dimention of the matrix, also the number of observations
    
    :return:
    - X : the design matrix"""
    X = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            X[i,j] = 1
    return X

def break_points_list(D = 10, sigma_signal = 2):
    """
    Generate a list of breakpoints position and the size of the corresponding jumps
    
    :parameters:
    - D : (int, 10) the number of breakpoint detection
    - sigma_signal : (float, 2) the standart deviation of the jump size at breakpoints
    
    :return:
    As tuple:
    - t_i : list of breakpoint normalized positions
    - Jp : Jump size
    """
    t_i = np.random.random(D+1)
    t_i[0] = 0
    Jp = np.random.randn(D+1)*sigma_signal
    return t_i, Jp

def model_from_bklist(t_i, Jp, n, sigma_noise = 1):
    """
    Take a list of jump positions and sizes to generate a LASSO Model.
    
    :parameters:
    - t_i : list of breakpoint normalized positions
    - Jp : Jump size
    - n : number of observations
    
    :return:
    - Y : observations vector
    - beta : breakpoints vector
    - E : noise vector
    - S : sparse vector
    """
    E = np.random.randn(n)*sigma_noise
    S = (t_i*n).astype("int")
    beta = np.zeros(n)
    beta[S] = Jp
    X = build_design_matrix(n)
    Y = X.dot(beta) + E
    return Y, beta, E, S, X
    
t_i, Jp = break_points_list(D = 10)
Y, beta, E, S, X = model_from_bklist(t_i, Jp, n = 1000)
plt.plot(Y, "r*")
plt.vlines(S, ymin = Y.min(), ymax= Y.max())
plt.plot(X.dot(beta))
C = 3
def LASSO(Y):
    """
    Return the beta hat estimator
    
    :parameters:
    - Y : observations vector
    
    :return:
    - beta_hat : beta hat estimator"""
    n = Y.shape[0]
    _lambda = C*np.sqrt(np.log(n)/n)
    X = build_design_matrix(n = Y.shape[0])
    LR = Lasso(alpha=_lambda)
    LR.fit(X, Y)
    return LR.coef_
t_i, Jp = break_points_list(D = 10)
Y, beta, E, S, X = model_from_bklist(t_i, Jp, n = 1000)
beta_hat = LASSO(Y)
print("beta_hat_S = \n", beta_hat[S])
def norm_X(X, beta, beta_hat):
    return np.sqrt((X.dot(beta_hat-beta)).T.dot((X.dot(beta_hat-beta))))


def scalar_product(X, beta, beta_hat, S, Sc):
    return (X[:,Sc].dot(beta_hat[Sc])).T.dot((X[:,S].dot(beta_hat[S]-beta[S])))

def norm_S(X, beta, beta_hat, S):
    return np.sqrt((X[:,S].dot(beta_hat[S]-beta[S])).T.dot((X[:,S].dot(beta_hat[S]-beta[S]))))

def norm_Sc(X, beta, beta_hat, Sc):
    return np.sqrt((X[:,Sc].dot(beta_hat[Sc])).T.dot((X[:,Sc].dot(beta_hat[Sc]))))

def eta_factor(S, beta, beta_hat, X, n):
    return scalar_product(X, beta, beta_hat, S, Sc)/(norm_S(X, beta, beta_hat, S)*norm_Sc(X, beta, beta_hat, Sc))

def one_step(t_i, Jp, n):
    Y, beta, E, S, X = model_from_bklist(t_i, Jp, n)
    Sc = list(set(range(n)) - set(S))
    beta_hat = LASSO(Y)
    nX = norm_X(X, beta, beta_hat)/np.sqrt(n)
    nXs = norm_S(X, beta, beta_hat, S)/np.sqrt(n)
    nXsc = norm_Sc(X, beta, beta_hat, Sc)/np.sqrt(n)
    ps = scalar_product(X, beta, beta_hat, S, Sc)/n
    if (nXs*nXsc) != 0:
        eta = ps/(nXs*nXsc)
    else :
        eta = "NAN"
    
    return {"nX":nX,
           "nXs":nXs,
           "nXsc":nXsc,
           "ps":ps,
           "eta":eta}
one_step(t_i, Jp, n=1000)
LnX = []
LnXs = []
LnXsc = []
Lps = []
Leta = []
for n in range(20, 3000):
    if n%10 == 0:
        print(f"progress : {n/(3000-20)*100:.2f}%", flush = True, end = "\r")
    res = one_step(t_i, Jp, n=n)
    LnX.append(res['nX']) 
    LnXs.append(res['nXs']) 
    LnXsc.append(res['nXsc']) 
    Lps.append(res['ps']) 
    Leta.append(res['eta'])
plt.plot([r for r in Leta if r !='NAN'], "*")
plt.plot(LnXs)
plt.plot(LnXsc, "*")
pLnX = []
pLnXs = []
pLnXsc = []
pLps = []
pLeta = []
for n in range(1000):
    if n%10 == 0:
        print(f"progress : {n/(1000)*100:.2f}%", flush = True, end = "\r")
    res = one_step(t_i, Jp, n=200)
    pLnX.append(res['nX']) 
    pLnXs.append(res['nXs']) 
    pLnXsc.append(res['nXsc']) 
    pLps.append(res['ps']) 
    pLeta.append(res['eta'])
np.array(pLnX) - np.array(pLnXs)
plt.hist(pLnXsc)
plt.hist([r for r in pLeta if r !='NAN'])
gLnX = []
gLnXs = []
gLnXsc = []
gLps = []
gLeta = []
for n in range(1000):
    if n%10 == 0:
        print(f"progress : {n/(1000)*100:.2f}%", flush = True, end = "\r")
    res = one_step(t_i, Jp, n=1000)
    gLnX.append(res['nX']) 
    gLnXs.append(res['nXs']) 
    gLnXsc.append(res['nXsc']) 
    gLps.append(res['ps']) 
    gLeta.append(res['eta'])
plt.hist(gLnXsc)
plt.hist([r for r in gLeta if r !='NAN'])
np.array(gLnX) - np.array(gLnXs)
gLnX = []
gLnXs = []
gLnXsc = []
gLps = []
gLeta = []
for n in range(100):
    if n%10 == 0:
        print(f"progress : {n/(100)*100:.2f}%", flush = True, end = "\r")
    res = one_step(t_i, Jp, n=5000)
    gLnX.append(res['nX']) 
    gLnXs.append(res['nXs']) 
    gLnXsc.append(res['nXsc']) 
    gLps.append(res['ps']) 
    gLeta.append(res['eta'])
plt.hist(gLnXsc)
plt.hist([r for r in gLeta if r !='NAN'])
np.array(gLnX) - np.array(gLnXs)