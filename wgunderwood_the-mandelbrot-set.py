import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# The total number of complex numbers considered (number of pixels) is n x n
n = 5000

# The maximum number of iterations in testing escape
maxiter = 500

# The plot size in inches
size = 15
xs = np.linspace(-2,1,n)
ys = np.linspace(-1.5,1.5,n)
Z0 = np.array(xs).reshape((1,n)) + np.array(ys).reshape((n,1))*1j
Z = np.zeros((n,n))
T = np.zeros((n,n))

for k in range(maxiter):
    Z = Z**2 + Z0
    T += np.int_(T==0) * (abs(Z) > 2) * k
fig = plt.figure(figsize=(size,size))
plt.imshow(T**0.5,cmap='magma',interpolation='gaussian',aspect='equal',vmax=0.5*maxiter**0.5)
plt.xlabel('Re(c)',fontsize=30)
plt.ylabel('Im(c)',fontsize=30)
plt.xticks([i*n/6-0.5 for i in range(7)],[i for i in np.arange(-2,1.5,0.5)],fontsize=20)
plt.yticks([i*n/6-0.5 for i in range(7)],[i for i in np.arange(1.5,-2,-0.5)],fontsize=20)
plt.tick_params(pad=10,length=10)
plt.title('Figure 1: The Mandelbrot Set',fontsize=40,pad=20)
fig.show()
# The total number of pixels is n x n
n = 5000

# The maximum number of iterations in testing escape
maxiter = 500

# The total number of random points in the simulation
m = int(2e8)

# The plot size in inches
size = 15 
# Generate random points
Z0 = np.random.random(m)*4-2 + (np.random.random(m)*4-2)*1j

# Remove points outside radius-2 circle around 0
Z0 = Z0[abs(Z0)<2]

# Remove points in cardioid
p = (((Z0.real-0.25)**2) + (Z0.imag**2))**.5
Z0 = Z0[Z0.real > p-(2*p**2) + 0.25]

# Remove points in period-2 bulb
Z0 = Z0[((Z0.real+1)**2) + (Z0.imag**2) > 0.0625]
Z = Z0.copy()
to_try = np.ones_like(Z0,dtype=bool)

for k in range(maxiter):
    Z[to_try] = Z[to_try]**2 + Z0[to_try]
    to_try[abs(Z)>2] = False
    
Z0 = Z0[np.logical_not(to_try)]
# Create pixels array
B = np.zeros([n,n])

Z = Z0.copy()
total_length = len(Z)

while(len(Z)):
    x = np.array((Z.real+2)/4 * n,int)
    y = np.array((Z.imag+2)/4 * n,int)
    B[y,x] += 1
    B[n-1-y,x] += 1
    Z = Z**2 + Z0
    keep = abs(Z)<2
    Z0 = Z0[keep]
    Z = Z[keep]
fig = plt.figure(figsize=(size,size))
plt.imshow(B.T**0.5,cmap='magma',interpolation='gaussian',aspect='equal',vmin=2)
plt.xlabel('Re(z)',fontsize=30)
plt.ylabel('Im(z)',fontsize=30)
plt.xticks([i*n/6-0.5 for i in range(7)],[i for i in np.arange(-2,1.5,0.5)],fontsize=20)
plt.yticks([i*n/6-0.5 for i in range(7)],[i for i in np.arange(1.5,-2,-0.5)],fontsize=20)
plt.tick_params(pad=10,length=10)
plt.title('Figure 2: The Buddhabrot',fontsize=40,pad=20)
fig.show()