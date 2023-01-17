import pylab as pl
import numpy as np

gaussian = lambda x: 1/(np.sqrt(2*np.pi)*1.5)*np.exp(-(x-0)**2/(2*(1.5**2)))
x = np.arange(-5,5,0.01)
y = gaussian(x)
pl.ion()
pl.plot(x,y,'k',linewidth=3)
pl.xlabel('x')
pl.ylabel('y(x)')
pl.axis([-5,5,0,0.3])
pl.title('Gaussian Function (mean 0, standard deviation 1.5)')
pl.show()
x = np.arange(-5,5,0.01)
s = 1
mu = 0
y = 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(x-mu)**2/s**2)
pl.plot(x,y,'k')

pl.close('all')
mu = np.array([2,-3])
s = np.array([1,1])
#s = array([0.5,2])
x = np.random.normal(mu,scale=s,size = (500,2))
pl.plot(x[:,0],x[:,1],'ko')
#axis(array([0,3,-8,4]))
pl.axis('equal')

theta = np.arange(0,2.1*np.pi,np.pi/20)

pl.plot(mu[0]+2*np.cos(theta),mu[1]+2*np.sin(theta),'k-')
pl.plot(mu[0]+3*np.cos(theta),mu[1]+3*np.sin(theta),'k-')


pl.figure()

mu = np.array([2,-3])
s = np.array([0.5,2])
x = np.random.normal(mu,scale=s,size = (500,2))
phi = 2*np.pi/3
pl.plot(x[:,0]*np.cos(phi)+x[:,1]*np.sin(phi),x[:,0]*(-np.sin(phi)) + x[:,1]*np.cos(phi),'ko')
pl.axis('equal')

theta = np.arange(0,2.1*np.pi,np.pi/20)
pl.plot((mu[0]+3*s[0]*np.cos(theta))*np.cos(phi)+(mu[1]+3*s[1]*np.sin(theta))*np.sin(phi), (mu[0]+3*s[0]*np.cos(theta))*np.sin(-phi)+(mu[1]+3*s[1]*np.sin(theta))*np.cos(phi), 'k-')

pl.figure()
mu = np.array([2,-3])
s = np.array([0.5,2])
x = np.random.normal(mu,scale=s,size = (500,2))
pl.plot(x[:,0],x[:,1],'ko')
pl.axis('equal')

theta = np.arange(0,2.1*np.pi,np.pi/20)
pl.plot(mu[0]+3*s[0]*np.cos(theta),mu[1]+3*s[1]*np.sin(theta), 'k-')

pl.show()