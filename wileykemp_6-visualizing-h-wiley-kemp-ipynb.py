from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook
# In the SI system
R = .082

# vdw parameters for your chosen gas
a = 1.370/1.0132
b = 0.0387

# Also lay out a range of pressures from 1 to 1000
P = linspace (1, 1000)
T_center = 850
T_interval = 100
T = linspace(T_center-T_interval, T_center+T_interval)
Pgrid,Tgrid = meshgrid(P,T)

# Calculate H(T,P) for the ideal gas (remember to use Tgrid and Pgrid where appropriate)
Hideal = 7/2*R*Tgrid

# Calculate H(T,P) for the real gas (remember to use Tgrid and Pgrid where appropriate)
Hreal = 7/2*R*Tgrid - 2*a*Pgrid/(R*Tgrid) + b*Pgrid

# Graph ideal gas H in 3d
ax = figure().gca(projection='3d') # Set up a three dimensional graphics window 
ax.plot_wireframe(Pgrid, Tgrid, Hideal, rstride=2, cstride=2, color='blue') # Make the mesh plot
ax.set_xlabel('P (atm)') # Label axes
ax.set_ylabel('T (K)')
ax.set_zlabel('H')

# Graph real gas H in 3d
ax = figure().gca(projection='3d') # Set up a three dimensional graphics window 
ax.plot_wireframe(Pgrid, Tgrid, Hreal, rstride=2, cstride=2, color='blue') # Make the mesh plot
ax.set_xlabel('P (atm)') # Label axes
ax.set_ylabel('T (K)')
ax.set_zlabel('H')

# Graph real gas H as a contour plot
figure()
grid(True)
contour(Tgrid, Pgrid, Hreal,linestyles='solid')
xlabel('T(K)') # Label the x axis
ylabel('P(atm)') # Label the y axis
# Calculate and print the inversion temperature from vdw parameters based on your derivation
Tinversion = (2*a)/(b*R)
print(Tinversion)