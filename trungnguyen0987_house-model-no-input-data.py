# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install gekko
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import pandas as pd
import math
from pathlib import Path
import os
def qsun(tclim,
        Dh,
        En,
        gamma,
        beta,
        rground):
        """    (scalar) Dh = diffuse horizontal irradiation[W/m2]
        (scalar) En = direct normal irradiation [W/m2]
        (scalar) t = time seconds after midnight 1 january
        (scalar) gamma = azimuth angle of the surface,
        east:gamma = -90, west:gamma = 90
        south:gamma = 0, north:gamma = 180
        (scalar) beta = inclination angle of the surface,
        horizontal: beta=0, vertical: beta=90
        default geographical position: De Bilt
        default ground reflectivity (albedo): 0.2"""
    
    #(scalar) iday = day of the year (1-365)
    #(scalar) LST = Local Standard time (0 - 23) [hour]

    
        iday = 1+math.floor(tclim/(24*3600))
        LST  = math.floor((tclim/3600) % 24)
        # L = Latitude [graden]
        L    = 52.1
        # LON = Local Longitude [graden] oost is positief
        LON  = 5.1
        # LSM = Local Standard time Meridian [graden] oost is positief
        LSM  = 15*1
        # rground = albedo
        # rground=0.2;
        r    = math.pi/180;
        L    = L*r
        beta = beta*r
        theta= 2*math.pi*(iday-1)/365.25
        el   = 4.901+0.033*math.sin(-0.031+theta)+theta  # elevation
        # declination

        delta= math.asin(math.sin(23.442*r)*math.sin(el))
        q1   = math.tan(4.901+theta)
        q2   = math.cos(23.442*r)*math.tan(el)
        # equation of time

        ET   = (math.atan((q1-q2)/(q1*q2+1)))*4/r
        AST  = LST+ET/60+(4/60)*(LSM-LON)                    # change from minus to plus
        h    = (AST-12)*15*r
        
        # hai=sin(solar altitude), leght of incline surface??? check sun position with the surface
        hai  = math.cos(L)*math.cos(delta)*math.cos(h)+math.sin(L)*math.sin(delta)
        
       
        E    = np.zeros((1,4))
        # teta = incident angle on the tilted surface
        teta = 0
        #determination of zet = solar zenith angle (pi/2 - solar altitude).        
        zet=0
        # calculation of extraterrestrial radiation
        Eon=0
        if hai>0:
            # salt=solar altitude

            salt= math.asin(hai)
            phi = math.acos((hai*math.sin(L)-math.sin(delta))/(math.cos(salt)*math.cos(L)))*np.sign(h)
            gam = phi-gamma*r;
            # cai=cos(teta)
            cai = math.cos(salt)*math.cos(abs(gam))*math.sin(beta)+hai*math.cos(beta)
            # teta = incident angle on the tilted surface
            teta= math.acos(cai)
            # salts=solar altitude for an inclined surface

            salts=math.pi/2-teta
            # Perez (zie Solar Energy volume 39 no. 3)
            # berekening van de diffuse straling op een schuin vlak
            # Approximatin of A and C, the solid angles occupied by the circumsolar region,
            # weighed by its average incidence on the slope and horizontal respectively.
            # In the expression of diffuse on inclined surface the quotient of A/C is
            # reduced to XIC/XIH. A=2*(1-cos(beta))*xic, C=2*(1-cos(beta))*xih
            # gecontroleerd okt 1996 martin de wit

            # alpha= the half-angle circumsolar region
            alpha=25*r

            if salts<-alpha:

                xic=0

            elif salts>alpha:

                xic=cai

            else:
                xic=0.5*(1+salts/alpha)*math.sin((salts+alpha)/2)


            if salt>alpha:

                xih=hai

            else:
                xih=math.sin((alpha+salt)/2)


            epsint=[1.056 ,1.253 ,1.586, 2.134, 3.23 ,5.98, 10.08 ,999999]
            f11acc=[-0.011 ,-0.038 ,0.166 ,0.419 ,0.710 ,0.857 ,0.734 ,0.421]
            f12acc=[0.748 ,1.115 ,0.909 ,0.646 ,0.025 ,-0.370 ,-0.073 ,-0.661]
            f13acc=[-0.080 ,-0.109 ,-0.179 ,-0.262 ,-0.290 ,-0.279 ,-0.228 ,0.097]
            f21acc=[-0.048 ,-0.023 ,0.062 ,0.140 ,0.243 ,0.267 ,0.231 ,0.119]
            f22acc=[0.073 ,0.106 ,-0.021 ,-0.167 ,-0.511 ,-0.792 ,-1.180 ,-2.125]
            f23acc=[-0.024 ,-0.037 ,-0.050 ,-0.042 ,-0.004 ,0.076 ,0.199 ,0.446]
            
            # determination of zet = solar zenith angle (pi/2 - solar altitude).
            
            zet=math.pi/2-salt
            
            # determination of inteps with eps

            inteps=0

            if Dh>0:

                eps=1+En/Dh
                inteps=7     # give big random number for starting point
                for i in range(len(epsint)):
                    if epsint[i]>=eps:
                        temp_i =i        
                        inteps=min(temp_i,inteps)
                #print(inteps)
                
                #inteps=min(i)

            # calculation of inverse relative air mass

            airmiv=hai

            if salt<10*r:
                airmiv=hai+0.15*(salt/r+3.885)**(-1.253)   #change ^ to **
            # calculation of extraterrestrial radiation

            Eon=1370*(1+0.033*math.cos(2*math.pi*(iday-3)/365))
            
            #delta is "the new sky brightness parameter"

            delta=Dh/(airmiv*Eon)

            # determination of the "new circumsolar brightness coefficient
            # (f1acc) and horizon brightness coefficient (f2acc)"// Why use circumsolar
            
            f1acc=f11acc[inteps]+f12acc[inteps]*delta+f13acc[inteps]*zet
            f2acc=f21acc[inteps]+f22acc[inteps]*delta+f23acc[inteps]*zet
            
            # determination of the diffuse radiation on an inclined surface

            E[0,0]=Dh*(0.5*(1+math.cos(beta))*(1-f1acc)+f1acc*xic/xih+f2acc*math.sin(beta))

            if E[0,0]<0:

                E[0,0]=0

            # horizontal surfaces treated separately
            # beta=0 : surface facing up, beta=180(pi) : surface facing down
            #3/22/19 10:05 AM E:\NEN5060\qsun.m 4 of 4
            
            if beta>-0.0001 and beta<0.0001:

                E[0,0]=Dh

            if beta>(math.pi-0.0001) and beta<(math.pi+0.0001):

                E[0,0]=0
                
             # Isotropic sky
             # E(1)=0.5*(1+cos(beta))*Dh;

             # direct solar radiation on a surface

            E[0,1]=En*cai    # beam  aoi projection, beam_component

            if E[0,1]<0.0:

                E[0,1]=0

            # the ground reflected component: assume isotropic
            
            # ground conditions.

            Eg=0.5*rground*(1-math.cos(beta))*(Dh+En*hai)
            
            # global irradiation

            E[0,3]=Dh+En*hai

             # total irradiation on an inclined surface
            E[0,2]=E[0,0] + E[0,1] + Eg
            
        return E[0,2]

print(Path.cwd())
#print(os.listdir("../kaggle/input"))
parent_dir = os.chdir('/kaggle/')
data_dir = Path.cwd() /'input/nen2018'
#output_dir = Path.cwd()/'working'/'submit'
NENdata_path = data_dir/'NEN5060-2018.xlsx'
xls = pd.ExcelFile(NENdata_path)    
xls.sheet_names  # Check sheet names
"""Select sheet in NEN data files.

     - k=1 by NEN default by default
     - k=2,3 for building construction purposes read NEN docs for more info.

"""
k=1
""" Exchange of NEN5060 data in climate files for Python

    Exchange of NEN5060_2018 data in Excel
    For irradiation on S, SW, W, NW, N, NE, E, SE and horizontal and Toutdoor
    Irradiation can be used for solar irradiation on windows
    Matlab version September 17th 2018 by Arie Taal THUAS (The Hague University of Applied Sciences)
    Python version 28/05/2020 by Trung Nguyen HAN University of Applied Sciences

"""


rground=0  #ground reflection is ignored
#______________##________________________

if k==1:
    NUM = pd.read_excel(xls,'nen5060 - energie') # this file is part of NEN 5060 20018
elif k==2:
    NUM = pd.read_excel(xls,'ontwerp 1%')
elif k==3:
    NUM = pd.read_excel(xls,'ontwerp 5%')
#Convert data frame to array
to_array=NUM.to_numpy()
to_array=np.delete(to_array, 0, 0)

#______________##________________________
    
dom=to_array[:,2] # day of month
hod=to_array[:,3] # hour of day
qglob_hor=to_array[:,4]
qdiff_hor=to_array[:,5]
qdir_hor=to_array[:,6]
qdir_nor=to_array[:,7]
Toutdoor=to_array[:,8]/10
phioutdoor=to_array[:,9]
xoutdoor=to_array[:,10]/10
pdamp=to_array[:,11]
vwind=to_array[:,12]/10  #% at 10 m height
dirwind=to_array[:,13]
cloud=to_array[:,14]/10
rain=to_array[:,15]/10

#______________##__________________________

t=(np.array(list(range(1,8761)))-1)*3600
E= np.zeros((8760,9))

n=0
k=1

'''
(scalar) gamma = azimuth angle of the surface,
    east:gamma = -90 (270), west:gamma = 90
    south:gamma = 0, north:gamma = 180
    (scalar) beta = inclination angle of the surface,
    horizontal: beta=0, vertical: beta=90
'''

for j in range(9):
    j=j+1
    
    if j<10:
        
        gamma=45*(j-1)    #gamma 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE), 270 (E), 315(SE) , 360
        beta=90
        #print(j)
        #print(gamma)
    else:
            
        gamma=90
        beta=0
    
    #k=k+1
    
    for i in range(8760):
    
        E[i,n]=qsun(t[i],qdiff_hor[i],qdir_nor[i],gamma,beta,rground)
    n=n+1

#______________##________________________

myarray = np.asarray(E)       
#myarray = np.delete(myarray,8759, 0)        
qsunS=np.vstack((t,myarray[:,0]))
qsunSW=np.vstack((t,myarray[:,1]))
qsunW=np.vstack((t,myarray[:,2]))
qsunNW=np.vstack((t,myarray[:,3]))
qsunN=np.vstack((t,myarray[:,4]))
qsunNE=np.vstack((t,myarray[:,5]))
qsunE=np.vstack((t,myarray[:,6]))
qsunSE=np.vstack((t,myarray[:,7]))
qsunhor=np.vstack((t,myarray[:,8]))
Tout=np.vstack((t,Toutdoor))
phiout=np.vstack((t,phioutdoor))
xout=np.vstack((t,xoutdoor))
pout=np.vstack((t,pdamp))
vout=np.vstack((t,vwind))
dirvout=np.vstack((t,dirwind))
cloudout=np.vstack((t,cloud))
rainout=np.vstack((t,rain))   
            
#Np.newaxis convert an 1D array to either a row vector or a column vector.
"""Create 2D array from 1D array with 2 elements"""
qsunS=qsunS[np.newaxis]     
qsunSW=qsunSW[np.newaxis]     
qsunW=qsunW[np.newaxis]     
qsunNW=qsunNW[np.newaxis]     
qsunN=qsunN[np.newaxis]     
qsunNE=qsunNE[np.newaxis]     
qsunE=qsunE[np.newaxis]     
qsunSE=qsunSE[np.newaxis]     
qsunhor=qsunhor[np.newaxis]     
Tout=Tout[np.newaxis]     
phiout=phiout[np.newaxis]     
xout=xout[np.newaxis]     
pout=pout[np.newaxis]     
vout=vout[np.newaxis]     
dirvout=dirvout[np.newaxis]     
cloudout=cloudout[np.newaxis]     
rainout=rainout[np.newaxis]        
"""Transpose array"""
qsunS=qsunS.T
qsunSW=qsunSW.T
qsunW=qsunW.T
qsunNW=qsunNW.T
qsunN=qsunN.T
qsunNE=qsunNE.T
qsunE=qsunE.T
qsunSE=qsunSE.T
qsunhor=qsunhor.T
Tout=Tout.T
phiout=phiout.T
xout=xout.T
pout=pout.T
vout=vout.T
dirvout=dirvout.T
cloudout=cloudout.T
rainout=rainout.T   
T_outdoor=Tout[:,1]
#numrows = len(input)    # 3 rows in your example
#numcols = len(input[0]) # 2 columns in your example

plt.plot(Tout[:,0],Toutdoor,label=r'$T_1$ Toutdoor')
plt.ylabel('Temperature (degC)')
plt.xlabel('time (sec)')
plt.legend(loc=2)
import gc
gc.collect() # collect garbages
"""window surface in m2: [S SW W NW N NE E SE]"""
glass=[9.5,9.5,0,0,0,0,0,0]
"""Window solar transmitance, g-value"""
g_value =0.7
"""Time base on 1 hour sampling from NEN"""
time=qsunS[:,0]
print(len(time))
Qsolar = (qsunS[:,1]*glass[0] + qsunSW[:,1]*glass[1] + 
                      qsunW[:,1]*glass[2] + qsunNW[:,1]*glass[3] + 
                      qsunN[:,1]*glass[4] + qsunNE[:,1]*glass[5] + 
                      qsunE[:,1]*glass[6] + qsunSE[:,1]*glass[7]) * g_value
print(len(Qsolar[0]))
print(len(Qsolar))
plt.plot(time,Qsolar)
from scipy import signal

DeltaQ     = 150                     #Internal heat gain difference between day and night
#day_DeltaQ = DeltaQ                 #Day Delta Q internal [W]
Qday       = 400                     #Day internal heat gain W
nightQ     = Qday - DeltaQ           #Night internal heat gain

t1= 8                                #Presence from [hour]
t2= 23                               #Presence until [hour]

days_hours   = 24                    #number_of_hour_in_oneday + start hour at 0
days         = 365                   #number of simulation days
periods      = 24*3600*days          #in seconds (day_periods*365 = years)
pulse_width  = (t2-t1)/24            # % of the periods
phase_delay  = t1                    #in seconds


#t = np.linspace(0, 24*3600, 24)
t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0
pulseday = signal.square(2 * np.pi* days * t,duty=pulse_width)
pulseday = np.clip(pulseday, 0, 1)
# add delay to array
pulseday=np.roll(pulseday,phase_delay)

#______pulse week generator______________

week = days/7
pulse_w  = 0.99

#t = np.linspace(0, 24*3600, 24)
pulse_week = signal.square(2*np.pi*week*t,duty=pulse_w)
pulse_week = np.clip(pulse_week, 0, 1)

#pulse_week=np.roll(pulse_week,phase_delay)

#create simulation time
time_t = np.linspace(0,periods,(days_hours*days)+1)

#Internal heat gain

Qinternal = nightQ + pulseday*DeltaQ*pulse_week
Qinternal=Qinternal[np.newaxis]
Qinternal=Qinternal.T

#Plot 48 hours

plt.plot(time_t[0:48], Qinternal[0:48])
plt.ylabel('Internal heat gain (W)')
plt.xlabel('time (sec)')
plt.legend(loc=2)
#print(Qinternal)
Qinternal=np.delete(Qinternal, -1, 0)
""" Initialization Dwelling
      
        - Arie Taal, Baldiri Salcedo HHS 10 July 2018
        - The incident solar heat is divided between Cwall and Cair by the convection factor (CF=0.8)
        - Qinst :  isntant  by heating or cooling needed at this moments
        
        Last modify by Trung Nguyen
"""

"""Envelope surface (facade + roof + ground) [m2]"""
A_facade = 160.2
"""Envelope thermal resitance, R-value [m2/KW]"""
Rc_facade =1.3

"""
- Windows surface [N,S,E,W,SE,SW,NE,NW] [m2]
- glass =[0,0,9.5,9.5,0,0,0,0]
- Window thermal transmittance, U-value [W/m2K]
"""

Uglass = 2.9
"""Window solar transmitance, g-value"""
g_value =0.7
CF=0.8
"""Ventilation, air changes per hour [#/h]"""
n=0.55
"""Internal volume [m3]"""
V_dwelling = 275.6
"""Facade construction

    - Middle_weight =2 Light_weight =1 / Heavy_weight
"""
N_facade  = 2
"""Floor and internal walls surface [m2]"""
A_internal_mass = 300
"""Floor and internal walls construction

    - Middle_weight =2 / Light_weight=1 / Heavy_weight

"""
N_internal_mass = 2             

N_internal_mass = 2;   """1: Light weight construction / 2: Middle weight construction / 3: Heavy weight construction""" 
N_facade = 2;          """1: Light weight construction / 2: Middle weight construction / 3: Heavy weight construction"""
"""Initial parameters file for House model"""

##Predefined variables 
 
rho_air = 1.20;              """Density air in [kg/m3] """
c_air  = 1005;               """Specific heat capacity air [J/kgK] """ 
alpha_i_facade = 8;
alpha_e_facade = 23;
alpha_internal_mass = 8;

""" 
Variables from Simulink model, dwelling mask
Floor and internal walls construction.
It is possible to choose between light, middle or heavy weight construction

"""

#Light weight construction  
if N_internal_mass==1:          
    
    c_internal_mass=840;         """Specific heat capacity construction [J/kgK] """
    th_internal_mass=0.1;        """Construction thickness [m] """
    rho_internal_mass=500;       """Density construction in [kg/m3] """
    
#Middle weight construction      
elif N_internal_mass==2:        
    
    c_internal_mass=840;        """Specific heat capacity construction [J/kgK] """
    th_internal_mass=0.1;       """Construction thickness [m] """
    rho_internal_mass=1000;     """Density construction in [kg/m3] """

#Heavy weight construction    
else:                          
         
    c_internal_mass=840;        """Specific heat capacity construction [J/kgK] """
    th_internal_mass=0.2;       """Construction thickness [m] """
    rho_internal_mass=2500;     """Density construction in [kg/m3]   """ 

"""Facade construction

    It is possible to choose between light, middle or heavy weight construction
"""

#Light weight construction   
if N_facade==1:               
    
    c_facade=840;             """Specific heat capacity construction [J/kgK] """
    rho_facade=500;           """Density construction in [kg/m3] """
    th_facade=0.1;            """Construction thickness [m] """ 
#Middle weight construction 
elif  N_facade==2:                 
    
    c_facade=840;             """Specific heat capacity construction [J/kgK] """
    rho_facade=1000;          """Density construction in [kg/m3] """
    th_facade=0.1;            """Construction thickness [m] """

#Heavy weight construction 
else:                        
    
    c_facade=840;             """Specific heat capacity construction [J/kgK] """
    rho_facade=2500;          """Density construction in [kg/m3] """
    th_facade=0.2;            """Construction thickness [m] """


Aglass=sum(glass);             """Sum of all glass surfaces [m2] """

"""Volume floor and internal walls construction [m3] 
    
    A_internal_mass:  Floor and internal walls surface [m2] 
"""
V_internal_mass=A_internal_mass*th_internal_mass  

""" n: ventilation air change per hour;  
    V_dwelling : internal volume m3 """

qV=(n*V_dwelling)/3600;            """Ventilation, volume air flow [m3/s] """ 
qm=qV*rho_air;                    """Ventilation, mass air flow [kg/s] """

""" 
Dwelling temperatures calculation
Calculation of the resistances
"""
Rair_wall=1/(A_internal_mass*alpha_internal_mass);  """Resistance indoor air-wall """
U=1/(1/alpha_i_facade+Rc_facade+1/alpha_e_facade);  """U-value indoor air-facade """
Rair_outdoor=1/(A_facade*U+Aglass*Uglass+qm*c_air); """Resitance indoor air-outdoor air """

"""Calculation of the capacities """
Cair = rho_internal_mass*c_internal_mass*V_internal_mass/2+ rho_air*c_air*V_dwelling; """Capacity indoor air + walls """
Cwall= rho_internal_mass*c_internal_mass*V_internal_mass/2;                           """Capacity walls """

"""Define Temperature SP

    Assume that in normal working day, people wake up at 7.00, 
    go to work at 8.00 return home at 18.00 and go to sleep at 23.00
    
"""


"""Define Temperature SP for 1 days (24 hours) """

Night_T_SP=17
Day_T_SP=20

"""Temperature different between day and night."""
delta_T= Day_T_SP - Night_T_SP

"""Define Wake up time """
Wu_time =7;           """Wake up at 7 in the morning """
duty_wu = 23-7       

"""Go to work time/ leave the house """
Work_time = 8;           """Go to work at 8 in the morning """
duty_w   = 23-8        

"""Back to home """
back_home = 18;         """Back home at 18.00 """
duty_b   = 23-18 


#-----------------------
t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
temp1 = signal.square(2 * np.pi* days * t,duty=duty_wu/24)
temp1 = np.clip(temp1, 0, 1)
# add delay to array
temp1=np.roll(temp1,Wu_time)

#----------------
t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
temp2 = signal.square(2 * np.pi* days * t,duty=duty_w/24)
temp2 = np.clip(temp2, 0, 1)
# add delay to array
temp2=np.roll(temp2,Work_time)

#___________
t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
temp3 = signal.square(2 * np.pi* days * t,duty=duty_b/24)
temp3 = np.clip(temp3, 0, 1)
# add delay to array
temp3=np.roll(temp3,back_home)

# Calculate SP

temp4=temp1-temp2+temp3
SP=(temp4*delta_T)+Night_T_SP

SP=SP[np.newaxis]
SP=SP.T

# Plot 48 hours
plt.plot(time_t[0:48], SP[0:48])
plt.ylabel('Temperature_SP (degC)')
plt.xlabel('time (sec)')
plt.legend(loc=2)
#print(Qinternal)
SP=np.delete(SP, -1, 0)

def weird_division(n, d):
    return n / d if d else 0
#Define Simulation time

days_Sim = 20                          #number of simulation days

time_sim      = time[0:days_Sim*24][:,0]
Qsolar_Sim    = Qsolar[0:days_Sim*24][:,0] 
Qinternal_Sim = Qinternal[0:days_Sim*24][:,0]
#Qinst_Sim     = Qinst_Sim[0:days_Sim*24][:,0]
T_outdoor_Sim = T_outdoor[0:days_Sim*24][:,0]

#Set point
SP_Sim=SP[0:days_Sim*24][:,0]
print(len(T_outdoor_Sim))
print(len(Qsolar_Sim))
print(len(Qinternal_Sim))
print(len(SP_Sim))
print(len(time_sim))
from scipy.integrate import odeint
import math
import pandas as pd 


kP = 10000
ki = 0
kd = 0

# Devide by zero function
def weird_division(n, d):
    return n / d if d else 0


#Define model
def House_Tem(x,t,T_outdoor_Sim,Qinternal_Sim,Qsolar_Sim,SP_T):
    
    # Inputs (8):
    
    # Isolar         : Solar Irradiation [W/m2]
    # Toutside       : Outside temperature [K]
    # FP             : Pump 1 Flow [m3/s]
    # Ftap           : Tap water flow  [m3/s]
    # TSTCin = T     : Tank tempearature out with delay [K]
    # THEin  = TSTC4 : Solar collector temperature with delay [K]
    # Tfloor         : Water temperature output from Floor  [K]
    #FP2             : Pump 2 Flow [m3/s]
    

# States :
    
    Tair            = x[0]   # Temperature Buffer Tank (K)
    Twall           = x[1]   # Return Temperature to Floor (K)
    integal         = x[2]
    #Qinstdt         = x[2]
    
    err      = SP_T-Tair
    integaldt= err
    integald= np.clip(integaldt, 0, 5)
    Qinst    = kP*(err) + ki*integal # kd*Tair.dt() # PID form
    Qinst=np.clip(Qinst, 0, 7000)
    #m.Equation(Integl.dt()==err )
    Tairdt   = ((T_outdoor_Sim-Tair)/Rair_outdoor + (Twall-Tair)/Rair_wall  + Qinst + Qinternal_Sim + CF*Qsolar_Sim)/Cair
    Twalldt  = ((Tair-Twall)/Rair_wall + (1-CF)*Qsolar_Sim)/Cwall
    
    # Return 
    return [Tairdt,Twalldt,integaldt]

# Initial Conditions for the States
    
#Solar Collector
Tair0    = 20   
Twall0   = 20
integal0=0

y0 = [Tair0,Twall0,integal0]

# Time Interval (sec)
#t = np.linspace(0,60*60,days_Sim*24)  # Define Simulation time with resolution
#time_sim      = time[0:days_Sim*24]
t = time_sim           # Define Simulation time with sampling time

# Model Paramters Inputs

Qsolar_Sim    =  Qsolar_Sim
Qinternal_Sim =  Qinternal_Sim
#Qinst_Sim     = m.Param(value=Qinst_Sim)
#SP_Sim        = SP_Sim


# Outside Temperature (K)
#T_outdoor_Sim = np.ones(len(t))* 15
T_outdoor_Sim =T_outdoor_Sim
#Toutside[50:] = 283


Tair  = np.ones(len(t))*Tair0
Twall  = np.ones(len(t))*Twall0
integal= np.ones(len(t))*integal0
Qinst = np.zeros(len(t))
SP_T=SP_Sim
#SP_T= np.ones(len(t))*20


for i in range(len(t)-1):
    
    inputs = (T_outdoor_Sim[i],Qinternal_Sim[i],Qsolar_Sim[i],SP_T[i])
    ts = [t[i],t[i+1]]
    y = odeint(House_Tem,y0,ts,args=inputs)
    # Store results
    
#delatat is a delay base on distance between buffer tank and Solar Collector    
    #deltat    = rp**(2)*pi*lp/(FP[i+1])
    
    Tair[i+1]         = y[-1][0]
    Twall[i+1]         = y[-1][1]
    integal[i+1]         = y[-1][2]

    # Adjust initial condition for next loop
    y0 = y[-1]
    
# Construct results and save data file
data = np.vstack((t,Tair,Twall,integal)) # vertical stack
#data = data.T             # transpose data
np.savetxt('data.txt',data,delimiter=',')
#print(Tair)

# Plot the inputs and results
plt.figure(figsize=(10,7))
#plt.plot(t,TSTC4,'k--',linewidth=3)
#plt.plot(t,THE4,'b--',linewidth=3)
plt.plot(t,Tair,'r--',linewidth=3)
plt.plot(t,SP_T) #Qsolar_Sim
#plt.plot(t,(Qsolar_Sim*0.001+18)) #Qsolar_Sim
#plt.plot(t,(Qinst*0.001+18)) #Qsolar_Sim

plt.plot(t,T_outdoor_Sim) #Qsolar_Sim

#plt.plot(t,THES4,'g--',linewidth=3)
plt.ylabel('T (K)')
plt.legend(['Water Temperature'],loc='best')
plt.xlabel('Time (sec)')
plt.figure()
plt.show()
Qinst    = kP*(SP_T-Tair) + ki*integal # kd*Tair.dt() # PID form
Qinst=np.clip(Qinst, 0, 7000)
# Plot the inputs and results
plt.figure(figsize=(10,7))
#plt.plot(t,TSTC4,'k--',linewidth=3)
#plt.plot(t,THE4,'b--',linewidth=3)
plt.plot(t,Qinst,'r--',linewidth=3)
time_sim= time[0:365*24][:,0]
plt.plot(time_sim,T_outdoor)
Tem_profile=np.ones(8760)
Tem_profile=Tem_profile*14
Tem_out=T_outdoor.flatten()
#--------------
temp_diff=Tem_profile - Tem_out
#-----------------
sumtepdiff=np.sum(temp_diff)
kwh_per_degree=97000/sumtepdiff
print(kwh_per_degree)
Q_profile= kwh_per_degree*temp_diff
Q_profile=np.clip(Q_profile, 0, 100)
plt.plot(Q_profile)
kwh_per_degree=5907.875/sumtepdiff
print(kwh_per_degree)
Q_profile= kwh_per_degree*temp_diff
Q_profile=np.clip(Q_profile, 0, 10)
plt.figure(figsize=(15, 6))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (hour)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(Q_profile)
Qtap = np.array([0.105, 1.4, 0.105, 0.105, 0.105, 0.105,
                 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 
                 0.315, 0.105, 0.105, 0.105, 0.105, 0.105, 
                 0.105, 0.105, 0.735, 0.105, 1.4])

f = np.array([3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
              3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 6])

Tm = np.array([25, 40, 25, 25, 25, 25, 25, 25, 25, 10, 25, 
               25, 10, 25, 25, 25, 25, 40, 40, 25, 10, 25, 40])

Tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 
               0, 55, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0])

time_event = np.array([7*60, (7*60+5), (7.5*60), (8*60+1), (8.25*60), (8.5*60),
                       (8.75*60), (9*60), (9.5*60), (10.5*60), (11.5*60), (11.75*60),
                       (12.75*60), (14.5*60), (15.5*60), (16.5*60), (18*60), (18.25*60),
                       (18.5*60), (19*60), (20*60), (21.25*60), (21.5*60)])
T_cold=np.ones(23)*10
Th_water = np.ones(len(Tm))

for i in range (0,len(Tm)):
        if Tp[i]>0:
            Th_water[i] = Tp[i]
        else:
            Th_water[i] = Tm[i]


m_tap_water = (Qtap*3600)/(4.19*(Th_water-T_cold))
Qtap_per_min = Qtap*60 # kw energy use per minutes
water_use_time = m_tap_water/f   # in minutes
water_use_time = np.round(water_use_time,decimals=0)
water_use_time = water_use_time.astype(int)
Ave_energy_use_perminute = Qtap_per_min/water_use_time #kw
temp2 = np.zeros(24*60)
k=0
for i in range (24*60):
    if k>=23:
        break
    elif i == time_event[k]:
        for f in range(water_use_time[k]):
            temp2[i+f]= Ave_energy_use_perminute[k]
        k=k+1
'''
1 days hot water profile in minutes
'''
hot_water_profile = temp2
plt.figure(figsize=(15, 6))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (minutes)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(hot_water_profile)
Qtap = np.array([0.105, 1.4, 0.105, 0.105, 0.105, 0.105,
                 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 
                 0.315, 0.105, 0.105, 0.105, 0.105, 0.105, 
                 0.105, 0.105, 0.735, 0.105, 1.4])

f = np.array([3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
              3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 6])

Tm = np.array([25, 40, 25, 25, 25, 25, 25, 25, 25, 10, 25, 
               25, 10, 25, 25, 25, 25, 40, 40, 25, 10, 25, 40])

Tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 
               0, 55, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0])

'''
time_event = np.array([7*60, (7*60+5), (7.5*60), (8*60+1), (8.25*60), (8.5*60),
                       (8.75*60), (9*60), (9.5*60), (10.5*60), (11.5*60), (11.75*60),
                       (12.75*60), (14.5*60), (15.5*60), (16.5*60), (18*60), (18.25*60),
                       (18.5*60), (19*60), (20*60), (21.25*60), (21.5*60)])
'''
# Redefine time event for hourly rate

time_event = np.array([7, (7+5/60), (7.5), (8+1/60), (8.25), (8.5),
                       (8.75), (9), (9.5), (10.5), (11.5), (11.75),
                       (12.75), (14.5), (15.5), (16.5), (18), (18.25),
                       (18.5), (19), (20), (21.25), (21.5)])

time_event_new = np.floor(time_event)
Qtap_per_hour = Qtap*24 # kw energy use per minutes
print(Qtap_per_hour)
#temp = np.zeros(24)
temp1 = np.zeros(24)

n=0
i=0
for k in range (24):
    temp1[k] = 0
    #print(temp1)

    while n == time_event_new[i]:  
        temp= Qtap_per_hour[i]
                #print(temp)
        temp1[k] =temp1[k]+temp
        if i >21:
            break
        else:
            i=i+1
    n=n+1
hot_water_profile = temp1
plt.figure(figsize=(17, 5))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (hour)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(hot_water_profile)
Q_building_profile = (hot_water_profile.tolist()*365) + Q_profile*24
plt.figure(figsize=(15, 6))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (hour)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(Q_building_profile)
#qtaph1= 0.270 + 0.107*math.sqrt(1) + 0.00325*1
#qtap24= 0.270 + 0.107*math.sqrt(24) + 0.00325*24
qtaph1= 243 + 679*math.sqrt(1) + 12.8*1
qtap24= 243 + 679*math.sqrt(24) + 12.8*1
ratio = qtap24/qtaph1
#print(ratio)
fnew=f*ratio
#print(fnew)
T_cold=np.ones(23)*10
Th_water = np.ones(len(Tm))
for i in range (0,len(Tm)):
        if Tp[i]>0:
            Th_water[i] = Tp[i]
        else:
            Th_water[i] = Tm[i]


m_tap_water = (Qtap*3600)/(4.19*(Th_water-T_cold))
water_use_time = m_tap_water*24/fnew   # in minutes
#print(water_use_time)
#temp = np.zeros(24)
temp1 = np.zeros(24)
n=0
i=0
for k in range (24):
    temp1[k] = 0
    #print(temp1)
    #print(n)
    while n == time_event_new[i]:  
        temp= water_use_time[i]
        temp1[k] =temp1[k]+temp
        if i >21:
            break
        else:
            i=i+1
    n=n+1
water_use_timenew = temp1/60  #in hours
print(water_use_timenew)
'Shift the extra time to the next hour'

temp2=water_use_timenew
for i in range (24):
    if temp2[i] >1:
        temp2[i+1]= temp2[i+1] + (temp2[i]-1)
        temp2[i]=1        
        #print(temp2)
    else: 
        temp2[i]=temp2[i]

print(temp2)
hot_water_profile_cor = hot_water_profile*temp2
print(hot_water_profile_cor)
plt.figure(figsize=(17, 5))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (hour)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(hot_water_profile,'r--o',label="hot_water_profile")
plt.plot(hot_water_profile_cor,label="hot_water_profile_cor")
plt.legend()
plt.show()
Q_building_profile = (hot_water_profile_cor.tolist()*365) + Q_profile*24
plt.figure(figsize=(15, 6))
plt.ylabel('Energy demand (kw)', fontsize=20)
plt.xlabel('Time (hour)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(Q_building_profile)
#!pip list

