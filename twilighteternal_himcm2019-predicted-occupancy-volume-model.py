import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import math

import random
font = {'family': 'Times New Roman',

        'weight': 'normal',

        'size': 12,

        }

font2 = {'family': 'Times New Roman',

        'weight': 'normal',

        'size': 7,

         }
class gsep():#graph-structure evolve predictor

    def __init__(self,max_q,m=4,device_type='car'):#requirements:n>=2, device_type should be among the 'car' or the 'phone' 

        self.m=m

        self.max_q=max_q

        self.n=math.floor(1+math.log((((self.max_q+2)*self.m - 2*self.max_q)/(self.m**2)),self.m-1))#quantity of levels

        if self.n<2:

            print("Invalid Quantity!")

        self.species = [[] for x in range(self.n)]

        self.max_age = 80

        self.ages = [[] for x in range(self.n)]#ages of the people

        

        #the life cycle for the device chosen in the model

        if device_type == 'car':

            self.max_device_life = 10

            self.cost = 10000

            self.age_lower_bound = 21

        else: 

            self.max_device_life = 2

            self.cost = 2000

            self.age_lower_bound = 12

        self.device_ages = [[] for x in range(self.n)]

        self.labels = [[] for x in range(self.n)]

        self.wages = [[] for x in range(self.n)]

        return

    

    def get_n(self):

        return self.n

    def get_q(self):

        return self.q

    

    def init_graph(self, lowest_wage, relative_increasing_percentages, μ, init_ar=0.80, city_center_ratio=0.4):

        #parameters saving into the class

        self.q = self.m+((((self.m)**2)*(1-((self.m-1)**(self.n-1))))/(2-self.m))#the quantity of population we use in the model

        self.init_ar = init_ar

        self.op = self.q*self.init_ar#owning population

        self.μ = μ

        present_op = 2

        self.relative_increasing_percentages = relative_increasing_percentages

        self.basic_wages = [lowest_wage for x in range(self.n)]

        self.city_center_ratio = city_center_ratio

        for i in range(1,self.n):

            j = self.n-i-1

            #print(self.n,",",j)

            #print(self.basic_wages)

            self.basic_wages[j] = self.basic_wages[j+1]*self.relative_increasing_percentages[j+1] + self.basic_wages[j+1]

        

        self.init_wage_largest = [0 for x in range(self.n)]#now - because of random, in the future, because of the incentive of buying phone that brings to the work

        self.init_lwage_elements = [(0,0,0) for x in range(self.n)]#the largest wages' elements

        

        #initialize 0

        self.species[0] = [[0 for x in range(self.m)]]

        self.ages[0] = [[70 for x in range(self.m)]]

        self.labels[0] = [[False for x in range(self.m)]]

        self.wages[0] = [[int(self.basic_wages[0]) for x in range(self.m)]]

        self.device_ages[0] = [[int((2/3)*self.max_device_life) for x in range(self.m)]]

        #first centers of every specie

        self.species[0][0][0] = 1

        self.ages[0][0][0] = 60

        self.wages[0][0][0] = self.wages[0][0][0]*(1+self.city_center_ratio)

        self.device_ages[0][0][0] = 0

        self.init_wage_largest[0] = self.wages[0][0][0]

        self.init_lwage_elements[0] = [0,0,0]

        

        #initialize 1

        self.species[1] = [[0 for y in range(self.m)] for x in range(self.m)]

        self.ages[1] = [[70 for y in range(self.m)] for x in range(self.m)]

        self.labels[1] = [[False for y in range(self.m)] for x in range(self.m)]

        self.device_ages[1] = [[int((2/3)*self.max_device_life) for y in range(self.m)] for x in range(self.m)]

        self.wages[1] = [[int(self.basic_wages[1]) for y in range(self.m)] for x in range(self.m)]

        #first centers of every specie

        self.species[1][0][1] = 1

        self.ages[1][0][1] = 65

        self.device_ages[1][0][1] = int((1/3)*self.max_device_life)

        self.wages[1][0][1] = self.wages[1][0][1]*(1+self.city_center_ratio/2)

        self.init_wage_largest[1] = self.wages[1][0][1]

        self.init_lwage_elements[1] = [1,0,1]

        #other centers of every specie

        for j in range(1, len(self.species[1])):

            self.species[1][j][0] = 1

            self.ages[1][j][0] = 65

            self.device_ages[1][j][0] = int((1/3)*self.max_device_life)

            self.wages[1][j][0] = self.wages[1][j][0]*(1+self.city_center_ratio/2)

            tpw = self.init_wage_largest[1]

            self.init_wage_largest[1] = max(self.wages[1][j][0],tpw)

            if self.init_wage_largest[1] > tpw:

                self.init_lwage_elements[1] = [1,j,0]

            present_op += 1

            

        #process the rest species

        for i in range(2,len(self.species)):

            #as the first time !father

            #first-always: 0 1 0 ...

            #others-always: 1 0 0 ...

            #initialize others

            self.species[i] = [[0 for y in range(self.m)] for x in range(len(self.species[i-1])*(self.m-1))]

            self.ages[i] = [[max(int(70-0.3*i-0.002*x-0.005*y),5) for y in range(self.m)] for x in range(len(self.ages[i-1])*(self.m-1))]

            self.labels[i] = [[False for y in range(self.m)] for x in range(len(self.species[i-1])*(self.m-1))]

            self.device_ages[i] = [[0 for y in range(self.m)] for x in range(len(self.device_ages[i-1])*(self.m-1))]

            self.wages[i] = [[int(self.basic_wages[i]) for y in range(self.m)] for x in range(len(self.wages[i-1])*(self.m-1))]

            #first centers of every specie

            self.species[i][0][1] = 1

            self.ages[i][0][1] -= 5

            self.device_ages[i][0][1] = int((1/3)*self.max_device_life)

            self.wages[i][0][1] = self.wages[i][0][1]*(1+self.city_center_ratio/2)

            self.init_wage_largest[i] = self.wages[i][0][1]

            self.init_lwage_elements[i] = [i,0,1]

            present_op += 1

            for j in range(1,len(self.species[i])):

                #other centers of every specie

                self.species[i][j][0] = 1

                self.ages[i][j][0] -= 5

                self.device_ages[i][j][0] = int((1/3)*self.max_device_life)

                self.wages[i][j][0] = self.wages[i][j][0]*(1+self.city_center_ratio/2)

                tpw = self.init_wage_largest[i]

                self.init_wage_largest[i] = max(self.wages[i][j][0],tpw)

                if self.init_wage_largest[i] > tpw:

                    self.init_lwage_elements[i] = [i,j,0]

                present_op += 1

        

        #op validation check

        if present_op > self.op:

            print("invalid ar! not reality!")

        

        #append the rest of the near-centers, however only specie's device value is nearly the center. 

        #When finish filling, the algorithm automatically stops

        for i in range(len(self.species[0])):

            if present_op>=self.op:

                return

            if self.species[0][i] == 0:

                self.species[0][i] = 1

                present_op+=1

        for i in range(1,len(self.species)):

            for j in range(len(self.species[i])):

                for k in range(len(self.species[i][j])):

                    if present_op>=self.op:

                        return

                    if self.species[i][j][k] == 0:

                        self.species[i][j][k] = 1

                        present_op += 1

        print(self.init_wage_largest)

        print(self.init_lwage_elements)

        return



    def evolve(self, cw, phi, sc, delta, years=40, fix_rip=0.1, substitides = 0.3, old_to_new_rate = 0.5, wage_increase_rate=0.07):#sc = sensitive coefficience for wage because of happiness

        #first step: replace the death

        otnr = old_to_new_rate

        all_array = []

        for year in range(years):

            #whether to breed the next generation

            breed = False

            if (year+1)%10==0:

                breed = True

            total_aid = 0

            for i in range(len(self.species)):

                for j in range(len(self.species[i])):

                    for k in range(len(self.species[i][j])):

                        self.ages[i][j][k]+=1

                        self.device_ages[i][j][k]+=1

                        self.wages[i][j][k]*=(1+wage_increase_rate)

                        if self.ages[i][j][k]>=self.max_age or self.labels[i][j][k]==True:

                            total_aid += otnr*self.species[i][j][k]

                            self.labels[i][j][k] = False #clear the labels

                            if i < len(self.species)-2:

                                #the following process is practical because this is actually a kind of inner-breed, but not actually transfering.

                                #Therefore, we can always use the same element in the level

                                #select the best from the next-best level and transfer

                                self.species[i][j][k] = self.species[self.init_lwage_elements[i+1][0]][self.init_lwage_elements[i+1][1]][self.init_lwage_elements[i+1][2]]

                                self.ages[i][j][k] = 1+self.ages[self.init_lwage_elements[i+1][0]][self.init_lwage_elements[i+1][1]][self.init_lwage_elements[i+1][2]]

                                self.device_ages[i][j][k] = 1+self.device_ages[self.init_lwage_elements[i+1][0]][self.init_lwage_elements[i+1][1]][self.init_lwage_elements[i+1][2]]                             

                                self.wages[i][j][k] = (1+substitides)*self.wages[self.init_lwage_elements[i+1][0]][self.init_lwage_elements[i+1][1]][self.init_lwage_elements[i+1][2]]                             

                                #the following will not be immediately transfer, as this is high in cost. Actually, we will process them in the following calculation by labelling them

                                self.labels[self.init_lwage_elements[i+1][0]][self.init_lwage_elements[i+1][1]][self.init_lwage_elements[i+1][2]] = True

                            else:

                                #generate the new-sample

                                self.species[i][j][k] = 0

                                self.ages[i][j][k] = 0

                                self.device_ages[i][j][k] = 0

                                self.wages[i][j][k] = (1-fix_rip)*(1+substitides)*self.wages[i][j][0]#aid by parents



                        #update the lwage and the largest wage

                        if i>=len(self.init_wage_largest):

                            self.init_wage_largest.append(self.species[i][j][k])

                            self.init_lwage_elements.append([i,j,k])

                        else:

                            tpw = self.init_wage_largest[i]

                            self.init_wage_largest[i] = max(self.species[i][j][k],tpw)

                            if self.init_wage_largest[i]>tpw:

                                self.init_lwage_elements[i] = [i,j,k]

                        

                        #update the aar, by sampling, assume only the same level will cause influence

                        aar_ideal = self.μ * (np.mean(np.array(random.sample(self.species[i][j],math.ceil((1/2)*len(self.species[i][j]))))))

                        mcss = int(math.floor((1/2)*len(self.species[min(i-1,0)][0])))

                        if len(self.species[min(i-1,0)])>=mcss:

                            inner_sample = random.sample(self.species[min(i-1,0)],mcss)

                            aar_ideal += (self.μ**2) * np.mean(np.array(random.sample(inner_sample,1)))

                            aar_ideal = (1/2)*aar_ideal

                        aar_reality = cw * (self.wages[i][j][k]-self.cost)

                        aar = 0.6*aar_ideal + 0.4*aar_reality

                        delta_specie = 0

                        if self.ages[i][j][k]>=self.age_lower_bound and aar>delta:

                            delta_specie = int(math.log(phi*aar,delta))

                            self.wages[i][j][k]+=sc*delta_specie #in the instance, only once is added, we ignored the future impact

                        if self.device_ages[i][j][k]>self.max_device_life:

                            self.device_ages[i][j][k] = 0

                            total_aid += math.ceil(2/10*self.species[i][j][k])

                            self.species[i][j][k] *= int((7/10))

                        self.species[i][j][k]+=delta_specie #update the number

                        if self.species[i][j][k]==0:

                            if total_aid > 0:

                                total_aid -= 1

                                self.species[i][j][k] += 1

            if breed == True:

                self.species.append([[0 for y in range(self.m)] for x in range(len(self.species[i])*(self.m-1))])

                self.ages.append([[max(int(70-0.3*i-0.002*x-0.005*y),5) for y in range(self.m)] for x in range(len(self.ages[i])*(self.m-1))])

                self.labels.append([[False for y in range(self.m)] for x in range(len(self.species[i])*(self.m-1))])

                self.device_ages.append([[0 for y in range(self.m)] for x in range(len(self.device_ages[i])*(self.m-1))])

                self.wages.append([[int(self.wages[i][0][0])*(1-fix_rip) for y in range(self.m)] for x in range(len(self.wages[i])*(self.m-1))])

            sum_of_year = np.sum(np.sum(np.array(self.species)))

            q = self.q

            all_array.append(sum_of_year/q)

        return all_array
a = gsep(6245.8+9272.37,m=4,device_type='phone')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(2091,ii,0.3,0.85)

all_array1 = a.evolve(cw=0.005216945355,phi=1,sc=240,delta=500,years=30)

print("First Tier:")

print("\t",all_array1)
a = gsep(8506.16,m=4,device_type='phone')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(1810,ii,0.3,0.8/51920274.09999999*28967454.7)

all_array2 = a.evolve(cw=0.002915090505,phi=1,sc=240,delta=500,years=30)

print("Second Tier:")

print("\t",all_array2)
a = gsep(16803.02,m=4,device_type='phone')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(1591,ii,0.3,0.8/51920274.09999999*19112271.20033326)

all_array3 = a.evolve(cw=0.001863490534528176,phi=1,sc=240,delta=500,years=30)

print("Third Tier:")

print("\t",all_array3)
fig = plt.figure()

ax = fig.add_subplot(111)

t = [x for x in range(30)]

ax.plot(t, all_array1,color='darkorange')

ax.plot(t, all_array2,color='mediumseagreen')

ax.plot(t, all_array3,color='#2716b3')

ax.legend(prop=font2,loc='best')

ax.set_title('The predicted occupancy volume of phone for different tiers of cities',font)

ax.set_xlabel('Δtime (year)',font)

ax.set_ylabel('pov',font)



ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.tick_params(labelsize=10)

label1 = ax.get_xticklabels() + ax.get_yticklabels()

plt.show()
a = gsep(6245.8+9272.37,m=4,device_type='car')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(2091,ii,0.3,0.4)

all_array4 = a.evolve(cw=0.005216945355,phi=10,sc=240,delta=500,years=30)

print("First Tier car:")

print("\t",all_array4)
a = gsep(8506.16,m=5,device_type='car')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(1604,ii,0.3,0.4/51920274.09999999*28967454.7)

all_array5 = a.evolve(cw=0.002915090505,phi=10,sc=240,delta=500,years=30)

print("Second Tier car:")

print("\t",all_array5)
a = gsep(16803.02,m=6,device_type='car')

n = a.get_n()

ii = [i**2 for i in range(1,n+1)]

a.init_graph(1591,ii,0.3,0.5/51920274.09999999*19112271.20033326)

all_array6 = a.evolve(cw=0.001863490534528176,phi=10,sc=240,delta=500,years=30)

print("Third Tier car:")

print("\t",all_array6)
fig = plt.figure()

ax = fig.add_subplot(111)

t = [x for x in range(30)]

ax.plot(t, all_array4, color='darkorange')

ax.plot(t, all_array5, color='mediumseagreen')

ax.plot(t, all_array6, color='#2716b3')



ax.legend(prop=font2,loc='best')

ax.set_title('The predicted occupancy volume of car for different tiers of cities',font)

ax.set_xlabel('Δtime (year)',font)

ax.set_ylabel('pov',font)



ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.tick_params(labelsize=10)

label1 = ax.get_xticklabels() + ax.get_yticklabels()

plt.show()