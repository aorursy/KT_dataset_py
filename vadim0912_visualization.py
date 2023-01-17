import numpy as np

import csv

from sklearn.ensemble import RandomForestClassifier as RF



from matplotlib import pyplot as plt

%matplotlib inline
i = 3168

x = np.zeros((i,21))

t = 0

with open('../input/voice.csv', newline='') as csvfile:

    file_ = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for row in file_:

        if t == 0:

            labels = row[0].split(',')

            t += 1

            continue

        else:

            if row[0].split(',')[-1] == '"male"':

                x[t-1,-1] = 1

            elif row[0].split(',')[-1] == '"female"':

                x[t-1,-1] = 0

            x[t-1,:-1] = row[0].split(',')[:-1]

            t += 1
idx = np.array([3,5,12])
male = x[x[:,-1] == 1]

female = x[x[:,-1] == 0]
plt.figure(figsize=(10,10))

plt.xlabel(labels[3][1:-1])

plt.ylabel(labels[5][1:-1])

plt.scatter(male[:,3],male[:,5], c = 'b', s = 20,label = "male")

plt.scatter(female[:,3],female[:,5], c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(labels[3][1:-1])

plt.ylabel(labels[12][1:-1])

plt.scatter(male[:,3],male[:,12], c = 'b', s = 20,label = "male")

plt.scatter(female[:,3],female[:,12], c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(labels[5][1:-1])

plt.ylabel(labels[12][1:-1])

plt.scatter(male[:,5],male[:,12], c = 'b', s = 20,label = "male")

plt.scatter(female[:,5],female[:,12], c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(labels[3][1:-1])

plt.ylabel(labels[5][1:-1])

plt.scatter(np.log(male[:,3]),np.log(male[:,5]), c = 'b', s = 20,label = "male")

plt.scatter(np.log(female[:,3]),np.log(female[:,5]), c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(labels[3][1:-1])

plt.ylabel(labels[12][1:-1])

plt.scatter(np.log(male[:,3]),np.log(male[:,12]), c = 'b', s = 20,label = "male")

plt.scatter(np.log(female[:,3]),np.log(female[:,12]), c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(labels[5][1:-1])

plt.ylabel(labels[12][1:-1])

plt.scatter(np.log(male[:,5]),np.log(male[:,12]), c = 'b', s = 20,label = "male")

plt.scatter(np.log(female[:,5]),np.log(female[:,12]), c = 'r',s = 20,label = "female")

plt.legend()

plt.show()
for i in range(idx.size):

    min1 = np.min(x[:,idx[i]])

    max1 = np.max(x[:,idx[i]])

    male[:,idx[i]] = (x[:1584,idx[i]] - min1)/(max1 - min1)

    female[:,idx[i]] = (x[1584:,idx[i]] - min1)/(max1 - min1)
m1 = np.sum(male,axis = 1)

f1 = np.sum(female,axis = 1)
Ox = np.ones(100)*0.474

Oy = np.linspace(1,8,100)

plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.xlabel(labels[12][1:-1])

plt.ylabel("log(new_feature1)")

plt.grid()

plt.scatter(male[:,12],np.log(m1), c = 'b', s = 10,label = "male")

plt.scatter(female[:,12],np.log(f1), c = 'r',s = 10,label = "female")

plt.legend()
accuracy = (np.sum(male[:,12] < 0.474) + np.sum((female[:,12] > 0.474)))/x.shape[0] 

accuracy
index_male_fail = (male[:,12] > 0.474) 

index_female_fail = (female[:,12] < 0.474)
Ox = np.ones(100)*0.474

Oy = np.linspace(1,8,100)

plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.xlabel(labels[12][1:-1])

plt.ylabel("log(new_feature1)")

plt.grid()

plt.scatter(male[index_male_fail,12],np.log(m1)[index_male_fail], c = 'b', s = 10,label = "male")

plt.scatter(female[index_female_fail,12],np.log(f1)[index_female_fail], c = 'r',s = 10,label = "female")

plt.legend()
m11 = np.sum(male[:,idx],axis = 1)

f11 = np.sum(female[:,idx],axis = 1)
Ox = np.linspace(0.2,0.7,100)

Oy = 5*Ox - 1

plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.xlabel(labels[12][1:-1])

plt.ylabel("log(new_feature2)")

plt.grid()

plt.scatter(male[:,12],m11, c = 'b', s = 10,label = "male")

plt.scatter(female[:,12],f11, c = 'r',s = 10,label = "female")

plt.legend()
accuracy1 = (np.sum(m11 - male[:,12]*5 + 1 > 0) + np.sum(f11 - female[:,12]*5 + 1 < 0))/x.shape[0]

accuracy1
index_male_fail = (m11 - male[:,12]*5 + 1 < 0)

index_female_fail = f11 - female[:,12]*5 + 1 > 0
Ox = np.linspace(0.2,0.7,100)

Oy = 5*Ox - 1

plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.xlabel(labels[12][1:-1])

plt.ylabel("log(new_feature2)")

plt.grid()

plt.scatter(male[index_male_fail,12],m11[index_male_fail], c = 'b', s = 10,label = "male")

plt.scatter(female[index_female_fail,12],f11[index_female_fail], c = 'r',s = 10,label = "female")

plt.legend()
Ox = np.linspace(-0.5,1.5,1000)

a = -1.5

b = 0.55

Oy =  a* Ox + b
plt.figure(figsize=(10,10))

plt.xlabel(labels[12][1:-1])

plt.ylabel("log(new_feature2)")

plt.grid()

plt.scatter(male[:,12],np.log(m11), c = 'b', s = 10,label = "male")

plt.scatter(female[:,12],np.log(f11), c = 'r',s = 10,label = "female")

plt.legend()
plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.xlabel(labels[12][1:-1] + " - log(new_feature2)")

plt.ylabel("log(new_feature2)")

plt.grid()

plt.scatter(male[:,12]-np.log(m11),np.log(m11), c = 'b', s = 10,label = "male")

plt.scatter(female[:,12]-np.log(f11),np.log(f11), c = 'r',s = 10,label = "female")

plt.legend()
accuracy2 = (np.sum((male[:,12]-np.log(m11))*(a) + b - np.log(m11) > 0) \

             + np.sum((female[:,12]-np.log(f11))*(a) + b - np.log(f11) < 0))/x.shape[0]

accuracy2
index_male_fail = (male[:,12]-np.log(m11))*(a) + b - np.log(m11) < 0

index_female_fail = ((female[:,12]-np.log(f11))*(a) + b - np.log(f11) > 0)
plt.figure(figsize=(10,10))

plt.plot(Ox,Oy)

plt.grid()

plt.scatter((male[:,12]-np.log(m11))[index_male_fail],np.log(m11)[index_male_fail], c = 'b', s = 10,label = "male")

plt.scatter((female[:,12]-np.log(f11))[index_female_fail],np.log(f11)[index_female_fail], c = 'r',s = 10,label = "female")

plt.legend()
accuracy3 = (np.sum((male[:,12]-np.log(m11))*(-1.5) + 0.55 - np.log(m11) > 0) \

             + np.sum((female[:,12]-np.log(f11))*-1.5 + 0.55 - np.log(f11) < 0))/x.shape[0]

accuracy3