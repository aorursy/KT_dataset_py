import numpy as np
from matplotlib import pyplot as plt
x = [2,3,4,5,6,7]
y = [7,3,6,4,2,1]
plt.plot(x,y,color='R')
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
x = [2,3,4,5,6,7]
y = [2,3,4,5,6,7]

x2 = [2,3,4,5,6,7]
y2 = [4,5,8,9,12,14]

plt.subplot(1,2,1)
plt.plot(x,y,color='R')
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

plt.subplot(1,1,1)
plt.plot(x2,y2,color='R')
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
fruit = {'apple':30,'mango':45,'orange':90}
x = list(fruit.keys())
y = list(fruit.values())
plt.barh(x,y,color='R')
plt.title("Bar Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
x = [2,3,4,5,6,7]
a = [7,3,6,4,2,1]
b = [1,2,3,9,7,6]

plt.figure(figsize=(5,5))
plt.scatter(x,a,color='r',s=100,marker='2')
plt.scatter(x,b,color='g',s=100)
plt.legend(['a','b'])
plt.show()
a=[1,2,3,4,5,6]
b=[1,2,3]
plt.hist(a,b)
plt.show()
a = [1,2,3,4,5,6,7,8,9]
b = [1,2,3,4,5,4,3,2,1]
c = [9,8,7,5,4,3,1,2,4]
data = list([a,b,c])
plt.boxplot(data)
plt.show()
a = [1,2,3,4,5,6,7,8,9]
b = [1,2,3,4,5,4,3,2,1]
c = [9,8,7,5,4,3,1,2,4]
data = list([a,b,c])
plt.violinplot(data)
plt.grid(True)
plt.show()
fruit = {'apple':30,'mango':45,'orange':90}
x = list(fruit.keys())
y = list(fruit.values())
plt.pie(y,labels=x,shadow = True,autopct = '%0.1f%%',explode =(1,1,1))
plt.show()
a = [1,2,3,4,5,6,7,8,9]
b = [1,2,3,4,5,4,3,2,1]
plt.stackplot(a,b)
plt.show()