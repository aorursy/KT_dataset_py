import arff
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(2)
data = [(x,y,int(c)) for x,y,c in arff.load('../input/kiwhs-comp-1-complete/train.arff')]
print('datapoints:', len(data))

validation = random.sample(data, 100)
train = set(data) - set(validation)

print('train set:', len(train))
print('validation set:', len(validation))
def plot_datapoints(data):
    x1,y1 = zip(*[(x,y) for x,y,c in data if c == 1])
    x2,y2 = zip(*[(x,y) for x,y,c in data if c == -1])
    plt.plot(x1, y1, 'o')
    plt.plot(x2, y2, 'o')

def plot_line(line, color='black'):
    m,b = line
    points_x = range(-3,4)
    points_y = [m*x+b for x in points_x]
    plt.plot(points_x, points_y, '-', lw=2, color=color)
    
plot_datapoints(data)
plot_datapoints(validation)
plt.title('validation set and divider')
def get_successrate(cb):
    correct = 0
    for v in validation:
        if cb(v[0],v[1]) == v[2]: correct += 1
    return correct / len(validation)

def classify_bisect(x, y):
    plt.plot([0,0], [-3,3], 'k-', lw=2)
    return 1 if x > 0 else -1

print('successrate', get_successrate(classify_bisect))
plot_datapoints(validation)
plt.title('validation set and best-fit of every generation')

def classify_linear(x, y, line):
    m,b = line
    return -1 if m*x+b > y else 1

def fitness(line):
    correct = 0
    for v in validation: 
        if classify_linear(v[0],v[1], line) == v[2]: correct += 1
    return correct / len(validation)

def procreate(p1, p2):
    child = random.choice([(p1[0],p2[1]),(p2[0],p1[1])])
    return child

def mutate(line):
    m,b = line
    m = m * random.uniform(-1, 1)
    b = b * random.uniform(-2, 2)
    return m,b

population = [(p[0], p[1]) for p in np.random.uniform(-3, 3, (20,2))];
for i in range(1000):
    population = sorted(population, key=fitness, reverse=True)
    plot_line(population[0], color='lightgray')
    children = [procreate(population[i], population[i+1]) for i in range(0,10,2)]
    children = [mutate(child) for child in children]
    population = population[:-5] + children

plot_line(population[0])
print('selected function: ' + str(population[0][0]) + '*x+' + str(population[0][1]))
print('successrate', fitness(population[0]))
import pandas as pd

test = pd.read_csv('../input/ki-whs-a3/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
test['Category (String)'] = test.apply(lambda row: classify_linear(row.X, row.Y, population[0]), axis=1)
test.drop(['X', 'Y'], axis=1, inplace=True)
test.to_csv('submission.csv')