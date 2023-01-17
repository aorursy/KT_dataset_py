import numpy as np

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

import random

import matplotlib.ticker as ticker
font = {'family': 'Times New Roman',

        'weight': 'normal',

        'size': 12,

        }

font2 = {'family': 'Times New Roman',

        'weight': 'normal',

        'size': 7,

         }
#city : wuhan

wuhan_targets = np.array([192.875,192.485,193.72,207.6488889,223.98])

wuhan_targets_list = [154.1833333,153.4433333,161.8533333,np.nan,np.nan,192.875,192.485,193.72,207.6488889,223.98]

wuhan_datas = np.array([np.array([x]) for x in range(len(wuhan_targets))])



regr = SVR(kernel='linear')

regr.fit(wuhan_datas.reshape(-1,1), wuhan_targets.reshape(-1,1))

wuhan_targets_list[4] = regr.predict(np.array(-1).reshape(-1,1))

wuhan_targets_list = np.array(wuhan_targets_list)



imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0)

imp.fit(wuhan_targets_list.reshape(-1,1))

wuhan_targets_list = imp.transform(wuhan_targets_list.reshape(-1,1))

print(wuhan_targets_list)

print("")



#city : hangzhou

hangzhou_targets = np.array([217.9033333,233.5833333,262.59,255.0616667,272.4383333,293.625,302.010275])

hangzhou_targets_list = [217.9033333,233.5833333,262.59,255.0616667,272.4383333,293.625,302.010275,np.nan,np.nan,325.2066667]

hangzhou_datas = np.array([np.array([x]) for x in range(len(hangzhou_targets))])



regr = SVR(kernel='linear')

regr.fit(hangzhou_datas.reshape(-1,1), hangzhou_targets.reshape(-1,1))

hangzhou_targets_list[7] = regr.predict(np.array(7).reshape(-1,1))



hangzhou_targets_list = np.array(hangzhou_targets_list)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0)

imp.fit(hangzhou_targets_list.reshape(-1,1))

hangzhou_targets_list = imp.transform(hangzhou_targets_list.reshape(-1,1))



regr = SVR(kernel='linear')

hangzhou_datas = np.array([np.array([x]) for x in range(len(hangzhou_targets_list))])

regr.fit(hangzhou_datas.reshape(-1,1), hangzhou_targets_list.reshape(-1,1))

hangzhou_targets_list[8] = regr.predict(np.array(8).reshape(-1,1))

print(hangzhou_targets_list)

print("")



#city : nannin

nannin_targets = np.array([114.25,119.38855,124.8666667,126.1333333,133.32])

nannin_targets_list = [83.7,np.nan,np.nan,np.nan,np.nan,114.25,119.38855,124.8666667,126.1333333,133.32]

nannin_datas = np.array([np.array([x]) for x in range(len(nannin_targets))])

regr = SVR(kernel='linear')

regr.fit(nannin_datas.reshape(-1,1), nannin_targets.reshape(-1,1))

nannin_targets_list[4] = regr.predict(np.array(-1).reshape(-1,1))[0]



nannin_targets = np.array(nannin_targets_list[4:])

regr = SVR(kernel='linear')

nannin_datas = np.array([np.array([x]) for x in range(len(nannin_targets))])

regr.fit(nannin_datas.reshape(-1,1), nannin_targets.reshape(-1,1))

nannin_targets_list[3] = regr.predict(np.array(-1).reshape(-1,1))[0]

nannin_targets_list[2] = regr.predict(np.array(-1).reshape(-1,1))[0]

nannin_targets_list[1] = regr.predict(np.array(-1).reshape(-1,1))[0]



nannin_targets = np.array(nannin_targets_list)

regr = SVR(kernel='linear')

nannin_datas = np.array([np.array([x]) for x in range(len(nannin_targets))])

regr.fit(nannin_datas.reshape(-1,1), nannin_targets.reshape(-1,1))

nannin_targets_list[1] = regr.predict(np.array(1).reshape(-1,1))[0]

nannin_targets_list[2] = regr.predict(np.array(2).reshape(-1,1))[0]

nannin_targets_list[3] = regr.predict(np.array(3).reshape(-1,1))[0]

nannin_targets_list[4] = regr.predict(np.array(4).reshape(-1,1))[0]

print(nannin_targets_list)

print("")



#jinan

jinan_targets = np.array([74.16666667, 116.875,124.2316, 140.36])

jinan_targets_list = [74.16666667,np.nan,np.nan,np.nan,np.nan,116.875,124.2316,np.nan,np.nan,140.36]



jinan_datas = np.array([np.array([0]),np.array([5]),np.array([6]),np.array([9])])



regr = SVR(kernel='linear')

regr.fit(jinan_datas.reshape(-1,1), jinan_targets.reshape(-1,1))

jinan_targets_list[7] = regr.predict(np.array(7).reshape(-1,1))[0]

jinan_targets_list[8] = regr.predict(np.array(8).reshape(-1,1))[0]

jinan_targets_list[4] = regr.predict(np.array(4).reshape(-1,1))[0]

jinan_targets_list[3] = regr.predict(np.array(3).reshape(-1,1))[0]

jinan_targets_list[2] = regr.predict(np.array(2).reshape(-1,1))[0]

jinan_targets_list[1] = regr.predict(np.array(1).reshape(-1,1))[0]

print(jinan_targets_list)

print("")
wti_datas=np.array([

    np.array(x) for x in range(12)

])

wti_targets=[

    97092*np.array([92.3,92.3,101.9,100.2,98.6,99.7,105.5,102.7,100.8,99.1,105.7,105.0]),

    97092*np.array([105.9,106.0,112.3,106.8,110.4,113.3,118.5,117.9,114.4,113.6,121.7,118.4]),

    97092*np.array([97.3,94.4,99.1,96.6,102.8,99.8,107.8,104.1,104.2,99.1,106.8,101.2])

]
airport_targets=np.array([

    1e4*np.array([395.3429167,401.29,423.1210417,428.97625,456.6175,464.5625,487.7926875,444.4372917,528.2870139,515.3816667]),

    1e4*np.array([112.2366667,125.5316417,129.5511889,132.9101806,139.3486167,144.875,149.5837083,156.0058333,161.3981019,166.9733333]),

    1e4*np.array([173.2633333,108.4266667,128.655,161.9183333,151.415,187.25,190.120375,231.7966667,190.7788889,240.0366667])

])

airport_datas=np.array([

    np.array(x) for x in range(len(airport_targets[0]))

])
tier_a  = np.array([64.586667,66.213333,67.593333,69.23,70.696667,71.593333,72,72.106667,72.84])

tier_b = np.array([63.86,66.265,68.135,69.615,69.44,69.10945,68.85,71.875,70.05])

university_targets = [1e4*tier_a, 1e4*tier_b]



university_datas = np.array([

    np.array(x-2015) for x in range(len(university_targets[0]))

])
def load_data_regression(X,y,idx):

    datas = X

    targets = y

    dataset = dict(data=datas,target=targets,feature_names=['time(season)'])

    return train_test_split(datas,targets,test_size=0.3,random_state=0)
md = []

regr_list = []

score_list = []

line_list = []

for i in range(len(wti_targets)):

    md.append(load_data_regression(wti_datas,wti_targets[i],i))

    regr = SVR(kernel='linear',C=100000)

    regr.fit(md[i][0].reshape(-1,1),md[i][2])

    score_list.append(regr.score(md[i][1].reshape(-1,1),md[i][3]))

    regr_list.append(regr)

    line_list.append([-regr.coef_[0][0],1,-regr.intercept_[0]])#-kx+y-b=0
md = []

score_list = []

for i in range(len(airport_targets)):

    md.append(load_data_regression(airport_datas,airport_targets[i],i))

    regr = SVR(kernel='linear',C=100000)

    regr.fit(md[i][0].reshape(-1,1),md[i][2])

    score_list.append(regr.score(md[i][1].reshape(-1,1),md[i][3]))

    regr_list.append(regr)

    line_list.append([-regr.coef_[0][0],1,-regr.intercept_[0]])#-kx+y-b=0
md = []

for i in range(len(university_targets)):

    md.append(load_data_regression(university_datas,university_targets[i],1))

    regr=SVR(kernel='linear',C=100000)

    regr.fit(md[i][0].reshape(-1,1),md[i][2])

    score_list.append(regr.score(md[i][1].reshape(-1,1),md[i][3]))

    regr_list.append(regr)

    line_list.append([-regr.coef_[0][0],1,-regr.intercept_[0]])#-kx+y-b=0
import math

"""

line_length = |Ax0+By0+C|/(sqrt(A^2+B^2))

"""

def lineLength(line, point):

    return float(abs(line[0]*point[0]+line[1]*point[1]+line[2]))/float(math.sqrt(line[0]**2+line[1]**2))
length_list = []

for target in range(len(wti_targets)):

    targets = wti_targets[target]

    length_list.append([])

    for i in range(len(targets)):

        ll = lineLength(line_list[target],[targets[i],i])

        length_list[target].append(ll)



llst=[]

for t in range(len(airport_targets)):

    ts = airport_targets[t]

    llst.append([])

    for i in range(len(ts)):

        ll = lineLength(line_list[t],[ts[i],i])

        llst[t].append(ll)



for i in llst:

    length_list.append(i)

    

lllst=[]

for t in range(len(university_targets)):

    ts = university_targets[t]

    lllst.append([])

    for i in range(len(ts)):

        ll = lineLength(line_list[t],[ts[i],i])

        lllst[t].append(ll)



for i in lllst:

    length_list.append(i)

    

length_regr_list=[]

for i in range(len(length_list)):

    length_datas = load_data_regression(

        np.array([np.array(x) for x in range(len(length_list[i]))]),

        np.array(length_list[i]),0

    )

    regr=SVR(kernel='linear',C=100000)

    regr.fit(length_datas[0].reshape(-1,1),length_datas[2])

    score_list.append(regr.score(length_datas[1].reshape(-1,1),length_datas[3]))

    length_regr_list.append(regr)

    line_list.append([-regr.coef_[0][0],1,-regr.intercept_[0]])#-kx+y-b=0
def predict_future_population(tier,year,fluctuate=True,k=0.05,ltype=0):

    y=None

    f=None

    if ltype==0:

        y = (1/4)*(

            regr_list[tier+ltype*3].predict(np.array(year*4).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year*4+1).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year*4+2).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year*4+3).reshape(-1,1))

        )

        f = [

            length_regr_list[tier+ltype*3].predict(np.array(year*4).reshape(-1,1)),

            length_regr_list[tier+ltype*3].predict(np.array(year*4+1).reshape(-1,1)),

            length_regr_list[tier+ltype*3].predict(np.array(year*4+2).reshape(-1,1)),

            length_regr_list[tier+ltype*3].predict(np.array(year*4+3).reshape(-1,1))

        ]

        f = k*((random.randint(-1,1)*f[0][0]+random.randint(-1,1)*f[1][0]+random.randint(-1,1)*f[2][0]+random.randint(-1,1)*f[3][0])/4)

    elif ltype==1:

        year_type = year%4

        if year_type==0:

            y = (1/3)*(regr_list[tier+ltype*3].predict(np.array(year).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year+1).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year+2).reshape(-1,1))

            )

            f = [

                length_regr_list[tier+ltype*3].predict(np.array(year).reshape(-1,1)),

                length_regr_list[tier+ltype*3].predict(np.array(year+1).reshape(-1,1)),

                length_regr_list[tier+ltype*3].predict(np.array(year+2).reshape(-1,1))

            ]

            f = k*((random.randint(-1,1)*f[0][0]+random.randint(-1,1)*f[1][0]+random.randint(-1,1)*f[2][0])/3)

        else:

            y = (1/2)*(regr_list[tier+ltype*3].predict(np.array(year+3+(year_type-1)*2).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year+4+(year_type-1)*2).reshape(-1,1))+

            regr_list[tier+ltype*3].predict(np.array(year+5+(year_type-1)*2).reshape(-1,1))

            )

            f = [

                length_regr_list[tier+ltype*3].predict(np.array(year+3+(year_type-1)*2).reshape(-1,1)),

                length_regr_list[tier+ltype*3].predict(np.array(year+4+(year_type-1)*2).reshape(-1,1)),

                length_regr_list[tier+ltype*3].predict(np.array(year+5+(year_type-1)*2).reshape(-1,1))

            ]

            f = k*((random.randint(-1,1)*f[0][0]+random.randint(-1,1)*f[1][0])/2)

    elif ltype == 2:

        y = regr_list[tier+ltype*3].predict(np.array(year)).reshape(-1,1)

        f = k*random.randint(-1,1)*length_regr_list[tier+ltype*3].predict(np.array(year).reshape(-1,1))

    if fluctuate==False:

        return y[0]

    return (y+f)[0]
def calculate_ppv_coefficience(tier,present_year=2017,year_interval=1,use_fluctuation=True,alpha=0.3,max_weight=0.7,k=0.05,ltype=0,base=2016):

    beta = max_weight-alpha

    present_year -= base

    initial_year = present_year-year_interval

    present_value = 0

    present_value = predict_future_population(tier,np.array([present_year]).reshape(-1,1),k=k,fluctuate=use_fluctuation,ltype=ltype)

    pre_value = 0

    future_value = 0

    for i in range(year_interval):

        pre_value += predict_future_population(tier,np.array([initial_year+i]).reshape(-1,1),k=k,fluctuate=use_fluctuation,ltype=ltype)

    for j in range(year_interval):

        future_value += predict_future_population(tier,np.array([present_year+i]).reshape(-1,1),k=k,fluctuate=use_fluctuation,ltype=ltype)

    return 12*math.floor(alpha*pre_value+beta*(future_value-present_value))
#ppv_wti_coefficience_list

#ppv_apt_coefficience_list

#ppv_uvs_coefficience_list

#calculate_ppv_coefficience(0,present_year=2015+j,k=0.05,ltype=2,base=2015)

lw = np.array([0.1208,0.3985,0.2656])#wti, apt, uvs

#city weight

cw_tiers = 100000000*np.array([

    np.array([0.406805958,0.51920274,0.547647207,0.524186331]),

    np.array([0.431399537,0.289674546,0.25792568,0.293343555]),

    np.array([0.161794505,0.191122713,0.194427114,0.182470115])

])

cw_tier_1 = cw_tiers[0]

cw_tier_2 = cw_tiers[1]

cw_tier_3 = cw_tiers[2]

t_datas = np.array([0, 1, 2, 3])
import numpy as np











class pso(object):

    def __init__(self,fitness_fun, population_size, max_steps):

        self.w = 0.6  # 惯性权重

        self.c1 = self.c2 = 2

        self.population_size = population_size  # 粒子群数量

        self.dim = 2 # 搜索空间的维度

        self.max_steps = max_steps  # 迭代次数

        self.x_bound = [0.00000000001, 1000000]  # 解空间范围

        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],

                                   (self.population_size, self.dim))  # 初始化粒子群位置

        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度

        self.fitness_fun=fitness_fun

        fitness = self.fitness_fun(self.x)

        self.p = self.x  # 个体的最佳位置

        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置

        self.individual_best_fitness = fitness  # 个体的最优适应度

        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

        self.save_optimal_p=[]





    def evolve(self):







        for step in range(self.max_steps):

            r1 = np.random.rand(self.population_size, self.dim)

            r2 = np.random.rand(self.population_size, self.dim)

            # 更新速度和权重

            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)

            self.x = self.v + self.x



            self.x = np.where(self.x < self.x_bound[0], self.x_bound[0], self.x)

            self.x = np.where(self.x > self.x_bound[1], self.x_bound[1], self.x)





            fitness = self.fitness_fun(self.x)





            # 需要更新的个体

            update_id = np.greater(self.individual_best_fitness, fitness).flatten()

            # print(update_id)



            self.p[update_id] = self.x[update_id]



            self.individual_best_fitness[update_id] = fitness[update_id]

            self.save_optimal_p.append(self.x[np.argmin(fitness)])

            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置

            if np.min(fitness) < self.global_best_fitness:

                self.pg = self.x[np.argmin(fitness)]

                self.global_best_fitness = np.min(fitness)



        return np.array(self.save_optimal_p)
import numpy as np

from sklearn.svm import SVR



cw_tiers = 100000000*np.array([

    np.array([0.406805958,0.51920274,0.547647207,0.524186331]),

    np.array([0.431399537,0.289674546,0.25792568,0.293343555]),

    np.array([0.161794505,0.191122713,0.194427114,0.182470115])

])



t_datas = np.array([0, 1, 2, 3])



def evaluate_score(regr, tier):

    loss = 0

    temporary = 0

    for i in range(4):

        temporary = regr.predict(np.array([i]).reshape(-1, 1))[0] - cw_tiers[tier][i]

    loss += temporary ** 2

    return loss



def calculate_fitness0(x):

    m,n = x.shape[0], x.shape[1]

    tier = 0

    spring_fitness = np.zeros((m,1))

    for i in range(m):

        C, gamma = x[i,0], x[i,1]

        if C <= 0:C=0.001

        if gamma<=0:gamma=0.001

        regr_cw = SVR(kernel='rbf', C=C, gamma = gamma)

        regr_cw.fit(t_datas.reshape(-1,1),cw_tiers[tier])

        spring_fitness[i]= evaluate_score(regr_cw, tier)

    return spring_fitness

def calculate_fitness1(x):

    m,n = x.shape[0], x.shape[1]

    tier = 1

    spring_fitness = np.zeros((m,1))

    for i in range(m):

        C, gamma = x[i,0], x[i,1]

        if C <= 0:C=0.001

        if gamma<=0:gamma=0.001

        regr_cw = SVR(kernel='rbf', C=C, gamma = gamma)

        regr_cw.fit(t_datas.reshape(-1,1),cw_tiers[tier])

        spring_fitness[i]= evaluate_score(regr_cw, tier)

    return spring_fitness

def calculate_fitness2(x):

    m,n = x.shape[0], x.shape[1]

    tier = 2

    spring_fitness = np.zeros((m,1))

    for i in range(m):

        C, gamma = x[i,0], x[i,1]

        if C <= 0:C=0.001

        if gamma<=0:gamma=0.001

        regr_cw = SVR(kernel='rbf', C=C, gamma = gamma)

        regr_cw.fit(t_datas.reshape(-1,1),cw_tiers[tier])

        spring_fitness[i]= evaluate_score(regr_cw, tier)

    return spring_fitness

PSO_Instance0 = pso(calculate_fitness0, 200, 300)

q_best0 = PSO_Instance0.evolve()

print(q_best0)

print()



PSO_Instance1 = pso(calculate_fitness1, 200, 300)

q_best1 = PSO_Instance0.evolve()

print(q_best1)

print()



PSO_Instance2 = pso(calculate_fitness2, 200, 300)

q_best2 = PSO_Instance2.evolve()

print(q_best2)
c_tier0 = np.mean(np.array([q_best0[x][0] for x in range(len(q_best0))]))

gamma_tier0 = np.mean(np.array([q_best0[x][1] for x in range(len(q_best0))]))

c_tier1 = np.mean(np.array([q_best1[x][0] for x in range(len(q_best1))]))

gamma_tier1 = np.mean(np.array([q_best1[x][1] for x in range(len(q_best1))]))

c_tier2 = np.mean(np.array([q_best2[x][0] for x in range(len(q_best2))]))

gamma_tier2 = np.mean(np.array([q_best2[x][1] for x in range(len(q_best2))]))

print(c_tier0,gamma_tier0,c_tier1,gamma_tier1,c_tier2,gamma_tier2)
#pso_process(100000,10,50,0,99999,9,1000,1)

regr_cw_tier_1 = SVR(kernel='rbf', C=c_tier0, gamma=gamma_tier0)

#pso_process(100000,10,50,1,99999,9,1000,1)

regr_cw_tier_2 = SVR(kernel='rbf', C=c_tier1, gamma=gamma_tier1)

#pso_process(100000,10,50,2,99999,9,1000,1)

regr_cw_tier_3 = SVR(kernel='rbf', C=c_tier2, gamma=gamma_tier2)

regr_cw_tier_1.fit(t_datas.reshape(-1,1),cw_tier_1)

regr_cw_tier_2.fit(t_datas.reshape(-1,1),cw_tier_2)

regr_cw_tier_3.fit(t_datas.reshape(-1,1),cw_tier_3)



def evaluate_score(regr,tier):

    loss = 0

    for i in range(4):

        temporary = regr.predict(np.array([i]).reshape(-1,1))[0]-cw_tiers[tier][i]

        loss += temporary**2

    return loss



fig = plt.figure(figsize=(10,3))

ax = fig.add_subplot(111)



ax.plot(np.array([x for x in range(0,30)]).reshape(-1,1),np.array([regr_cw_tier_1.predict(np.array([y]).reshape(-1,1))/100000000 for y in range(0,30)]),color='darkorange')



ax.plot(np.array([x for x in range(0,30)]).reshape(-1,1),np.array([regr_cw_tier_2.predict(np.array([y]).reshape(-1,1))/100000000 for y in range(0,30)]),color='mediumseagreen')



ax.plot(np.array([x for x in range(0,30)]).reshape(-1,1),np.array([regr_cw_tier_3.predict(np.array([y]).reshape(-1,1))/100000000 for y in range(0,30)]),color='#2761b3')



ax.legend(prop=font2,loc='best')

ax.set_title('The Predicted City Weight for Different Tiers of Cities through Time',font)

ax.set_xlabel('year',font)

ax.set_ylabel('cw',font)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.tick_params(labelsize=10)

label1 = ax.get_xticklabels() + ax.get_yticklabels()

plt.show()
cw_tier_1_list = []

cw_tier_2_list = []

cw_tier_3_list = []



for i in range(2015,2045+1):

    cw_tier_1_list.append(regr_cw_tier_1.predict(np.array(i-2015).reshape(-1,1)))

    cw_tier_2_list.append(regr_cw_tier_2.predict(np.array(i-2015).reshape(-1,1)))

    cw_tier_3_list.append(regr_cw_tier_3.predict(np.array(i-2015).reshape(-1,1)))



print("City weight prediction from year 2015 to 2045")

print("tier 1:")

print(np.array(cw_tier_1_list))

print()

print("tier 2:")

print(np.array(cw_tier_2_list))

print()

print("tier 3:")

print(np.array(cw_tier_3_list))

print()
def ppv(year,k_val,coef):

    ppv_wti_tier0 = calculate_ppv_coefficience(0,present_year=year, k=k_val, ltype=0, base=2015)

    ppv_wti_tier1 = calculate_ppv_coefficience(1,present_year=year, k=k_val, ltype=0, base=2015)

    ppv_wti_tier2 = calculate_ppv_coefficience(2,present_year=year, k=k_val, ltype=0, base=2015)

    ppv_apt_tier0 = calculate_ppv_coefficience(0,present_year=year, k=k_val, ltype=1, base=2015)

    ppv_apt_tier1 = calculate_ppv_coefficience(1,present_year=year, k=k_val, ltype=1, base=2015)

    ppv_apt_tier2 = calculate_ppv_coefficience(2,present_year=year, k=k_val, ltype=1, base=2015)

    ppv_uvs_tier0 = calculate_ppv_coefficience(0,present_year=year, k=k_val, ltype=2, base=2015)

    ppv_uvs_tier1 = calculate_ppv_coefficience(1,present_year=year, k=k_val, ltype=2, base=2015)

    ppv_wti = lw[0]*regr_cw_tier_1.predict(np.array([year-2015]).reshape(-1,1))*ppv_wti_tier0 + lw[0]*regr_cw_tier_2.predict(np.array([year-2015]).reshape(-1,1))*ppv_wti_tier1 +lw[0]*regr_cw_tier_3.predict(np.array([year-2015]).reshape(-1,1))*ppv_wti_tier2

    ppv_apt = lw[1]*regr_cw_tier_1.predict(np.array([year-2015]).reshape(-1,1))*ppv_apt_tier0 + lw[1]*regr_cw_tier_2.predict(np.array([year-2015]).reshape(-1,1))*ppv_apt_tier1 + lw[1]*regr_cw_tier_3.predict(np.array([year-2015]).reshape(-1,1))*ppv_apt_tier2

    ppv_uvs = lw[2]*regr_cw_tier_1.predict(np.array([year-2015]).reshape(-1,1))*ppv_uvs_tier0 + lw[2]*regr_cw_tier_2.predict(np.array([year-2015]).reshape(-1,1))*ppv_uvs_tier1

    ppv_raw_val = ppv_wti+ppv_apt+ppv_uvs

    ppv_avr_val = coef*(ppv_raw_val)/3

    ppv_wti -= ppv_avr_val

    ppv_apt -= ppv_avr_val

    ppv_uvs -= ppv_avr_val

    ppv_val = ppv_wti+ppv_apt+ppv_uvs

    return [ppv_val,[ppv_wti,ppv_apt,ppv_uvs],[[ppv_wti_tier0,ppv_wti_tier1,ppv_wti_tier2],[ppv_apt_tier0,ppv_apt_tier1,ppv_apt_tier2],[ppv_uvs_tier0,ppv_uvs_tier1]]]



ppv_pack = [ppv(x, 0, 0.9) for x in range(2015,2045+1)]



ppv_list_wti = [ppv_pack[x-2015][1][0] for x in range(2015,(2045+1))]

ppv_list_apt = [ppv_pack[x-2015][1][1] for x in range(2015,(2045+1))]

ppv_list_uvs = [ppv_pack[x-2015][1][2] for x in range(2015,(2045+1))]



ppv_coefficience_wti_tier1 = [ppv_pack[x-2015][2][0][0]  for x in range(2015,(2045+1))]

ppv_coefficience_wti_tier2 = [ppv_pack[x-2015][2][0][1]  for x in range(2015,(2045+1))]

ppv_coefficience_wti_tier3 = [ppv_pack[x-2015][2][0][2]  for x in range(2015,(2045+1))]

ppv_coefficience_apt_tier1 = [ppv_pack[x-2015][2][1][0]  for x in range(2015,(2045+1))]

ppv_coefficience_apt_tier2 = [ppv_pack[x-2015][2][1][1]  for x in range(2015,(2045+1))]

ppv_coefficience_apt_tier3 = [ppv_pack[x-2015][2][1][2]  for x in range(2015,(2045+1))]

ppv_coefficience_uvs_tier1 = [ppv_pack[x-2015][2][2][0]  for x in range(2015,(2045+1))]

ppv_coefficience_uvs_tier2 = [ppv_pack[x-2015][2][2][1]  for x in range(2015,(2045+1))]



ppv_list = [ppv_pack[x-2015][0] for x in range(2015,(2045+1))]

fig = plt.figure(figsize=(15,3))

ax = fig.add_subplot(141)

ax.plot([x for x in range(len(ppv_list))],ppv_list,'s-',color='#ff0000')

ax = fig.add_subplot(142)

ax.plot([x for x in range(len(ppv_list_wti))],ppv_list_wti,'s-',color='darkorange')

ax = fig.add_subplot(143)

ax.plot([x for x in range(len(ppv_list_apt))],ppv_list_apt,'s-',color='mediumseagreen')

ax = fig.add_subplot(144)

ax.plot([x for x in range(len(ppv_list_uvs))],ppv_list_uvs,'s-',color='#2716a3')



plt.show()
fig = plt.figure(figsize=(12,3))

ax = fig.add_subplot(131)

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_wti_tier1,color='darkorange')

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_wti_tier2,color='mediumseagreen')

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_wti_tier3,color='#2716a3')

ax = fig.add_subplot(132)

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_apt_tier1,color='darkorange')

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_apt_tier2,color='mediumseagreen')

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_apt_tier3,color='#2716a3')

ax = fig.add_subplot(133)

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_uvs_tier1,color='darkorange')

ax.plot([x for x in range(2015,2045+1)],ppv_coefficience_uvs_tier2,color='mediumseagreen')

plt.show()

print("PPV coefficience datas:\n")

print("Shopping Mall Datas:")

print("\tPPV_coefficience_wti_tier1:")

print("\t",ppv_coefficience_wti_tier1)

print("\tPPV_coefficience_wti_tier2:")

print("\t",ppv_coefficience_wti_tier2)

print("\tPPV_coefficience_wti_tier3:")

print("\t",ppv_coefficience_wti_tier3)

print("\nAirport Datas:")

print("\tPPV_coefficience_apt_tier1:")

print("\t",ppv_coefficience_apt_tier1)

print("\tPPV_coefficience_apt_tier2:")

print("\t",ppv_coefficience_apt_tier2)

print("\tPPV_coefficience_apt_tier3:")

print("\t",ppv_coefficience_apt_tier3)

print("\nUniversity Datas:")

print("\tPPV_coefficience_uvs_tier1:")

print("\t",ppv_coefficience_uvs_tier1)

print("\tPPV_coefficience_uvs_tier2:")

print("\t",ppv_coefficience_uvs_tier2)

print("ppv_list = ", ppv_list)

print("ppv_list_wti = ", ppv_list_wti)

print("ppv_list_apt = ", ppv_list_apt)

print("ppv_list_uvs = ", ppv_list_uvs)
def sigmoid(x, kd=5*1e14, sc=100, bias=0):

    return sc/(1+np.exp(-x/kd))+bias
sgm_ppv_wti = sigmoid(np.array(ppv_list_wti))

sgm_ppv_apt = sigmoid(np.array(ppv_list_apt))

sgm_ppv_uvs = sigmoid(np.array(ppv_list_uvs))

fig = plt.figure(figsize=(10,3))

ax = fig.add_subplot(111)

ax.plot(np.array([x for x in range(2015,2045+1)]),sgm_ppv_wti,color='darkorange')

ax.plot(np.array([x for x in range(2015,2045+1)]),sgm_ppv_apt,color='mediumseagreen')

ax.plot(np.array([x for x in range(2015,2045+1)]),sgm_ppv_uvs,color='#2716a3')



plt.show()
print("after sigmoid PPVs:->\n")

print("a) Shopping Mall:")

print(sgm_ppv_wti)

print()

print("b) Airport:")

print(sgm_ppv_apt)

print()

print("c) University:")

print(sgm_ppv_uvs)