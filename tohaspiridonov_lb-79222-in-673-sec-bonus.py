import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time # performace checking

import copy # data helpers
N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125

N_CHOISES = 10



def penalty_f(c, n):

    if c == 0:

        return 0

    elif c == 1:

        return 50

    elif c == 2:

        return 50 + 9 * n

    elif c == 3:

        return 100 + 9 * n

    elif c == 4:

        return 200 + 9 * n

    elif c == 5:

        return 200 + 18 * n

    elif c == 6:

        return 300 + 18 * n

    elif c == 7:

        return 300 + 36 * n

    elif c == 8:

        return 400 + 36 * n

    elif c == 9:

        return 500 + 36 * n + 199 * n

    else:

        return 500 + 36 * n + 398 * n    
class OptimizedData:

    def __init__(self):

        s__ = time.time()

        data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

        self.pd_data = data

        self.np_id_offset = 0

        self.np_n_offset = 11

        self.np_choise_offset = 1

        self.np_data = np.array(data)

        for i, r in enumerate(self.np_data):

            if r[self.np_id_offset] != i:

                raise 'Blah'

                

        choise_checker = [{r[self.np_choise_offset + i]: i for i in range(10)} for r in self.np_data]

        self.which_shortcut = np.array([[

            0 if d == 0 else choise_checker[f][d] if d in choise_checker[f] else 10

            for f in range(self.families())] 

            for d in range(N_DAYS+1)])

        self.penalty_shortcut = np.array([[

            penalty_f(self.which_shortcut[d, f], self.n(f))

            for f in range(self.families())] 

            for d in range(N_DAYS+1)])

        print('Init Santas Data in', time.time() - s__, 'seconds')

    def n(self, id_):

        return self.np_data[id_, self.np_n_offset]

    def choise(self, id_, i):

        return self.np_data[id_, self.np_choise_offset + i]

    def which(self, id_, i):

        return self.which_shortcut[i, id_]

    def families(self):

        return len(self.np_data)

    def penalty(self, id_, i):

        return self.penalty_shortcut[i, id_]

data = OptimizedData()
class OptimisedSubmission:

    def __init__(self, pair_values_to_copy = None, list_values_to_copy = None, rhs = None):

        if rhs is not None:

            self.values = rhs.values.copy()

            self.daily_occupancy = rhs.daily_occupancy.copy()

            self.under_occuped = rhs.under_occuped

            self.over_occuped = rhs.over_occuped

            self.penalty = rhs.penalty

            self.accounting_cost = rhs.accounting_cost

            return

        self.values = [0 for k in range(data.families())]

        self.daily_occupancy = [0 for k in range(N_DAYS+1)]

        self.under_occuped = N_DAYS

        self.over_occuped = 0

        self.penalty = 0

        self.accounting_cost = 0

        if pair_values_to_copy is not None:

            for f, d in pair_values_to_copy:

                self.set_f(f, d)

        if list_values_to_copy is not None:

            for f, d in enumerate(list_values_to_copy):

                self.set_f(f, d)

    

    def copy(self):

        lhs = OptimisedSubmission(rhs=self)

        return lhs

                

    def __accounting_cost(self, d):

        today_count = self.daily_occupancy[d]

        if today_count <= 125:

            return 0

        yesterday_count = self.daily_occupancy[d+1] if d < N_DAYS else today_count

        diff = abs(today_count - yesterday_count)

        return (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0)

                

    def __set_daily_delta(self, d, n):

        b = self.daily_occupancy[d]

        if b < MIN_OCCUPANCY:

            self.under_occuped -= 1

        if b > MAX_OCCUPANCY:

            self.over_occuped -= 1

        self.accounting_cost -= self.__accounting_cost(d-1)

        self.accounting_cost -= self.__accounting_cost(d)

        a = b + n

        self.daily_occupancy[d] = a

        self.accounting_cost += self.__accounting_cost(d)

        self.accounting_cost += self.__accounting_cost(d-1)

        if a < MIN_OCCUPANCY:

            self.under_occuped += 1

        if a > MAX_OCCUPANCY:

            self.over_occuped += 1

                

    def set_f(self, f, i):

        s = self.values[f]

        if s == i:

            return

        n = data.n(f)

        if s > 0: 

            self.__set_daily_delta(s, -n)

            self.penalty -= data.penalty(f, s)

        self.values[f] = i

        if i > 0:

            self.__set_daily_delta(i, n)

            self.penalty += data.penalty(f, i)

            

    def is_valid(self):

        return self.under_occuped == 0 and self.over_occuped == 0

    

    def cost_function(self):

        if not self.is_valid():

            raise('Blah')

        return self.penalty + self.accounting_cost
def complete_somehow(self):

    residual = []

    for f, d in enumerate(self.values):

        if d < 1:

            residual += [[data.n(f), f]]

    daily_occupancy = [[self.daily_occupancy[k], k] for k in range(1, N_DAYS+1)]

    residual = sorted(residual)[::-1]

    for n, f in residual:

        daily_occupancy=sorted(daily_occupancy)

        self.set_f(f, daily_occupancy[0][1])

        daily_occupancy[0][0] += n

    return self



s__ = time.time()

print('Some valid submission LB=', complete_somehow(OptimisedSubmission()).cost_function(), 'build in', (time.time() - s__), 'seconds')
def to_max_choise_k(self, limit, k_):

    residual = [[] for k in range(N_DAYS+1)]

    for f, d in enumerate(self.values):

        if d < 1:

            residual[data.choise(f, k_)] += [[data.n(f), f]]

    for d in range(N_DAYS+1):

        rday = sorted(residual[d])[::-1]

        for n, f in rday:

            if self.daily_occupancy[d] >= limit:

                break

            self.set_f(f, d)

    return self



def to_max_choise(self, limit, k_=9):

    for i in range(k_+1):

        to_max_choise_k(self, limit, i)

    return self



def some_optimized(self):

    to_max_choise(self, MIN_OCCUPANCY)

    to_max_choise(self, (3*MIN_OCCUPANCY+MAX_OCCUPANCY)//4)

    to_max_choise(self, (MIN_OCCUPANCY+MAX_OCCUPANCY)//2)

    to_max_choise(self, (MIN_OCCUPANCY+3*MAX_OCCUPANCY)//4)

    return self



s__ = time.time()

preoptimized = complete_somehow(some_optimized(OptimisedSubmission()))

print('Some resonable submission LB=', preoptimized.cost_function(), 'build in', (time.time() - s__), 'seconds')
def incremental_optimize_by_one_item(self):

    best = self.cost_function()

    for f in range(data.families()):

        s = self.values[f]

        for i in range(5):

            self.set_f(f, data.choise(f, i))

            c = self.penalty + self.accounting_cost

            if c < best:

                if self.is_valid():

                    best = c

                    s = data.choise(f, i)

                    break

        self.set_f(f, s)

    return self

incremental_optimize_by_one_item(preoptimized)

print('Optimised submission LB=', preoptimized.cost_function(), 'build in', (time.time() - s__), 'seconds')
def optimize_by_sequential_pairs(self):

    begin = time.time()

    now = time.time()

    for f1 in range(data.families()):

        s1 = self.values[f1]

        for f2 in range(data.families()):

            s2 = self.values[f2]

            i1 = data.which(f1, s2)

            if (i1 >= 4 or s2 == s1) and f2 != f1:

                continue

            cp = self.copy()

            cp.set_f(f1, s2)

            for i2 in range(4):

                pr = data.choise(f2, i2)

                cp.set_f(f2, pr)

                c = cp.penalty + cp.accounting_cost

                if c < self.cost_function():

                    if cp.is_valid():

                        s1 = cp.values[f1]

                        s2 = cp.values[f2]

                        break

            self.set_f(f1, s1)

            self.set_f(f2, s2)

            if time.time()>now+2:

                now = time.time()

                print('Keep computing, at', int(1000*f1/data.families())/10, '%, current cost_function is ', int(10*self.cost_function())/10, end='   \r')

    now = time.time()

    if now-begin > 2:

        print()

    return self

optimize_by_sequential_pairs(preoptimized)

print('Optimised submission LB=', preoptimized.cost_function(), 'build in', (time.time() - s__), 'seconds')
# Uncomment to get LB=76168 in 2936 sec

def optimize_by_neighbouring_pairs(self):

    begin = time.time()

    now = time.time()

    for f1 in range(data.families()):

        s1 = self.values[f1]

        for f2 in range(data.families()):

            for d in range(-1, 2):

                s2 = self.values[f2]

                if s2+d < 1 or s2+d > N_DAYS:

                    continue

                i1 = data.which(f1, s2 + d)

                if (i1 >= 4 or s2 == s1) and f2 != f1:

                    continue

                cp = self.copy()

                cp.set_f(f1, s2 + d)

                for i2 in range(4):

                    pr = data.choise(f2, i2)

                    cp.set_f(f2, pr)

                    c = cp.penalty + cp.accounting_cost

                    if c < self.cost_function():

                        if cp.is_valid():

                            s1 = cp.values[f1]

                            s2 = cp.values[f2]

                            break

                self.set_f(f1, s1)

                self.set_f(f2, s2)

                if time.time()>now+5:

                    now = time.time()

                    print('Keep computing, at', int(1000*f1/data.families())/10, '%, current cost_function is ', int(10*self.cost_function())/10, end='   \r')

    now = time.time()

    if now-begin > 5:

        print()

    return self

#optimize_by_neighbouring_pairs(preoptimized)

#print('Optimised submission LB=', preoptimized.cost_function(), 'build in', (time.time() - s__), 'seconds')
# Uncomment to get LB~74500 in another 40 mins

def optimize_by_rebalancing_hard_days(self):

    begin = time.time()

    now = time.time()        

    for d in range(1, N_DAYS+1):

        possible_options = dict()

        positive_options = dict()

        positive = 0

        for f in range(data.families()):

            s = self.values[f]

            if s != d:

                if data.which(f, d) >= 4:

                    continue

                if self.daily_occupancy[s] - data.n(f) < MIN_OCCUPANCY:

                    continue

                cp = self.copy()

                cp.set_f(f, d)

                delta = cp.penalty - self.penalty

                possible_options[(f, d)] = delta

                if delta < 0:

                    positive_options[(f, d)] = delta

            else:

                cp = self.copy()

                for i in range(4):

                    t = data.choise(f, i)

                    if t == d:

                        continue

                    if self.daily_occupancy[t] - data.n(f) > MAX_OCCUPANCY:

                        continue

                    cp.set_f(f, t)

                    delta = cp.penalty - self.penalty

                    possible_options[(f, t)] = delta

                    if delta < 0:

                        positive_options[(f, t)] = delta

        pol = list(possible_options)

        for f1, s1 in positive_options:

            for i2 in range(len(pol)):

                f2, s2 = pol[i2]

                if (s1 == d) == (s2 == d):

                    continue

                if -positive_options[(f1, s1)] < possible_options[(f2, s2)]:

                    continue

                cf = self.cost_function()

                cp = self.copy()

                b1=cp.values[f1]

                cp.set_f(f1, s1)

                b2=cp.values[f2]

                cp.set_f(f2, s2)

                br = False

                for f3, s3 in pol[i2:]:

                    if -positive_options[(f1, s1)] < possible_options[(f3, s3)]:

                        continue

                    b3=cp.values[f3]

                    cp.set_f(f3, s3)                

                    if cp.is_valid() and cp.cost_function() < cf - 0.00001:

                        self.set_f(f1, cp.values[f1])

                        self.set_f(f2, cp.values[f2])

                        self.set_f(f3, cp.values[f3])

                        br = True

                        break

                    cp.set_f(f3, b3)

                if br:

                    break

                cp.set_f(f2, b2)                    

                cp.set_f(f1, b1)

                if time.time()>now+5:

                    now = time.time()

                    print('Keep computing, current cost_function is ', int(10*self.cost_function())/10, end='   \r')

    now = time.time()

    if now-begin > 5:

        print()

    return self

#optimize_by_rebalancing_hard_days(preoptimized)

#print('Optimised submission LB=', preoptimized.cost_function(), 'build in', (time.time() - s__), 'seconds')