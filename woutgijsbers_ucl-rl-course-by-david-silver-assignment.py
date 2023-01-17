import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Some helper functions for plotting graphs.

def plot_dct(dct, x_val, y_val, name):
    # Plots a 2D linegraph from a dictionary.
    plt.figure(figsize=[15, 8])
    ax = plt.axes()
    x = [key for key in dct]
    y = [dct[key] for key in x]
    ax.plot(x, y)
    ax.set(xlabel=x_val, ylabel=y_val, title=name)
    plt.show()

def plot_Vstar(Q):
    # Plots the optimized state value function.
    fig = plt.figure(figsize=[15, 8])
    ax = fig.add_subplot(111, projection='3d')

    x = range(10)
    y = range(21)
    X, Y = np.meshgrid(x, y)

    V_star = np.zeros((21, 10))
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            V_star[j, i] = max(Q[i, j])
    Z = np.array(V_star)

    ax.set_title("V* for each state")
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player total')
    ax.set_zlabel('V*')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()

# Assignment 1: Implementation of Easy21.

class Easy21():
    def __init__(self):
        self.cardValMin, self.cardValMax = 1, 10
        self.validValMin, self.validValMax = 1, 21
        self.dealerValPass = 17
        
    def setup(self):
        # Initializes game. Deals out starting cards. s0, s1 = dealer, player.
        card_dealer = random.randint(self.cardValMin, self.cardValMax)
        card_player = random.randint(self.cardValMin, self.cardValMax)
        s = (card_dealer, card_player)
        return s

    def check_terminal(self, s):
        # Checks to see if terminal state has been reached.
        if s[0] < self.validValMin or s[0] > self.validValMax:
            r = 1
            return True, r
        elif s[1] < self.validValMin or s[1] > self.validValMax:
            r = -1
            return True, r
        elif s[0] >= self.dealerValPass:
            if s[0] == s[1]:
                r = 0
            elif s[0] > s[1]:
                r = -1
            else:
                r = 1
            return True, r
        else:
            r = 0
            return False, r

    def draw(self):
        # Draws a card according to an equal distribution across values 1-10 
        # and a 1 to 2 distribution across red and black.
        value = random.randint(1, 10)
        if np.random.random() <= 1/3:
            return -value
        else:
            return value

    def step(self, s, a):
        # Performs one step of the game and returns the reward and next state.
        # 0 = Hit, 1 = Stick
        if a == 0:
            val = s[1] + self.draw()
            s = (s[0], val)
            terminal, r = self.check_terminal(s)
            if terminal:
                s = 'terminal'
        else:
            terminal = False
            while not terminal:
                val = s[0] + self.draw()
                s = (val, s[1])
                terminal, r = self.check_terminal(s)
            s = 'terminal'
        return r, s

# Assignment 2: Monte-Carlo control in Easy21.

class MonteCarlo():
    def __init__(self, environment, N0=100):
        self.env = environment
        self.Q = np.zeros((10, 21, 2))
        self.Nv = np.zeros((10, 21))
        self.Nq = np.zeros((10, 21, 2))
        self.N0 = N0
    
    def epsilonGreedy(self, s):
        # Implementation of an epsilon greedy policy.
        epsilon = self.N0 / (self.N0 + self.Nv[s[0] - 1, s[1] - 1])
        choice = random.choices(['exploit', 'explore'], weights=[1 - epsilon, epsilon])
        if choice == 'exploit':
            a = np.argmax(self.Q[s[0] - 1, s[1] - 1, a] for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a                             
                                         
    def play(self):
        # Runs through one episode of the game.
        q_visited = []
        s = self.env.setup()
        while s != 'terminal':
            self.Nv[s[0] - 1][s[1] - 1] += 1
            a = self.epsilonGreedy(s)
            r, s1 = self.env.step(s, a)
            q_visited.append((s + (a,) + (r,)))
            s = s1
        return q_visited
                                         
    def run(self, iterations):
        # Runs through the specified number of episodes and 
        # updates the state-action function accordingly.
        for i in range(iterations):
            q_visited = self.play()
            g = sum(q[3] for q in q_visited)
            for q in q_visited:
                self.Nq[q[0] - 1, q[1] - 1, q[2]] += 1
                alpha = 1 / self.Nq[q[0] - 1, q[1] - 1, q[2]]
                self.Q[q[0] - 1, q[1] - 1, q[2]] += \
                alpha * (g - self.Q[q[0] - 1, q[1] - 1, q[2]])

env = Easy21()
model = MonteCarlo(env)
model.run(iterations=10000000)
plot_Vstar(model.Q)

# Assignment 3: TD learning in Easy21.

class Sarsa:
    def __init__(self, environment, N0=100):
        self.env = environment
        self.Q = np.zeros((10, 21, 2))
        self.E = np.zeros((10, 21, 2))
        self.Nv = np.zeros((10, 21))
        self.Nq = np.zeros((10, 21, 2))
        self.N0 = N0
        self.mse = {}

    def epsilon_greedy(self, s):
        # Implementation of an epsilon greedy policy.
        epsilon = self.N0 / (self.N0 + self.Nv[s[0] - 1, s[1] - 1])
        choice = random.choices(['exploit', 'explore'], weights=[1 - epsilon, epsilon])
        if choice == 'exploit':
            a = np.argmax(self.Q[s[0] - 1, s[1] - 1, a] for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a

    def alpha(self, q):
        # Calculates learning rate
        return 1 / self.Nq[q[0], q[1], q[2]]

    def play(self, lmbda):
        # Runs through one episode of the game and 
        # updates the state value function after each step.
        q_visited = []
        s = self.env.setup()
        a = self.epsilon_greedy(s)
        while s != 'terminal':
            self.E[s[0] - 1, s[1] - 1, a] += 1
            r, s1 = self.env.step(s, a)
            q = s + (a,)
            q_visited.append(q)
            self.Nq[q[0] - 1, q[1] - 1, q[2]] += 1
            if s1 == 'terminal':
                td_error = r - self.Q[s[0] - 1, s[1] - 1, a]
                s = s1
            else:
                a1 = self.epsilon_greedy(s1)
                td_error = r + self.Q[s1[0] - 1, s1[1] - 1, a1] - self.Q[s[0] - 1, s[1] - 1, a]
                s, a = s1, a1
            for (i, j, k) in set(q_visited):
                self.Q[i - 1, j - 1, k] += \
                self.alpha((i - 1, j - 1, k)) * td_error * self.E[i - 1, j - 1, k]
                self.E[i - 1, j - 1, k] *= lmbda

    def run(self, lmbda, Q_true, iterations):
        # Runs through the specified number of episodes and 
        # stores learning rate for lambda = 0 and lambda = 1.
        for i in range(iterations):
            self.E = np.zeros((10, 21, 2))
            self.play(lmbda)
            if lmbda == 0 or lmbda == 1:
                mse = np.sum(np.square(self.Q - Q_true)) / 420
                self.mse[i] = mse

env = Easy21()
model1 = Sarsa(env)
lmbda_mse = {}
for lmbda in list(np.arange(0, 11) / 10):
    model1.__init__(env)
    model1.run(lmbda, model.Q, iterations=10000)
    mse = np.sum(np.square(model1.Q - model.Q)) / 420
    print(f"MSE of Sarsa({lmbda}): ", mse)
    lmbda_mse[lmbda] = mse
    if lmbda == 0 or lmbda == 1:
        plot_dct(model1.mse, 'Episode', 'MSE', f'Learning curve of lambda = {lmbda}')
plot_dct(lmbda_mse, 'Lambda', 'MSE', 'MSE against Lambda')

# Assignment 4: Linear Function Approximation in Easy21.

class LinFuncApprox:
    def __init__(self, environment):
        self.env = environment
        self.w = np.zeros(36)
        self.Q = np.zeros((10, 21, 2))
        self.mse = {}
        self.epsilon = 0.05
        self.alpha = 0.01
        # Create list of features.
        self.f = []
        d = list(tuple(zip(range(1, 8, 3), range(4, 11, 3))))
        p = list(tuple(zip(range(1, 17, 3), range(6, 22, 3))))
        a = [(0,), (1,)]
        for i in range(len(d)):
            for j in range(len(p)):
                for k in range(len(a)):
                    temp = [d[i], p[j], a[k]]
                    self.f.append(temp)

    def sa_to_x(self, s, a):
        # Converts state-action pair to feature vector.
        x = np.zeros(36)
        for idx, feat in enumerate(self.f):
            x[idx] = (feat[0][0] <= s[0] <= feat[0][1] and \
                      feat[1][0] <= s[1] <= feat[1][1] and a == feat[2][0])
        return x

    def sa_to_q(self, s, a):
        # Calculates state-action value from state-action pair.
        x = self.sa_to_x(s, a)
        return sum(x * self.w)

    def epsilon_greedy(self, s):
        # Implementation of an epsilon greedy policy.
        choice = random.choices(['exploit', 'explore'],\
                                weights=[1 - self.epsilon, self.epsilon])
        if choice == 'exploit':
            a = np.argmax(self.sa_to_q(s, a) for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a

    def play(self, lmbda):
        # Runs through one episode of the game and updates the feature weights after each step
        # according to eligibility traces.
        E = np.zeros(36)
        s = self.env.setup()
        a = self.epsilon_greedy(s)
        while s != 'terminal':
            r, s1 = self.env.step(s, a)
            if s1 == 'terminal':
                td_error = r - self.sa_to_q(s, a)
            else:
                a1 = self.epsilon_greedy(s1)
                td_error = r + self.sa_to_q(s1, a1) - self.sa_to_q(s, a)
                a = a1
            E = lmbda * E + self.sa_to_x(s, a)
            self.w += self.alpha * td_error * E
            s = s1

    def form_Q(self):
        # Forms a 3D matrix containing all state-action values.
        for d in range(1, 11):
            for p in range(1, 22):
                for a in range(2):
                    s = (d, p)
                    self.Q[d - 1, p - 1, a] = self.sa_to_q(s, a)

    def run(self, lmbda, Q_true, iterations):
        # Initializes random weights and runs through the specified number of episodes.
        # Also stores learning rate for lambda = 0 and lambda = 1.
        self.w = np.random.rand(36)
        self.Q = np.zeros((10, 21, 2))
        self.mse = {}
        for i in range(iterations):
            self.play(lmbda)
            if lmbda == 0 or lmbda == 1:
                if i % 100 == 0:
                    self.form_Q()
                    mse = np.sum(np.square(self.Q - Q_true)) / 420
                    self.mse[i] = mse
        self.form_Q()

env = Easy21()
model2 = LinFuncApprox(env)
lmbda_mse = {}
for lmbda in list(np.arange(0, 11) / 10):
    model2.run(lmbda, model.Q, iterations=10000)
    mse = np.sum(np.square(model2.Q - model.Q)) / 420
    print(f"MSE of Sarsa({lmbda}) with a Linear Function Approximator: ", mse)
    lmbda_mse[lmbda] = mse
    if lmbda == 0 or lmbda == 1:
        plot_dct(model2.mse, 'Episode', 'MSE', f'Learning curve of lambda = {lmbda}')
plot_dct(lmbda_mse, 'Lambda', 'MSE', 'MSE against Lambda')
