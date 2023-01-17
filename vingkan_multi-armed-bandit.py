import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("hls")
def simulate_bandit(bandit, ads, df, verbose=False):
    pal = sns.color_palette("hls", len(ads) + 1)
    counts = np.zeros(len(ads))
    history = np.zeros((len(df), len(ads)))
    earnings = np.zeros(len(df))
    total_earnings = 0
    for i, visitor in enumerate(df.values):
        idx, ad = bandit.get()
        payoff = visitor[idx]
        bandit.give(idx, payoff)
        if verbose:
            print("Show {0} to visitor {1}, earn {2:.3f}".format(ad, i, payoff))
        counts[idx] += 1
        history[i] = counts
        total_earnings += payoff
        earnings[i] = total_earnings
    print("Total Payoff = {0:.3f}".format(total_earnings))
    time = list(range(len(df)))
    for i, (ad, color) in enumerate(zip(ads, pal)):
        sns.lineplot(x=time, y=history[:,i], label=ad, color=color)
    plt.title(type(bandit).__name__)
    plt.xlabel("Visitors")
    plt.ylabel("Number of Times Chosen")
    plt.legend()
    plt.show()
SEED = 5682
np.random.seed(SEED)
N = 1000
df = pd.DataFrame()
df["A"] = (np.random.ranf(size=N) >= 0.89).astype(int)
df["B"] = (np.random.ranf(size=N) >= 0.38).astype(int)
df["C"] = (np.random.ranf(size=N) >= 0.31).astype(int)
ads = df.columns
df.head()
pal = sns.color_palette("hls", len(ads) + 1)
fig, row = plt.subplots(1, len(ads), sharey=True)
for ad, cell, color in zip(ads, row, pal):
    sns.distplot(df[ad], label=ad, bins=2, kde=False, ax=cell, color=color)
    cell.set_xlabel(ad)
class Bandit():
    
    def __init__(self, arms, epsilon=0.2, initial=0, random_state=None):
        self.arms = arms
        if len(arms) < 1:
            raise ValueError("Array-like `arms` must contain at least one value.")
        self.payoffs = np.zeros(len(arms))
        self.counts = np.zeros(len(arms))
        self.initial = initial
        self.epsilon = epsilon
        np.random.seed(random_state)
        
    def get(self):
        if np.random.ranf() < self.get_epsilon():
            # Explore
            index = np.random.randint(0, len(self.arms))
            return index, self.arms[index]
        else:
            # Exploit
            index = self.get_best()
            return index, self.arms[index]
    
    def give(self, index, payoff):
        self.payoffs[index] += payoff
        self.counts[index] += 1
        
    def get_best(self):
        zero = 1e-10
        mean_payoffs = self.payoffs / (self.counts + zero)
        best_idx = np.argmax(mean_payoffs)
        return best_idx
        
    def clear(self):
        self.best = self.initial
        self.payoffs = np.zeros(len(arms))
        
    def get_epsilon(self):
        return self.epsilon

class EpsilonGreedyBandit(Bandit):
    
    pass


class EpsilonFirstBandit(Bandit):
    
    def __init__(self, *args, trials=100, **kwargs):
        if trials < 0:
            raise ValueError("Numerical `trials` cannot be negative.")
        self.trials = trials
        super().__init__(*args, **kwargs)
    
    def get_epsilon(self):
        end_exploration = self.epsilon * self.trials
        turn = self.counts.sum()
        if turn < end_exploration:
            return 1
        return 0

        
class EpsilonDecayBandit(Bandit):
    
    def __init__(self, *args, decay=1, **kwargs):
        if decay < 0:
            raise ValueError("Numerical `decay` cannot be negative.")
        self.decay = decay
        super().__init__(*args, **kwargs)
    
    def get_epsilon(self):
        return self.epsilon * (self.decay / (self.counts.sum() + self.decay))
greedy = EpsilonGreedyBandit(ads, epsilon=0.2, random_state=SEED)
simulate_bandit(greedy, ads, df)
first = EpsilonFirstBandit(ads, epsilon=0.2, trials=1000, random_state=SEED)
simulate_bandit(first, ads, df)
decay = EpsilonDecayBandit(ads, epsilon=0.2, decay=5, random_state=SEED)
simulate_bandit(decay, ads, df)
