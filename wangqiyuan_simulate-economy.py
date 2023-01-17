import math

import random

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes



num_agents = 100

num_raster = 10

num_wealth = 1000

seller_loose = 0.20

buyer_loose = 0.20

rounds = 1000

count_rounds = 0



agents = np.full(num_agents, num_wealth)

# uncomment next line, to simulate a wealthy casino playing against guests

# agents[0] = num_agents*num_wealth*10

# collect the number of rich agent's per round

rich_agents_per_round = list()

# collec the wallet size of all agent's per round

wallet_agent_per_round = list()



# ensure always the same "random" numbers to compare different parameter

np.random.seed(3432432432)
def gamble(buyer_wealth, seller_wealth) -> int:

    """ form perspective buyer: positive means buyer wins, negative buyer looses"""

    if buyer_wealth >= seller_wealth:

        budget = seller_wealth

    else:

        budget = buyer_wealth

    if np.random.randint(0, 1 + 1) == 1:

        # seller wins

        return -int(budget * buyer_loose)

    else:

        # buyer wins

        return int(budget * seller_loose)
def draw_agents():

    global count_rounds, agents

    # print("Agent wallets:",agents)

    fig = plt.figure(figsize=(15, 10))



    # draw the agent world as matrix

    fig.subplots_adjust(wspace=0.25)

    ax1 = fig.add_subplot(1, 2, 1)

    axins1 = inset_axes(ax1,

                        width="5%",  # width = 50% of parent_bbox width

                        height="100%",  # height : 5%

                        loc='lower right',

                        bbox_to_anchor=(0.18, 0., 1, 1),

                        bbox_transform=ax1.transAxes,

                        borderpad=0, )

    im = ax1.matshow(np.resize(agents, (num_raster, num_raster)), interpolation='nearest', vmin=0, vmax=10000)

    plt.colorbar(im, cax=axins1)

    axins1.yaxis.tick_left() # draw ticks of colorbar left side

    ax1.set_title("Wealth round " + str(count_rounds))



    # draw the histogram

    ax2 = fig.add_subplot(1, 2, 2)

    bins = np.arange(0, 10000, 100)

    ax2.set_ylim(0, 10000)

    ax2.set_ylabel('Wallet size')

    ax2.set_xlim(0, num_agents)

    ax2.set_xlabel('Number of agents with same wallet size')

    ax2.yaxis.set_label_position("left")

    ax2.tick_params(axis='y', direction='in', length=6, width=2, colors='r',

                   grid_color='r', grid_alpha=0.5)

    #ax2.yaxis.tick_left()

    ax2.set_yticklabels([]) # hide tick labels

    ax2.hist(agents, bins=bins, orientation='horizontal')

    # resize axes

    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]

    ax2.set_aspect(asp)

    plt.show()
def calc_round(rounds: int):

    global count_rounds, agents

    for round in range(rounds):

        # print("Round %i --------"%round)

        #print(".", end="")

        count_rounds += 1

        round_partners = np.arange(num_agents)

        np.random.shuffle(round_partners)

        for buyer in range(num_agents):

            buyer_wealth = agents[buyer]

            seller = round_partners[buyer]

            seller_wealth = agents[seller]

            if buyer != seller:

                if (buyer_wealth >= 1) & (seller_wealth >= 1):

                    delta = gamble(buyer_wealth, seller_wealth)

                    agents[buyer] += delta

                    agents[seller] -= delta

                    #print("B:%i(%i) <- S: %i(%i): %f" % (buyer, buyer_wealth, seller, seller_wealth, delta))

                else:

                    #print("%i skipped" % (buyer))

                    pass

        # log the rich agents per round

        rich_agents_per_round.append((agents >= num_wealth).sum())

        wallet_agent_per_round.append(agents.tolist())

    print("Wealth sum:", agents.sum())

    print("Wealthier>=1000",(agents >=num_wealth).sum())

    #print(agents)

    draw_agents()
# initial world, were everyone has the same wallet size of num_wallet

calc_round(0)
# after the first round, there are already loosers and winners

calc_round(1)
calc_round(1)
# now let's speed up

calc_round(5)
calc_round(13)
calc_round(80)
calc_round(900)
calc_round(1000)
"""

Show the history of rich wallets count

"""

#print(rich_agents_per_round)

#print(wallet_agent_per_round)

#print(agents)

rich_agents_per_round_ = np.asarray(rich_agents_per_round)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(1,1,1)

cax = ax.plot(np.arange(rich_agents_per_round_.size),rich_agents_per_round_)

ax.set_ylim(0,num_agents/2)

ax.set_title("Wealth per round: ")

plt.show()
"""

Show the history of all wallets

"""

wallet_agent_per_round_ = np.asarray(wallet_agent_per_round).transpose()

#print(wallet_agent_per_round_)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(1,1,1)

xaxis = np.arange(wallet_agent_per_round_[0].size)

for i in range(num_agents):

    cax = ax.plot(xaxis,wallet_agent_per_round_[i])

#ax.set_ylim(0,num_agents/2)

ax.set_title("Wealth per round: ")

plt.show()