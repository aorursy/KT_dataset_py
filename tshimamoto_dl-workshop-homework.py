%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import pickle



def histgram_act_dist(dists):

    plt.style.use("default")

    layer_num = len(dists.keys())

    fig = plt.figure(figsize=(20,2))

    fig.suptitle("Activation Distributions", y=1.15, fontsize=18)

    ax = fig.subplots(1, layer_num)

    for i, dist in enumerate(dists):

        key = "layer" + str(i)

        ax[i].hist(dists[key], bins=5)

        ax[i].set_title(key)

        ax[i].set_xlim(0.0, 2.0)

        ax[i].set_xlabel("Value after Affine-Transformation")

        if i == 0:

            ax[i].set_ylabel("Count")

        ax[i].set_ylim(0, 12)

    plt.show()



def bar_init_weight_effect(df):

    plt.style.use("default")

    df = df.drop(["init_weight", "v_avg_accuracy", "v_argmax", "memo"], axis=1)

    ax = df.plot.bar(x="act_func", y=["v_avg_loss", "v_max_accuracy"], figsize=(8,3))

    ax.set_title("Effect of Initial Weight Change")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    arrwprp_down = dict(arrowstyle="<-", color="deepskyblue", shrinkA=5, shrinkB=5, patchA=None, patchB=None, connectionstyle="arc3,rad=0.1")

    arrwprp_up = dict(arrowstyle="<-", color="deeppink", shrinkA=5, shrinkB=5, patchA=None, patchB=None, connectionstyle="arc3,rad=-0.1")

    for i in range(3):

        pos1 = 2*i

        pos2 = 2*i+1

        ax.annotate("Down!", color="deepskyblue",

            xy=(pos1-0.15, df.v_avg_loss[pos1]), xycoords='data',

            xytext=(pos2-0.3, df.v_avg_loss[pos2]), textcoords='data',

            arrowprops=arrwprp_down

        )

        ax.annotate("Up!",color="deeppink",

            xy=(pos1+0.1, df.v_max_accuracy[pos1]-0.01), xycoords='data',

            xytext=(pos2+0.05, df.v_max_accuracy[pos2]+0.01), textcoords='data',

            arrowprops=arrwprp_up

        )



    plt.show()



def plot_BN_effect(df_dict):

    plt.style.use("default")

    fig = plt.figure(figsize=(11, 3.5))

    fig.suptitle("Effect of Batch Normalization", y=1.05, fontsize=18)

    ax = fig.subplots(1, len(df_dict))

    for i, (key, val) in enumerate(df_dict.items()):

        ax1 = ax[i]



        accuracy_color = "darkorange"

        ax1.plot(val.epoch, val.l_accuracy, label="l_accu: accuracy on learning data", linestyle="dashed", color=accuracy_color)

        ax1.plot(val.epoch, val.v_accuracy, label="v_accu: accuracy on validation data", linestyle="solid", color=accuracy_color)

        ax1.set_xlabel("epoch")

        ax1.set_xlim(-3, 103)

        ax1.set_ylabel("accuracy", color=accuracy_color, horizontalalignment="left")

        ax1.set_ylim(0.84, 1.03)

        ax1.set_yticks([0.95,0.975,1.0])

        ax1.tick_params(axis="y", colors=accuracy_color)

        ax1.set_title(key)

        if i == len(df_dict)-1:

            ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

        ax1.grid(which='major',color='lightgray',linestyle='--')

        ax1.grid(which='minor',color='lightgray',linestyle='--')

        epochidx = 16

        ax1.annotate("l_accu",

                     xy=(val.epoch[epochidx], val.l_accuracy[epochidx]),

                     xytext=(val.epoch[epochidx], val.l_accuracy[epochidx]+0.014),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color="olive"),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=accuracy_color, alpha=0.1)

                    )

        ax1.annotate("v_accu",

                     xy=(val.epoch[epochidx], val.v_accuracy[epochidx]),

                     xytext=(val.epoch[epochidx], val.v_accuracy[epochidx]-0.021),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color="olive"),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=accuracy_color, alpha=0.1)

                    )



        ax2 = ax1.twinx()

        loss_color = "royalblue"

        ax2.plot(val.epoch, val.l_loss, label="l_loss: loss on learning data", linestyle="dashed", color=loss_color)

        ax2.plot(val.epoch, val.v_loss, label="v_loss: loss on validation data", linestyle="solid", color=loss_color)

        ax2.set_ylabel("loss", color=loss_color, horizontalalignment="right")

        ax2.set_ylim(-0.08, 0.5)

        ax2.set_yticks([0.0,0.05,0.1,0.15])

        ax2.tick_params(axis="y", colors=loss_color)

        if i == len(df_dict)-1:

            ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 0.8))

        ax2.grid(which='major',color='lightgray',linestyle='--')

        ax2.grid(which='minor',color='lightgray',linestyle='--')

        epochidx = 78

        ax2.annotate("l_loss",

                     xy=(val.epoch[epochidx], val.l_loss[epochidx]),

                     xytext=(val.epoch[epochidx], val.l_loss[epochidx]-0.06),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=loss_color),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=loss_color, alpha=0.1)

                    )

        ax2.annotate("v_loss",

                     xy=(val.epoch[epochidx], val.v_loss[epochidx]),

                     xytext=(val.epoch[epochidx], val.v_loss[epochidx]+0.04),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=loss_color),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=loss_color, alpha=0.1)

                    )



    plt.subplots_adjust(wspace=1, hspace=0)

    plt.tight_layout()

    plt.show()

#from graphutils import bar_init_weight_effect

df = pd.read_csv("../input/init_weight_effect.csv")

bar_init_weight_effect(df)
#from graphutils import plot_BN_effect

df_dict={}

df_dict["Minibatch-ReLU-AdaGrad"] = pd.read_csv("../input/Ex17.csv")

df_dict["Minibatch-ReLU-AdaGrad with BatchNormalization"] = pd.read_csv("../input/Ex17_E.csv")

plot_BN_effect(df_dict)
# 性能順モデル一覧（平均Lossの昇順、且つ、最大Accuracyの降順）

df = pd.read_csv("../input/result.csv")

#df = pd.read_csv("./result.csv")

df = df.sort_values(by=["v_avg_loss", "v_max_acc"], ascending=[True, False]).copy().reset_index()

df = df.drop(["index"], axis=1).rename(columns={"No.":"実験No."})

df.style.background_gradient(cmap="autumn", subset=["v_avg_loss","v_max_acc"], low=2.0, high=0.0)

#df.style.format({'v_avg_loss': "{:5.3f}", 'v_max_acc': '{:5.3f}'})