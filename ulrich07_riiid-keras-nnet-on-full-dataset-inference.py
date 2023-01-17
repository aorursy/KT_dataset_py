# useful

import numpy as np

import pandas as pd



# neural nets

import tensorflow as tf

import tensorflow.keras.models as M

import tensorflow.keras.layers as L



# custom

import riiideducation
# PIVOT DATAFRAMES

piv1 = pd.read_csv("../input/riiid-fixed-infos/content.csv")

piv2 = pd.read_csv("../input/riiid-fixed-infos/task.csv")

piv3 = pd.read_csv("../input/riiid-fixed-infos/user.csv")



for col, df in zip(["content_sum", "task_container_sum", "user_sum"], [piv1, piv2, piv3]):

    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

#

m1 = piv1["content_sum"].median()

m2 = piv2["task_container_sum"].median()

m3 = piv3["user_sum"].median()





# OTHER CONSTABTS

TARGET = "answered_correctly"

TIME_MEAN = 21000.0

TIME_MIN = 0.0

TIME_MAX = 300000.0

map_prior = {True:1, False:0}
def preprocess(df):

    df = df.merge(piv1, how="left", on="content_id")

    df["content_emb"] = df["content_emb"].fillna(0.5)

    df["content_sum"] = df["content_sum"].fillna(m1)

    

    df = df.merge(piv2, how="left", on="task_container_id")

    df["task_container_emb"] = df["task_container_emb"].fillna(0.5)

    df["task_container_sum"] = df["task_container_sum"].fillna(m2)

    

    df = df.merge(piv3, how="left", on="user_id")

    df["user_emb"] = df["user_emb"].fillna(0.5)

    df["user_sum"] = df["user_sum"].fillna(m3)

    

    df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].fillna(TIME_MEAN)

    df["duration"] = (df["prior_question_elapsed_time"] - TIME_MIN) / (TIME_MAX - TIME_MIN)

    df["prior_answer"] = df["prior_question_had_explanation"].map(map_prior)

    df["prior_answer"] = df["prior_answer"].fillna(0.5)

    #df = df.fillna(-1)

    epsilon = 1e-6

    df["score"] = 2*df["content_emb"]*df["user_emb"] / (df["content_emb"]+ df["user_emb"] + epsilon)

    return df

#=========
def make_ann(n_in):

    inp = L.Input(shape=(n_in,), name="inp")

    d1 = L.Dense(100, activation="relu", name="d1")(inp)

    d2 = L.Dense(100, activation="relu", name="d2")(d1)

    preds = L.Dense(1, activation="sigmoid", name="preds")(d2)

    

    model = M.Model(inp, preds, name="ANN")

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

#===================
FE = ["content_emb","content_sum" ,"task_container_emb", "task_container_sum",

      "user_emb", "user_sum","duration", "prior_answer","score"]
net0 = make_ann(len(FE))

net1 = make_ann(len(FE))

net2 = make_ann(len(FE))

net3 = make_ann(len(FE))

net4 = make_ann(len(FE))

net5 = make_ann(len(FE))

net0.load_weights("../input/riiid-nnet/w0.h5")

net1.load_weights("../input/riiid-nnet/w1.h5")

net2.load_weights("../input/riiid-nnet/w2.h5")

net3.load_weights("../input/riiid-nnet/w3.h5")

net4.load_weights("../input/riiid-nnet/w4.h5")

net5.load_weights("../input/riiid-nnet/w5.h5")
env = riiideducation.make_env()

iter_test = env.iter_test()



for test_df, sample_prediction_df in iter_test:

    test_df = preprocess(test_df)

    x_te = test_df[FE].values

    p0 = net0.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    p1 = net1.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    p2 = net2.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    p3 = net3.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    p4 = net4.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    p5 = net5.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    test_df['answered_correctly'] = (0.16*p0+ 0.17*p1 + 0.17*p2+ 0.17*p3 + 0.17*p4+ 0.16*p5)

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

#
test_df.head()