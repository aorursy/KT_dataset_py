import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/mydataset2/train_data.csv')
df
df.shape
df["word_freq_receive"].value_counts().plot(kind="pie")
df["word_freq_you"].value_counts().plot(kind="pie")
df["word_freq_free"].value_counts().plot(kind="pie")
df["char_freq_!"].value_counts().plot(kind="pie")
tdf = pd.read_csv('../input/mydataset2/test_features.csv')
tdf
tdf.shape
Xdf = df[["word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over","word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business","word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference"]]
Ydf = df.ham

Xtdf = tdf[["word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over","word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business","word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference"]]
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB().fit(Xdf, Ydf)  
NB.fit(Xdf,Ydf)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(NB, Xdf, Ydf, cv=20)
scores

np.mean(scores)
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=20)
kNN.fit(Xdf,Ydf)
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(kNN, Xdf, Ydf, cv=20)
scores2
np.mean(scores2)
predicted = NB.predict(Xtdf)
predicted
predicted2 = kNN.predict(Xtdf)
predicted2
predicted_df = pd.DataFrame(index=tdf.Id,columns=['ham'])
predicted_df['ham'] = predicted
predicted_df
predicted2_df = pd.DataFrame(index=tdf.Id,columns=['ham'])
predicted2_df['ham'] = predicted2
predicted2_df
predicted_df.to_csv('myPredT2_NB.csv')
predicted2_df.to_csv('myPredT2_kNN.csv')