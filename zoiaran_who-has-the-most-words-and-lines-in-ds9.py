import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
all_series_lines=pd.read_json("../input/all_series_lines.json")
episodes=all_series_lines['DS9'].keys()
total_word_counts={}
total_line_counts={}
for i,ep in enumerate(episodes):
    episode="episode "+str(i)
    if all_series_lines['DS9'][ep] is not np.NaN:
        for member in list(all_series_lines['DS9'][ep].keys()):
            total_words_by_member_in_ep = sum([len(line.split()) for line in all_series_lines['DS9'][ep][member]])
            total_lines_by_member_in_ep = len(all_series_lines['DS9'][ep][member])
            if member in total_word_counts.keys():
                total_word_counts[member]=total_word_counts[member]+total_words_by_member_in_ep
                total_line_counts[member]=total_line_counts[member]+total_lines_by_member_in_ep
            else:
                total_word_counts[member]=total_words_by_member_in_ep
                total_line_counts[member]=total_lines_by_member_in_ep
words_df=pd.DataFrame(list(total_word_counts.items()), columns=['Character','No. of Words'])
most_words=words_df.sort_values(by='No. of Words', ascending=False).head(25)

most_words.plot.bar(x='Character',y='No. of Words')
plt.show()
lines_df=pd.DataFrame(list(total_line_counts.items()), columns=['Character','No. of Lines'])
most_lines=lines_df.sort_values(by='No. of Lines', ascending=False).head(25)

most_lines.plot.bar(x='Character',y='No. of Lines')
plt.show()