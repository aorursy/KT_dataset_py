all_series_lines=pd.read_json("../input/all_series_lines.json")
episodes=all_series_lines['TNG'].keys()
total_lines_counts={}
line_counts_by_episode={}
for i,ep in enumerate(episodes):
    episode="episode "+str(i)
    line_counts_by_episode[episode]={}
    if all_series_lines['TNG'][ep] is not np.NaN:
        for member in list(all_series_lines['TNG'][ep].keys()):
            line_counts_by_episode[episode][member]=len(all_series_lines['TNG'][ep][member])
            if member in total_lines_counts.keys():
                total_lines_counts[member]=total_lines_counts[member]+len(all_series_lines['TNG'][ep][member])
            else:
                total_lines_counts[member]=len(all_series_lines['TNG'][ep][member])
TNG_df=pd.DataFrame(list(total_lines_counts.items()), columns=['Character','No. of Lines'])
Top20=TNG_df.sort_values(by='No. of Lines', ascending=False).head(20)

Top20.plot.bar(x='Character',y='No. of Lines')
plt.show()
