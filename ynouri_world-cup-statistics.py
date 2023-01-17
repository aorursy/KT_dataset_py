import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/results.csv")
df = df.sort_values(axis=0, by=['date'], ascending=False)  # Most recent games first
df['total_score'] = df.home_score + df.away_score  # Total score is used to compute average score by game
df['diff_score'] = abs(df.home_score - df.away_score)
df.head(5)  # For a quick look to the dataset.
def score_ordered(df):
    max_score = max(df.home_score, df.away_score)
    min_score = min(df.home_score, df.away_score)
    return "{}-{}".format(max_score, min_score)

df['score_ordered'] = df.apply(score_ordered, axis=1)  # Apply score ordering row-wise
def plot_bar_annotated(x, y, title):
    f, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlabel=title)
    ax.grid(False);
    ax.set_xlim((0, 3.2))
    ax.barh(x, y)
    for (p, y_i) in zip(ax.patches, y):
        ax.text(p.get_x() + p.get_width() + 0.02,
                p.get_y() + p.get_height() / 2,
                "{:0.2f}".format(y_i),
                va='center') 
def average_number_of_goals_for_year(year):
    x_world_cup = (df.tournament == 'FIFA World Cup') & df.date.str.contains(year)
    return df[x_world_cup].total_score.mean()

world_cup_years = [str(1978 + 4 * i) for i in range(0, 10)]  # 11 last world cups spanning 50 years of football
average_number_of_goals = [average_number_of_goals_for_year(year) for year in world_cup_years]

plot_bar_annotated(world_cup_years, average_number_of_goals, "Average number of goals per game")
def average_difference_of_goals_for_year(year):
    x_world_cup = (df.tournament == 'FIFA World Cup') & df.date.str.contains(year)
    return df[x_world_cup].diff_score.mean()

average_difference_of_goals = [average_difference_of_goals_for_year(year) for year in world_cup_years]

plot_bar_annotated(world_cup_years, average_difference_of_goals, "Average difference of goals per game")
x_wc = (df.tournament == 'FIFA World Cup')
def x_year(year): return df.date.str.contains(str(year)) 
x_sample = x_wc & (x_year(1998) | x_year(2002) | x_year(2006) | x_year(2010) | x_year(2014))
scores_frequencies = df[x_sample].score_ordered.value_counts(normalize=True).sort_values(ascending=False)
x = scores_frequencies.index
x_ticks = np.arange(len(x))
y = scores_frequencies.values
f, ax = plt.subplots(figsize=(15,5))
ax.bar(x_ticks, y, width=0.5)
ax.set_xticklabels(x)
ax.set_xticks(x_ticks, minor=False)
plt.title("Frequency of scores during the 5 last world cups");