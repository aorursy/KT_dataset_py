import os
import re
from pathlib import Path
path = Path("../input")

import numpy as np
import pandas as pd 
text_of_2_games = """Game started at: 2016/12/30 12:54:25
Game ID: 808941103 0.50/1 (PRR) Karkadann (Short) (Hold'em)
Seat 9 is the button
Seat 1: Maria_Pia (40).
Seat 2: RunnerLucker (58.13).
Seat 3: Pimpika (100.60).
Seat 4: twoakers (52.88).
Seat 5: _robbyJ (55.50).
Seat 6: IlxxxlI (53.50).
Seat 7: Ice Bank Mice Elf (76.75).
Seat 8: gust (49.31).
Seat 9: AironVega (56.76).
Player Maria_Pia has small blind (0.50)
Player RunnerLucker has big blind (1)
Player Maria_Pia received a card.
Player Maria_Pia received a card.
Player RunnerLucker received a card.
Player RunnerLucker received a card.
Player Pimpika received a card.
Player Pimpika received a card.
Player twoakers received a card.
Player twoakers received a card.
Player _robbyJ received a card.
Player _robbyJ received a card.
Player IlxxxlI received card: [Jd]
Player IlxxxlI received card: [Jc]
Player Ice Bank Mice Elf received a card.
Player Ice Bank Mice Elf received a card.
Player gust received a card.
Player gust received a card.
Player AironVega received a card.
Player AironVega received a card.
Player Pimpika folds
Player twoakers folds
Player _robbyJ calls (1)
Player IlxxxlI raises (4)
Player Ice Bank Mice Elf folds
Player gust folds
Player AironVega folds
Player Maria_Pia raises (10.50)
Player RunnerLucker folds
Player _robbyJ folds
Player IlxxxlI folds
Uncalled bet (7) returned to Maria_Pia
Player Maria_Pia mucks cards
------ Summary ------
Pot: 10. Rake 0
*Player Maria_Pia mucks (does not show cards). Bets: 4. Collects: 10. Wins: 6.
Player RunnerLucker does not show cards.Bets: 1. Collects: 0. Loses: 1.
Player Pimpika does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player twoakers does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player _robbyJ does not show cards.Bets: 1. Collects: 0. Loses: 1.
Player IlxxxlI does not show cards.Bets: 4. Collects: 0. Loses: 4.
Player Ice Bank Mice Elf does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player gust does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player AironVega does not show cards.Bets: 0. Collects: 0. Wins: 0.
Game ended at: 2016/12/30 12:55:45

Game started at: 2016/12/30 13:0:3
Game ID: 808943744 0.50/1 (PRR) Karkadann (Short) (Hold'em)
Seat 3 is the button
Seat 1: Maria_Pia (48.28).
Seat 2: RunnerLucker (56.63).
Seat 3: Pimpika (40).
Seat 4: twoakers (51.88).
Seat 5: _robbyJ (54.50).
Seat 6: IlxxxlI (49.50).
Seat 7: Ice Bank Mice Elf (151.50).
Seat 8: gust (49.31).
Seat 9: AironVega (56.76).
Player twoakers has small blind (0.50)
Player _robbyJ has big blind (1)
Player twoakers received a card.
Player twoakers received a card.
Player _robbyJ received a card.
Player _robbyJ received a card.
Player IlxxxlI received card: [9c]
Player IlxxxlI received card: [Ac]
Player Ice Bank Mice Elf received a card.
Player Ice Bank Mice Elf received a card.
Player gust received a card.
Player gust received a card.
Player AironVega received a card.
Player AironVega received a card.
Player Maria_Pia received a card.
Player Maria_Pia received a card.
Player RunnerLucker received a card.
Player RunnerLucker received a card.
Player Pimpika received a card.
Player Pimpika received a card.
Player IlxxxlI raises (3)
Player Ice Bank Mice Elf folds
Player gust folds
Player AironVega folds
Player Maria_Pia calls (3)
Player RunnerLucker folds
Player Pimpika folds
Player twoakers folds
Player _robbyJ calls (2)
*** FLOP ***: [4h 8s 2s]
Player _robbyJ checks
Player IlxxxlI checks
Player Maria_Pia bets (3.20)
Player _robbyJ folds
Player IlxxxlI raises (12)
Player Maria_Pia folds
Uncalled bet (8.80) returned to IlxxxlI
Player IlxxxlI mucks cards
------ Summary ------
Pot: 15.11. Rake 0.55. JP fee 0.24
Board: [4h 8s 2s]
Player Maria_Pia does not show cards.Bets: 6.20. Collects: 0. Loses: 6.20.
Player RunnerLucker does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player Pimpika does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player twoakers does not show cards.Bets: 0.50. Collects: 0. Loses: 0.50.
Player _robbyJ does not show cards.Bets: 3. Collects: 0. Loses: 3.
*Player IlxxxlI mucks (does not show cards). Bets: 6.20. Collects: 15.11. Wins: 8.91.
Player Ice Bank Mice Elf does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player gust does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player AironVega does not show cards.Bets: 0. Collects: 0. Wins: 0.
Game ended at: 2016/12/30 13:1:41
"""

a_game = """Game started at: 2016/12/30 12:54:25
Game ID: 808941103 0.50/1 (PRR) Karkadann (Short) (Hold'em)
Seat 9 is the button
Seat 1: Maria_Pia (40).
Seat 2: RunnerLucker (58.13).
Seat 3: Pimpika (100.60).
Seat 4: twoakers (52.88).
Seat 5: _robbyJ (55.50).
Seat 6: IlxxxlI (53.50).
Seat 7: Ice Bank Mice Elf (76.75).
Seat 8: gust (49.31).
Seat 9: AironVega (56.76).
Player Maria_Pia has small blind (0.50)
Player RunnerLucker has big blind (1)
Player Maria_Pia received a card.
Player Maria_Pia received a card.
Player RunnerLucker received a card.
Player RunnerLucker received a card.
Player Pimpika received a card.
Player Pimpika received a card.
Player twoakers received a card.
Player twoakers received a card.
Player _robbyJ received a card.
Player _robbyJ received a card.
Player IlxxxlI received card: [Jd]
Player IlxxxlI received card: [Jc]
Player Ice Bank Mice Elf received a card.
Player Ice Bank Mice Elf received a card.
Player gust received a card.
Player gust received a card.
Player AironVega received a card.
Player AironVega received a card.
Player Pimpika folds
Player twoakers folds
Player _robbyJ calls (1)
Player IlxxxlI raises (4)
Player Ice Bank Mice Elf folds
Player gust folds
Player AironVega folds
Player Maria_Pia raises (10.50)
Player RunnerLucker folds
Player _robbyJ folds
Player IlxxxlI folds
Uncalled bet (7) returned to Maria_Pia
Player Maria_Pia mucks cards
------ Summary ------
Pot: 10. Rake 0
*Player Maria_Pia mucks (does not show cards). Bets: 4. Collects: 10. Wins: 6.
Player RunnerLucker does not show cards.Bets: 1. Collects: 0. Loses: 1.
Player Pimpika does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player twoakers does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player _robbyJ does not show cards.Bets: 1. Collects: 0. Loses: 1.
Player IlxxxlI does not show cards.Bets: 4. Collects: 0. Loses: 4.
Player Ice Bank Mice Elf does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player gust does not show cards.Bets: 0. Collects: 0. Wins: 0.
Player AironVega does not show cards.Bets: 0. Collects: 0. Wins: 0.
Game ended at: 2016/12/30 12:55:45"""
total = 0
for i, file in enumerate(path.glob('*.txt')):
    with open(file) as f:
        ngame = 0
        for line in f:
            if 'Game ended at' in line:
                ngame += 1
        print(f'file "{file.name}" has {ngame} games')
        total += ngame
print(f'altogether, there are {total} games')
# isolate games
def getgames(file):
    for match in re.finditer(r'Game started at.*?Game ended at.*?\n\n', file,re.MULTILINE + re.DOTALL):
        yield match.group(0)
# next().group(0)

# isolate into part
def separate(game):
    match = re.match(r'Game started at: (.*)Game ID: (.*?)\n(.*)(Player.*)------ Summary ------(.*)Game ended at: (.*)', game, re.MULTILINE + re.DOTALL)
    try:
        start, gameid, playerstartmoney, actions, summary, end = match.groups()
        return start.strip(), gameid.strip(), playerstartmoney, actions, summary, end.strip()
    except AttributeError:
        return game, "", "", "", "", ""

separate(a_game)

files_games = []
for i, file in enumerate(path.glob('*.txt')):
    with open(file) as f:
        games = getgames(''.join(f.readlines()))
        file_games = []
        for game in games:
            parts = separate(game)
            parts = list(parts)
            parts.append(f.name)
            file_games.append(parts)
    files_games.extend(file_games)
len(files_games)
df = (pd.DataFrame(files_games, columns=['Start', 'ID', 'Money', 'Actions', 'Summary', 'End', 'File'])
        .assign(NPlayer=lambda df:df.Money.str.count('Seat')+1)
     )
df
df.NPlayer.value_counts()
