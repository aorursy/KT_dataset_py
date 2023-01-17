import pandas as pd
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
sorted(train['Stadium'].unique())
len(train['Stadium'].unique())
map_stad = {'Broncos Stadium at Mile High': 'Broncos Stadium At Mile High', 'CenturyField': 'CenturyLink Field', 'CenturyLink': 'CenturyLink Field', 'Everbank Field': 'EverBank Field', 'FirstEnergy': 'First Energy Stadium', 'FirstEnergy Stadium': 'First Energy Stadium', 'FirstEnergyStadium': 'First Energy Stadium', 'Lambeau field': 'Lambeau Field', 'Los Angeles Memorial Coliesum': 'Los Angeles Memorial Coliseum', 'M & T Bank Stadium': 'M&T Bank Stadium', 'M&T Stadium': 'M&T Bank Stadium', 'Mercedes-Benz Dome': 'Mercedes-Benz Superdome', 'MetLife': 'MetLife Stadium', 'Metlife Stadium': 'MetLife Stadium', 'NRG': 'NRG Stadium', 'Oakland Alameda-County Coliseum': 'Oakland-Alameda County Coliseum', 'Paul Brown Stdium': 'Paul Brown Stadium', 'Twickenham': 'Twickenham Stadium'}



for stad in train['Stadium'].unique():

    if stad in map_stad.keys():

        pass

    else:

        map_stad[stad]=stad
train['Stadium'] = train['Stadium'].map(map_stad)
sorted(train['Stadium'].unique())
len(sorted(train['Stadium'].unique()))