!pip install schrutepy

!pip install textgenrnn
from schrutepy import schrutepy



df = schrutepy.load_schrute()
from tqdm import tqdm

df['EpIden'] = 0

for i in tqdm(range(len(df))):

    df.loc[i,'EpIden'] = str(df.loc[i,'season'])+str(df.loc[i,'episode'])
import random

file = open('office_script.txt','a')

for identifier in tqdm(df['EpIden'].unique()):

    if random.randint(1,5) == 1:

        curr_data = df[df['EpIden']==identifier]

        for index in curr_data.index:

            final_string = '['+curr_data.loc[index,'character']+']: '

            final_string += str(curr_data.loc[index,'text'])

            final_string += '\n'

            file.write(final_string)

        file.write('-----------------------\n')

file.close()
!pip install textgenrnn
from textgenrnn import textgenrnn



textgen = textgenrnn()

textgen.train_from_file('office_script.txt', 

                        num_epochs=15)
for i in range(100):

    textgen.generate(10, temperature=1.0)