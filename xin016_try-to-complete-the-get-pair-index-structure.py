import pandas as pd, numpy as np
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
def get_pair_index_structure(structure):

    structure = np.array([struc for struc in structure], dtype="<U4")

    open_index = np.where(structure == "(")[0]

    closed_index = np.where(structure == ")")[0]

    #print(open_index)

    #print(closed_index)

    open_index2=open_index

    

    for ind in range(0,len(open_index)):

        structure[max(open_index2[open_index2<closed_index[ind]])] = closed_index[ind]

        structure[closed_index[ind]] = max(open_index2[open_index2<closed_index[ind]])

        open_index2=np.delete(open_index2, open_index2.tolist().index(max(open_index2[open_index2<closed_index[ind]])))

        

    structure[structure == "."] = -1

    structure = structure.astype(int) 

    #print(structure) 

    return structure





def get_pair_structure(data):

    for ind in range(0,len(data.sequence)):

        seq = data.sequence[ind]

        seq_map = np.array([struc for struc in seq], dtype="<U4")

        stru_map = get_pair_index_structure(data.structure[ind])

        #print(ind)

        

        seq_map[stru_map != -1] = seq_map[stru_map[stru_map != -1].astype(int)]                

        seq_map[stru_map == -1] = 'N'

        #print(seq_map)

        str_structure=''

        for s in seq_map:

            #print(s)

            str_structure = str_structure+s

        #print(str_structure)

        data["pair_structure"][ind]=str_structure

        #print(data)

    return data
train["pair_structure"] = ''

train=get_pair_structure(train)
train.loc[:, ['sequence', 'structure', 'predicted_loop_type','pair_structure']]