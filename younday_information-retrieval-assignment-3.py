import pandas as pd

import os



print(os.listdir("../input"))



def read_user_train_data():

    user_training_dict = {}

    user_test_dict = {}

    file = open('../input/matrix_training.csv')

    for line in file:

        line = line.strip()

        line = line.split(' ')

        if line[0] in user_training_dict:

            user_training_dict[line[0]].append(line[1])

        else:

            user_training_dict[line[0]] = [line[1]]

    file.close()

    f = open('../input/matrix_test.csv')

    for line in f:

        line = line.strip()

        line = line.split(' ')

        if line[0] in user_test_dict:

            user_test_dict[line[0]].append(line[1])

        else:

            user_test_dict[line[0]] = [line[1]]

    return (user_training_dict, user_test_dict)



train, test = read_user_train_data()



def read_similarity():

    user_sim_dict = {}

    file = open('../input/user_similarity.csv')

    for line in file:

        line = line.split(' ')

        if line[0] in user_sim_dict:

            user_sim_dict[line[0]].append((line[1], float(line[2].strip())))

        else:

            user_sim_dict[line[0]] = [(line[1], float(line[2].strip()))]

        if line[1] in user_sim_dict:

            user_sim_dict[line[1]].append((line[0], float(line[2].strip())))

        else:

            user_sim_dict[line[1]] = [(line[0], float(line[2].strip()))]

    return user_sim_dict

    

user_sim = read_similarity()



def read_pandas():

    df = pd.read_csv('../input/matrix_training.csv', sep=' ', names=['user_id', 'follower_id'])

    return df



df = read_pandas()

matrix = df.pivot_table(index='user_id', columns='follower_id', aggfunc=len, fill_value=0)

from sklearn.metrics.pairwise import cosine_similarity

p = cosine_similarity(matrix)

matrix = matrix.reset_index()

#print(matrix)

#print(p.shape)

#print(p[6][8], matrix.at[6, 'user_id'], matrix.at[8, 'user_id'])

#print(sorted(train[matrix.at[6, 'user_id']]), sorted(train[matrix.at[8, 'user_id']]))

#set(train[matrix.at[6, 'user_id']]) & set(train[matrix.at[8, 'user_id']])



def get_top_sim_cos(index, row, n=10):

    row = list(row)

    #r = [(item, row.index(item)) for item in row]

    #r = list(filter(lambda a: a[0] > 0.01, r))

    row = sorted(row, reverse=True)

    row = row[1:n]

    return row



def read_similarity_cos(matrix, p):

    user_dict = {}

    # Loop through each user in the matrix and get cosine similarity scores with other users

    l = len(matrix)

    num = 0

    for il in matrix.itertuples():

        index = il[0]

        user_name = il[1]

        #row = il[2:]

        row = [(index, value) for value, index in enumerate(il[2:]) if value != 0]

        print(num, l)

        p_row_l = list(p[index])

        p_row = [(item, p_row_l.index(item)) for item in p_row_l if item > 0.09]

        scores = get_top_sim_cos(index, p_row)

        user_vips = train[user_name]

        vip_dict = {}

        average_list = []

        for item in scores:

            score, i = item

            vips = train[matrix.at[i, 'user_id']]

            diff = set(vips) - set(user_vips)

            for vip in diff:

                if vip in vip_dict:

                    vip_dict[vip].append(score)

                else:

                    vip_dict[vip] = [score]

            for item, values in vip_dict.items():

                average = sum(values)

                average_list.append((item, average))

                sorted_list = sorted(average_list, key=lambda tup: tup[1], reverse=True)

                sorted_list = sorted_list[0:9]

            for item in sorted_list:

                if user_name in user_dict:

                    user_dict[user_name].append((item[0], item[1]))

                else:

                    user_dict[user_name] = [(item[0], item[1])]   

        num = num + 1

    return user_dict

            



#user_dict = read_similarity_cos(matrix, p)
def get_vips_user(user, train):

    return train[user]



def get_most_sim_user(user_sim, train, n=10):

    user_dicts = {}

    for user, items in user_sim.items():

        items.sort(key=lambda tup: tup[1], reverse=True)

        items = items[:(n-1)]

        vip_dict = {}

        average_list = []

        user_vips = get_vips_user(user, train)

        for item in items:

            vips = get_vips_user(item[0], train)

            diff = set(vips) - set(user_vips)

            for vip in diff:

                if vip in vip_dict:

                    vip_dict[vip].append(item[1])

                else:

                    vip_dict[vip] = [item[1]]

        for item, values in vip_dict.items():

            average = sum(values)

            average_list.append((item, average))

            sorted_list = sorted(average_list, key=lambda tup: tup[1], reverse=True)

            sorted_list = sorted_list[0:9]

        for item in sorted_list:

            if user in user_dicts:

                user_dicts[user].append((item[0], item[1]))

            else:

                user_dicts[user] = [(item[0], item[1])]   

    return user_dicts



def get_most_sim_users(matrix, df, n=10):

    user_dicts = {}

    user_list = list(train.keys())

    sim_dict = {}

    l = len(user_list)

    file = open('../input/user_similarity_cos.csv', 'w')

    for item in user_list:

        sim_list = list(matrix[user_list.index(item)])

        print(user_list.index(item), l)

        for i in sim_list:

            if i >= 0.01:

                file.write(item, ',', user_list[sim_list.index(i)], ',', i)

    return sim_dict



recommend_vip_dict = get_most_sim_user(user_sim, train)

#sim = get_most_sim_users(p, df, train)

#print(sim)
def recommend_vip(test, recommend_vip_dict):

    score = 0

    for user, follower in test.items():

        for item in follower:

            users = [item[0] for item in recommend_vip_dict[user]]

            if item in users:

                score = score + 1

                print("{} {}: {}".format(user, item, score))

            else:

                print("{} {}: {}".format(user, item, score))

    print(score)



recommend_vip(test, recommend_vip_dict)

    