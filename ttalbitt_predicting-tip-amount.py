import pandas

file = open('../input/restaurant-tips/tips.csv', 'r')
df = pandas.read_csv(file)

def change_sex(s):
    if s == 'Female':
        return 1
    else:
        return 0
def change_smoker(s):
    if s == 'Yes':
        return 1
    else:
        return 0
def change_time(t):
    if t == 'Dinner':
        return 1
    else:
        return 0
def change_day(d):
    values = {'Thur': [1,0,0,0], 'Fri': [0,1,0,0], 'Sat': [0,0,1,0], 'Sun': [0,0,0,1]}
    return values.get(d)
def spread_oh(df):
    thurs = []
    fri = []
    sat = []
    sun = []
    for oh in df['day']:
        if oh[0] == 1:
            thurs.append(1)
            fri.append(0)
            sat.append(0)
            sun.append(0)
        if oh[1] == 1:
            thurs.append(0)
            fri.append(1)
            sat.append(0)
            sun.append(0)
        if oh[2] == 1:
            thurs.append(0)
            fri.append(0)
            sat.append(1)
            sun.append(0)
        if oh[3] == 1:
            thurs.append(0)
            fri.append(0)
            sat.append(0)
            sun.append(1)
    df['thurs'] = thurs
    df['fri'] = fri
    df['sat'] = sat
    df['sun'] = sun
    df = df.drop(['day'], axis=1)
    return df
    

df['sex'] = df['sex'].apply(change_sex)
df['smoker'] = df['smoker'].apply(change_smoker)
df['time'] = df['time'].apply(change_time)
df['day'] = df['day'].apply(change_day)
df = spread_oh(df)

df.head()

greats = []
oks = []
bads = []
scores = []

for i in range(100):
    import random

    indices = [i for i in range(df.shape[0])]
    random.shuffle(indices)
    testers = indices[:40]
    trainers = indices[40:]

    train_data = df.drop(testers)
    test_data = df.drop(trainers)

    train_y = train_data['tip']
    train_x = train_data.drop('tip', axis=1)

    test_y = test_data['tip']
    test_x = test_data.drop('tip', axis=1)
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(train_x, train_y)

    y = reg.predict(test_x)

    differences = y - test_y.to_numpy()
    diff = abs(differences)
    spot_on = 0
    decent = 0
    off = 0
    for d in diff:
        if d < .1:
            spot_on += 1
        elif d < .5:
            decent += 1
        elif d > 2:
            off += 1
#     print('great ' + str(100*spot_on / len(diff)) + ' % of time')
#     print('ok ' + str(100*decent / len(diff)) + ' % of time')
#     print('bad ' + str(100*off / len(diff)) + ' % of time')
    
    greats.append(spot_on/len(diff))
    oks.append(decent/len(diff))
    bads.append(off/len(diff))
    
    scores.append(reg.score(test_x, test_y))

print('great avg rate:', 100*sum(greats)/len(greats))
print('decent avg rate:', 100*sum(oks)/len(oks))
print('bad avg rate:', 100*sum(bads)/len(bads))
print('R2 score avg:', sum(scores)/len(scores))

























































