import numpy as np
np.__version__
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df[:5, :])

print(df.shape)

df[:, 1] = df[:, 1]//365.25

print(df[:5, :])

s=df[1:,1]

f=s[s>50]

print(f.size) #пациенты старше 50 лет 

f = np.mean(df[df[:, 2] == 1, 3]) #женщины

print(f)

m = np.mean(df[df[:, 2] == 2, 3]) #мужчины

print(m)

if m > f:

    print('True')

else:

    print('False')

print(np.min(df[:, 4]))

print(np.max(df[:, 4])) 

i=df[:, 4][df[:, 4]==200]

print(i.size) #количество людей, имеющих максимальный вес








