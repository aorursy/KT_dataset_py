import csv

map = {}
with open('../input/cliente_tabla.csv', 'r', encoding='utf-8') as client_name:
    infile = csv.reader(client_name)
    for a in infile:
        b = " ".join(a[1].split())
        if b in map.keys():
            map[b] += 1
        else:
            map[b] = 1

# There are 281670 "NO IDENTIFICADO" client names
# Removing from the dictionary
map.pop('NO IDENTIFICADO')

print("Total number of unique client names:", len(map))

for x in map:
    if map[x] > 600:
        print(x,'-', map[x])
