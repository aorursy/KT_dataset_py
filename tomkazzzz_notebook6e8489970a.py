array = ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'a']
answer = {array[i]: [j for j in range(len(array)) if array[j] == array[i]] for i in range(len(array))}
print(answer)

set1 = {'0', '1', '2', '3', '4'}
set2 = {'1', '3', '4', '5', '6'}
intersection = 0
for i in set1:
        if i in set2:
            intersection += 1
union = len(a) + len(b) - intersection
print(intersection/union)

import csv
import json

with open('../input/sales-data/sales.json') as i:
    content = i.read()
    template = json.loads(content)
with open('./cout.csv', 'w') as f:
    writer = csv.writer(f)
    for item in template:
        for country, sales_data in item['sales_by_country'].items():
            line.append(country)
            for year, amount in sales_data.items():
                writer.writerow([item['item'], country, year, amount])
                print([item['item'], country, year, amount])