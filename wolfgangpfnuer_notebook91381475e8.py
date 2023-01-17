import csv

with open ('../input/train.csv') as csv_file:

    train = list(csv.DictReader(csv_file))

with open ('../input/test.csv') as test_file:

    test = list(csv.DictReader(test_file))

with open ('output.csv', 'w') as out_file:

    out_writer = csv.DictWriter(out_file, fieldnames=['PassengerId', 'Survived'])

    out_writer.writeheader()

    for data in test:

        out_writer.writerow({'PassengerId': data['PassengerId'], 'Survived': int(data['Sex'] == 'female')})

        