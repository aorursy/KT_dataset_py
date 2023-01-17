import pandas as pd
def list_files():
    from subprocess import check_output
    print(check_output(["ls", "../input"]).decode("utf8"))
    
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Sex'] = [1 if is_male else 0 for is_male in train['Sex'] == 'male']
train.describe()
