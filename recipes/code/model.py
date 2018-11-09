from data import make_dataset

train, test = make_dataset('../data/train.json', '../data/test.json')

print(train)
print(test)
