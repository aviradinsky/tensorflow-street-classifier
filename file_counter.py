from os import listdir

train_dir = './data/train'
test_dir = './data/test'

for dir in listdir(train_dir):
    print(dir,len(listdir(f'{train_dir}/{dir}')))