import csv
import codecs
from random import shuffle

def processTwitterData():
    data = []
    with codecs.open('./resources/trainingandtestdata/training.1600000.processed.noemoticon.csv', "r", encoding='utf-8', errors='ignore') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            label = "POS" if row[0] == "4" else "NEG"
            s = row[5]
            s = s + "@@@@" + label + "\n"
            data.append(s)

    shuffle(data)
    n_sample = len(data)
    split = int(n_sample * 0.7)
    train_set = data[:split]
    validation_set = data[split::]

    with open('./resources/twitter_train.txt', 'w') as train:
        for line in train_set:
            train.write(line)

    with open('./resources/twitter_validation.txt', 'w') as test:
        for line in validation_set:
            test.write(line)

def mergeData():
    with open('./resources/twitter_train.txt', 'r') as twitter_train_file:
        twitter_train = twitter_train_file.readlines()
    
    with open('./resources/twitter_validation.txt', 'r') as twitter_validation_file:
        twitter_validation = twitter_validation_file.readlines()

    with open('./resources/imdb_train.txt', 'r') as imdb_train_file:
        imdb_train = imdb_train_file.readlines()
    
    with open('./resources/imdb_validation.txt', 'r') as imdb_validation_file:
        imdb_validation = imdb_validation_file.readlines()
    
    train_set = twitter_train + imdb_train
    validation_set = twitter_validation + imdb_validation
    shuffle(train_set)
    shuffle(validation_set)

    with open('./resources/train_set', 'w') as train_set_file:
        for line in train_set:
            train_set_file.write(line)

    with open('./resources/validation_set', 'w') as validation_set_file:
        for line in validation_set:
            validation_set_file.write(line)

mergeData()
