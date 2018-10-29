from core.models.bayes import Bayes
from core.models.logistic import Logistic
import time

def train(model):
    print("Training: " + model.__class__.__name__)
    start = time.time()
    model.train("./resources/imdb_train.txt")
    end1 = time.time()
    print("train time: " + str(end1 - start))
    model.evaluateAccuracy("./resources/imdb_validation.txt")
    end2 = time.time()
    print("eval time: " + str(end2 - end1))
    model.save()

models = [Logistic(), Bayes()]
for model in models:
    train(model)
