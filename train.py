from core.models.bayes import Bayes
from core.models.logistic import Logistic
import time

model = Logistic()
start = time.time()
model.train("./resources/imdb_train.txt")
end1 = time.time()
print("train time: " + str(end1 - start))
model.evaluateAccuracy("./resources/imdb_validation.txt")
end2 = time.time()
print("eval time: " + str(end2 - end1))
model.save()
