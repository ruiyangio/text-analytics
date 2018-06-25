from core.bayes import Bayes
import time

model = Bayes.Bayes()
start = time.time()
model.train("./resources/train_set.txt")
end1 = time.time()
print(end1 - start)
print(model.evaluateAccuracy("./resources/validation_set.txt"))
end2 = time.time()
print(end2 - end1)
