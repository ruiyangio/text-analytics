from core.bayes import Bayes

model = Bayes.Bayes()
model.train("./resources/train_set.txt")
print(model.evaluateAccuracy("./resources/validation_set.txt"))
