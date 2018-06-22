from core.bayes import Bayes

model = Bayes.Bayes()
model.train("./resources/small_set.txt")
#model.evaluateAccuracy("./resources/small_set.txt")
