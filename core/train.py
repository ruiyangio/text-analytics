from bayes import Bayes

model = Bayes()
model.train("../resources/small_set.txt")
model.evaulate("../resources/small_set.txt")
