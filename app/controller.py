import random
from core.models.bayes import Bayes
from core.models.logistic import Logistic

logistic_model = Logistic()
logistic_model.load()

bayes_model = Bayes()
bayes_model.load()

def get_sentiment(text):
    model = logistic_model if random.random() >= 0.5 else bayes_model
    pred_res, proba = model.predict(text)
    label = "Positive" if pred_res else "Negative"
    return { "sentiment": label, "probability": proba, "model": model.__class__.__name__  }