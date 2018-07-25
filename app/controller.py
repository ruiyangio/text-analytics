from core.models.bayes import Bayes
from core.models.logistic import Logistic

model = Logistic()
model.load()

def get_sentiment(text):
    pred_res, proba = model.predict(text)
    label = "Positive" if pred_res else "Negative"
    return { "sentiment": label, "probability": proba }