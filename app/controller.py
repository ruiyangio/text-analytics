from core.bayes import Bayes

bayesModel = Bayes.Bayes()
bayesModel.initialze()

def get_sentiment(text):
    pred_res, proba = bayesModel.predict(text)
    label = "Positive" if pred_res else "Negative"
    return { "sentiment": label, "probability": proba }