from sklearn.linear_model import LogisticRegression
from ..modelbase import ModelBase

class Logistic(ModelBase):
    VECTOR_FILE = "./models/logistic_vectorizer.sav"
    TFIDF_FILE = "./models/logistic_tfidf.sav"
    MODEL_FILE = "./models/logistic_model.sav"

    def __init__(self):
        ModelBase.__init__(self, Logistic.VECTOR_FILE, Logistic.TFIDF_FILE, Logistic.MODEL_FILE)
        self.model = LogisticRegression()
