from sklearn.naive_bayes import MultinomialNB
from ..modelbase import ModelBase

class Bayes(ModelBase):
    VECTOR_FILE = "./models/bayes_vectorizer.sav"
    TFIDF_FILE = "./models/bayes_tfidf.sav"
    MODEL_FILE = "./models/bayes_model.sav"

    def __init__(self):
        ModelBase.__init__(self, Bayes.VECTOR_FILE, Bayes.TFIDF_FILE, Bayes.MODEL_FILE)
        self.model = MultinomialNB()
