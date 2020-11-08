import numpy as np
import sklearn
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common.util import Util
from sklearn.metrics import confusion_matrix

from core.extractor import FeatureExtractor

class ModelBase(object):
    def __init__(self, vector_save_path, tfidf_save_path, model_save_path, ngram_range=(1,1)):
        self.feature_extractor = FeatureExtractor(vector_save_path, tfidf_save_path, ngram_range)
        self.vector_save_path = vector_save_path
        self.tfidf_save_path = tfidf_save_path
        self.model_save_path = model_save_path
        self.model = None

    def load(self):
        self.feature_extractor.load()
        self.model = joblib.load(self.model_save_path)

    def save(self):
        self.feature_extractor.save(self.vector_save_path, self.tfidf_save_path)
        joblib.dump(self.model, self.model_save_path)

    def train(self, data_path):
        x, y = self.feature_extractor.fit_transform(data_path)
        self.model.fit(x, y)
        self.save()

    def predict(self, text):
        x = self.feature_extractor.transform_text(text)
        y_pred = self.model.predict(x)
        proba = self.model.predict_proba(x)
        proba = proba[0][y_pred]
        return (y_pred, proba)

    def evaluateAccuracy(self, data_path):
        y_pred = []
        data_content, y = FeatureExtractor.read_and_process(data_path)
        for text in data_content:
            pred, pred_proba = self.predict(text)
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        confusion = confusion_matrix(y, y_pred)
        
        print(confusion)
        print(accuracy_score(y, y_pred))
