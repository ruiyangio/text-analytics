import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from common.util import Util

class Bayes(object):
    LABEL_SEPRATOR = "@@@@"
    LABEL_POSITIVE = "pos"
    LABEL_NEGATIVE = "neg"
    VECTOR_FILE = "./models/bayes_vectorizer.sav"
    IFIDF_FILE = "./models/bayes_tfidf.sav"
    MODEL_FILE = "./models/bayes_model.sav"

    def __init__(self):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = CountVectorizer(min_df=2, tokenizer=Util.defaultTokenize)
        self.clf = None

    def initialze(self):
        self.vectorizer = joblib.load(self.VECTOR_FILE)
        self.tfidf_transformer = joblib.load(self.IFIDF_FILE)
        self.clf = joblib.load(self.MODEL_FILE)

    def train(self, data_path):
        target = []
        with open(data_path, "r") as train_file:
            train_content = train_file.readlines()

        for i, line in enumerate(train_content):
            line = line.rstrip().lower()
            parts = line.split(self.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1]

            if label == self.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            train_content[i] = s
        train_counts = self.vectorizer.fit_transform(train_content)
        train_tfidf = self.tfidf_transformer.fit_transform(train_counts)
        self.clf = MultinomialNB().fit(train_tfidf, target)
        joblib.dump(self.vectorizer, self.VECTOR_FILE)
        joblib.dump(self.tfidf_transformer, self.IFIDF_FILE)
        joblib.dump(self.clf, self.MODEL_FILE)

    def predict(self, text):
        countVector = self.vectorizer.transform([text])
        tf_idf = self.tfidf_transformer.transform(countVector)
        res = self.clf.predict(tf_idf)
        proba = self.clf.predict_proba(tf_idf)
        proba = proba[0][res]
        return (res, proba)

    def evaluateAccuracy(self, data_path):
        with open(data_path, "r") as test_file:
            test_content = test_file.readlines()

        correct = 0
        total = 0

        for line in test_content:
            total += 1
            line = line.rstrip().lower()
            parts = line.split(self.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1]
            pred_res, proba = self.predict(s)
            pred_label = self.LABEL_POSITIVE if pred_res else self.LABEL_NEGATIVE
            if pred_label == label:
                correct += 1

        return correct / total
