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
    VECTOR_FILE = "../../models/bayes_vectorizer.sav"
    MODEL_FILE = "../../models/bayes_model.sav"

    def __init__(self):
        self.tfidf_transformer = TfidfTransformer()
        self.vectorizer = None
        self.clf = None

    def initialze(self):
        self.vectorizer = joblib.load(self.VECTOR_FILE)
        self.clf = joblib.load(self.MODEL_FILE)

    def train(self, data_path, alpha=1):
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
        
        self.vectorizer = CountVectorizer(min_df=2, tokenizer=Util.defaultTokenize)
        tfidf_transformer = TfidfTransformer()
        train_counts = self.vectorizer.fit_transform(train_content)
        train_tfidf = tfidf_transformer.fit_transform(train_counts)
        self.clf = MultinomialNB().fit(train_tfidf, target)
        joblib.dump(self.vectorizer, self.VECTOR_FILE)
        joblib.dump(self.clf, self.MODEL_FILE)

    def predict(self, text):
        tokens = Util.defaultTokenize(text)
        countVector = self.vectorizer.fit_transform([tokens])
        tf_idf = self.tfidf_transformer.transform(countVector)
        res = self.LABEL_POSITIVE if self.clf.predict(tf_idf) == 1 else self.LABEL_NEGATIVE
        return res

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
            if self.predict(s) == label:
                correct += 1

        return correct / total
