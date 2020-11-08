import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from common.util import Util

class FeatureExtractor(object):
    LABEL_SEPRATOR = "@@@@"
    LABEL_POSITIVE = "POS"
    LABEL_NEGATIVE = "NEG"

    @staticmethod
    def train_test_split(x, y, split = 0.8):
        return train_test_split(x, y, test_size = split, random_state = 12)

    def __init__(self, vector_save_path, tfidf_save_path, ngram_range=(1,2)):
        self.vectorizer = CountVectorizer(min_df=2, tokenizer=Util.default_tokenize, ngram_range=ngram_range)
        self.tfidf_transformer = TfidfTransformer()
        self.vector_save_path = vector_save_path
        self.tfidf_save_path = tfidf_save_path
    
    @staticmethod
    def process_data(data_content):
        target = []
        for i, line in enumerate(data_content):
            parts = line.split(FeatureExtractor.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1].rstrip()
            if label == FeatureExtractor.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            data_content[i] = s
        return (data_content, target)
    
    @staticmethod
    def read_and_process(data_path):
        with open(data_path, "r") as data_file:
            data_content = data_file.readlines()
        return FeatureExtractor.process_data(data_content)

    def fit_transform(self, data_path, fit=True):
        data_content, target = FeatureExtractor.read_and_process(data_path)
        count_feature = self.vectorizer.fit_transform(data_content) if fit else self.vectorizer.transform(data_content)
        tfidf = self.tfidf_transformer.fit_transform(count_feature) if fit else self.tfidf_transformer.transform(count_feature)

        return (tfidf, np.array(target))

    def transform(self, data_path):
        return self.fit_transform(data_path, False)

    def transform_text(self, text):
        count_feature = self.vectorizer.transform([text])
        tfidf = self.tfidf_transformer.transform(count_feature)
        return tfidf

    def load(self):
        self.vectorizer = joblib.load(self.vector_save_path)
        self.tfidf_transformer = joblib.load(self.tfidf_save_path)

    def save(self, vector_save_path, tfidf_save_path):
        joblib.dump(self.vectorizer, vector_save_path)
        joblib.dump(self.tfidf_transformer, tfidf_save_path)
