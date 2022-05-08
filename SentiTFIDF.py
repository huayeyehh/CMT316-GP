import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

class SentiTFIDF(object):
    def __init__(self):
        self.model = sklearn.svm.SVC(kernel="linear", gamma='auto')
        self.vector = TfidfVectorizer()
        print("Senti TF-IDF Created.")

    def train(self, train_set_x, train_set_y):
        # Learn vocabulary and idf from training set
        self.vector.fit(train_set_x["text"])
        # Transform train and test input documents to document-term matrix
        tfidf_train_x = self.vector.transform(train_set_x["text"])
        # Train the classifier
        self.model.fit(tfidf_train_x, train_set_y.iloc[:,-1].to_numpy())
        print("Senti TF-IDF Model Trained.")

    def test(self, val_set_x, val_set_y):
        tfidf_val_x = self.vector.transform(val_set_x["text"])
        predictions = self.model.predict(tfidf_val_x)
        tfidf_val_y = val_set_y.to_numpy()
        print(str(precision_score(tfidf_val_y, predictions, average='macro')))
        print(str(recall_score(tfidf_val_y, predictions, average='macro')))
        print(str(f1_score(tfidf_val_y, predictions, average='macro')))
        print(str(accuracy_score(tfidf_val_y, predictions)))
