import operator
import nltk
import sklearn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

lemmatizer = nltk.stem.WordNetLemmatizer()
def get_vector(vocab, text):
    vector = np.zeros(len(vocab))
    words = []
    for sentence in nltk.tokenize.sent_tokenize(text):
        for token in nltk.tokenize.word_tokenize(sentence):
            words.append(lemmatizer.lemmatize(token).lower())
    for i, word in enumerate(vocab):
        if word in words:
            vector[i] = words.count(word)
    return vector

class SentiSVC(object):
    def __init__(self):
        self.model = sklearn.svm.SVC(kernel="linear", gamma='auto')
        self.vocabulary = []
        print("Senti SVC Created.")

    # Feature: n most frequent words of each label class and combining them together
    def train(self, train_set_x, train_set_y, n=100):
        # Data preprocessing
        stopwords = set(nltk.corpus.stopwords.words("english"))
        additional_stopwords = [".", ",", "'s", "``", "''", "'", "n't", "%", "-", "$", "(", ")", ":", ";", "@", "&", "'m", "user", "#", "!", "?", "..."]
        for sw in additional_stopwords: stopwords.add(sw)
        
        # Prepare vocabulary
        for label in ["0", "1", "2"]:
            # get texts with same label
            temp_list = []
            for i in train_set_x.index:
                if train_set_y.loc[i, "label"] == label:
                    temp_list.append(train_set_x.loc[i, "text"])
            
            # get n most frequent words of this label class
            dict_word_freq = {}
            for text in temp_list:
                for sentence in nltk.tokenize.sent_tokenize(text):
                    for token in nltk.tokenize.word_tokenize(sentence):
                        word = lemmatizer.lemmatize(token).lower()
                        if word in stopwords: continue
                        if word in dict_word_freq: dict_word_freq[word] += 1
                        else: dict_word_freq[word] = 1
                        
            # sort and add first n words in sorted list to vocabulary
            sorted_list = sorted(dict_word_freq.items(), key=operator.itemgetter(1), reverse=True)
            if n < len(sorted_list): sorted_list = sorted_list[:n]
            for word, frequency in sorted_list:
                if word not in self.vocabulary: self.vocabulary.append(word)

        # Create training data
        x, y = [], []
        for i in train_set_x.index:
            x.append(get_vector(self.vocabulary, train_set_x.loc[i, "text"]))
            y.append(train_set_y.loc[i, "label"])

        # Init and train model
        self.model.fit(np.asarray(x), np.asarray(y))
        print("Senti SVC Model Trained.")

    def test(self, val_set_x, val_set_y):
        # test with val set
        x, y = [], []
        for i in val_set_x.index:
            x.append(get_vector(self.vocabulary, val_set_x.loc[i, "text"]))
            y.append(val_set_y.loc[i, "label"])
        predictions = self.model.predict(x)
        y = np.asarray(y)
        print(str(precision_score(y, predictions, average='macro')))
        print(str(recall_score(y, predictions, average='macro')))
        print(str(f1_score(y, predictions, average='macro')))
        print(str(accuracy_score(y, predictions)))