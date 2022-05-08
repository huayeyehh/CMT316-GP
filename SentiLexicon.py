import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import numpy as np
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

# init lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()
# Convert between the PennTreebank tags to simple Wordnet tags
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

# stopwords
stopwords = set(nltk.corpus.stopwords.words("english"))
additional_stopwords = [".", ",", "'s", "``", "''", "'",
                        "n't", "%", "-", "$", "(", ")", ":",
                        ";", "@", "&", "'m", "user", "#", "!",
                        "?", "...", "a"]
for sw in additional_stopwords: stopwords.add(sw)

# get average sentimental score of a sentence, per word
def get_senti_score(sentence):
    token = nltk.word_tokenize(sentence)
    # remove stop words
    index = len(token) - 1
    while index >= 0:
        if token[index] in stopwords:
            token.pop(index)
        index -= 1
    after_tagging = nltk.pos_tag(token)
    sentiment = 0.0
    objective = 0.0
    tokens_count = 0
    word_types = [wn.VERB, wn.NOUN, wn.ADJ, wn.ADV]
    for word, tag in after_tagging:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in word_types: continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma: continue

        synsets = list(swn.senti_synsets(lemma, pos=wn_tag))
        if not synsets: continue

        swn_synset = synsets[0]

        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        objective += swn_synset.obj_score()
        tokens_count += 1
    if tokens_count == 0: return [0, 0]
    return [sentiment / tokens_count, objective / tokens_count]

class SentiLexicon(object):
    def __init__(self):
        self.model = sklearn.svm.SVC(kernel="linear", gamma='auto')
        print("Senti Lexicon Created.")

    def train(self, train_set_x, train_set_y):
        # prepare data
        x = []
        y = []
        for i in train_set_x.index:
            for sentence in nltk.tokenize.sent_tokenize(train_set_x.loc[i, "text"]):
                x.append(get_senti_score(sentence))
                y.append(train_set_y.loc[i, "label"])

        # Train model
        self.model.fit(np.asarray(x), np.asarray(y))
        print("Senti Lexicon Model Trained.")
    
    def test(self, val_set_x, val_set_y):
        x, y = [], []
        for i in val_set_x.index:
            for sentence in nltk.tokenize.sent_tokenize(val_set_x.loc[i, "text"]):
                x.append(get_senti_score(sentence))
                y.append(val_set_y.loc[i, "label"])
        predictions = self.model.predict(x)
        y = np.asarray(y)

        print("precision_score: " + str(precision_score(y, predictions, average='macro')))
        print("recall_score: " + str(recall_score(y, predictions, average='macro')))
        print("f1_score: " + str(f1_score(y, predictions, average='macro')))
        print("accuracy_score: " + str(accuracy_score(y, predictions)))
