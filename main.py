import pandas as pd
from SentiSVC import SentiSVC
from SentiRegression import SentiRegression
from SentiLexicon import SentiLexicon
from SentiNB import SentiNB
from SentiTFIDF import SentiTFIDF

# read data from txt
def read_data(set_name):
    text_file_name  = set_name + "_text.txt"
    label_file_name = set_name + "_labels.txt"
    text_file = open("data/" + text_file_name, "r", encoding="utf8")
    label_file = open("data/" + label_file_name, "r", encoding="utf8")
    x = text_file.readlines()
    y = label_file.readlines()
    text_file.close()
    label_file.close()
    for i in range(len(y)): y[i] = y[i][0]
    return pd.DataFrame(x, columns=["text"]), pd.DataFrame(y, columns=["label"])

train_set_x, train_set_y = read_data("train")
val_set_x,   val_set_y   = read_data("val")
test_set_x,  test_set_y  = read_data("test")

# senti_svc = SentiSVC()
# senti_svc.train(train_set_x, train_set_y, 100)
# print("Validation Set Result: ")
# senti_svc.test(val_set_x, val_set_y)
# print("Test Set Result: ")
# senti_svc.test(test_set_x, test_set_y)

# senti_regression = SentiRegression()
# senti_regression.train(train_set_x, train_set_y, 100)
# print("Validation Set: ")
# senti_regression.test(val_set_x, val_set_y)
# print("Test Set: ")
# senti_regression.test(test_set_x, test_set_y)

# senti_lexicon = SentiLexicon()
# senti_lexicon.train(train_set_x, train_set_y)
# print("Validation Set Result: ")
# senti_lexicon.test(val_set_x, val_set_y)
# print("Test Set: ")
# senti_lexicon.test(test_set_x, test_set_y)

# senti_nb = SentiNB()
# senti_nb.train(train_set_x, train_set_y)
# print("Validation Set Result: ")
# senti_nb.test(val_set_x, val_set_y)
# print("Test Set: ")
# senti_nb.test(test_set_x, test_set_y)

senti_tfidf = SentiTFIDF()
senti_tfidf.train(train_set_x, train_set_y)
print("Validation Set Result: ")
senti_tfidf.test(val_set_x, val_set_y)
print("Test Set: ")
senti_tfidf.test(test_set_x, test_set_y)