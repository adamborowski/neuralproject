import re
import pickle
import load_data
import numpy

words = None


def importWords():
    global words
    if words is None:
        words = pickle.load(open('target/words.p', 'rb'))
    return


def convert(sentence):
    tokens = tokenize(sentence)
    importWords()
    preformatted = [words[word] if words.has_key(word) else 0 for word in tokens]

    X, labels = load_data.load_data_by_param([preformatted], [1], nb_words=20000,
                                             test_split=0.2)
    return X


def tokenize(s):
    s = s.lower()
    s = re.sub(r"https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&/=]*)", "MAILUSED",
               s)
    s = re.sub(r"/?([\w_\-\.]+/){2,}\w+", "", s)
    s = re.sub(r"(?:[a-f\d]*\d[a-f\d]*){5,}", "", s)
    s = re.sub(r"(?:[a-z]\w+\.)+([A-Z]\w+\w+)(?![\w])", "", s)
    s = re.sub(r"([\S]*[@#][\S]*)", "",
               s)  # remove @mentions and #hastags TODO maybe mentions are to easy to learn, like #happy #nice
    s = re.sub(r"([\s\.,:;]+)", " ", s)
    s = re.sub(r"(\w+)['-](\w+)", "\g<1> \g<2>", s)  # didn't - didn t
    s = re.sub(r"([?!])+", " \g<1> ", s)

    s = s.strip(' ')
    return re.split("[\s\-_]+", s)
