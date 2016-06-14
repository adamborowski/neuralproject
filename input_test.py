from keras.preprocessing import sequence
from modules.word_indexer import IndexProvider
import pickle
import convert
import numpy as np


def test(model):
    indexProvider = IndexProvider(pickle.load(open('target/words.p', 'rb')))

    while True:
        p = raw_input("Enter your sentence: ")
        words = p.split()
        if len(words) == 0:
            break
        tokens = convert.tokenize(p)
        indexes = [1] + [3 + indexProvider.get_word_index(token) for token in tokens]

        truncated = sequence.pad_sequences([indexes], maxlen=80)

        result = model.predict_classes(np.array(truncated))
        print("result:" + "Positive" if result[0] == 1 else "Negative")


if __name__ == "__main__":
    test(None)
