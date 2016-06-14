import operator
import sys


class WordIndexer:
    def __init__(self):
        self._index = dict()
        pass

    def add_word(self, word):
        if self._index.has_key(word):
            self._index[word] += 1
        else:
            self._index[word] = 1

    def compute_occurrence_keys(self, max_items=sys.maxint):
        sorted_words = sorted(self._index.items(), reverse=True, key=operator.itemgetter(1))[0:max_items]
        sorted_words = map(lambda (i, w): (w[0], i), enumerate(sorted_words))
        return dict(sorted_words)


class IndexProvider:
    def __init__(self, occurence_keys):
        self._occurence_keys = occurence_keys

    def get_word_index(self, word):
        if self._occurence_keys.has_key(word):
            return self._occurence_keys[word] + 1
        else:
            return 0
