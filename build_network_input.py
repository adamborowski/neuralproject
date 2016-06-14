import pickle
from modules.word_indexer import *

NUM_MOST_POPULAR_WORDS = 300

print("Phase 0: load tokens.p as words to replace with indexes")

word_array = pickle.load(open('target/tokens.p', 'rb'))

print("Phase 1: get most common words and index by popularity")
indexer = WordIndexer()
for words in word_array:
    for word in words:
        indexer.add_word(word)

print("Phase 2: replace every word with the occurrence key which is related to word popularity")
occurrence_keys = indexer.compute_occurrence_keys(NUM_MOST_POPULAR_WORDS)
index_provider = IndexProvider(occurrence_keys)
input_data = []

for words in word_array:
    word_indexes = []
    input_data.append(word_indexes)
    for word in words:
        word_indexes.append(index_provider.get_word_index(word))

print("Phase 3: save input_data (indices of words), save occurence_keys (word to index resolver)")
pickle.dump(input_data, open("target/tokenIndexes.p", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(occurrence_keys, open("target/words.p", "wb"), pickle.HIGHEST_PROTOCOL)
