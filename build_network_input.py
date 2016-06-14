import convert
import pickle
import modules.word_indexer

array = pickle.load(open('target/tokens.p', 'rb'))

uniqueWords = 0
totalWords = 0
words = dict()
tokenIndexArray = []


def getWordIndex(word):
    global uniqueWords
    if words.has_key(word):
        return words.get(word)
    uniqueWords += 1
    words[word] = uniqueWords
    return uniqueWords


for tokens in array:
    tokenIndex = []
    tokenIndexArray.append(tokenIndex)

    for token in tokens:
        totalWords += 1
        tokenIndex.append(getWordIndex(token))

print "Tweets: {}, total words: {}, unique words: {}".format(len(array), totalWords, uniqueWords)

exit(0)
pickle.dump(tokenIndexArray, open("target/tokenIndexes.p", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(words, open("target/words.p", "wb"), pickle.HIGHEST_PROTOCOL)
