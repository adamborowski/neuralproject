import pickle
import convert
import csv

# preprocess csv twitter format to pickle arrays of sentiment and tokens

tokensArray = []
classificationArray = []

with open('/Users/aborowski/tmp/twitter-dataset.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    limit = 200000
    next(spamreader)
    for row in spamreader:
        tokens = convert.tokenize(row[1])
        tokensArray.append(tokens)
        classificationArray.append(int(row[0]))

        # print row[1]
        # print tokens
        limit -= 1
        if limit < 0:
            break



pickle.dump(tokensArray, open("target/tokens.p", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(classificationArray, open("target/sentiments.p", "wb"), pickle.HIGHEST_PROTOCOL)
