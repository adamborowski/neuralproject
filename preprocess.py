import csv
import pickle

from modules import convert

# preprocess csv twitter format to pickle arrays of sentiment and tokens

tokensArray = []
classificationArray = []

with open('target/twitter-dataset.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    # limit = 200000
    next(spamreader)
    for row in spamreader:
        tokens = convert.tokenize(row[1])
        tokensArray.append(tokens)
        classificationArray.append(int(row[0]))
        if len(tokensArray) % 10000 == 0:
            print len(tokensArray).__str__() + " tokens..."
            # print row[1]
            # print tokens
            # limit -= 1
            # if limit < 0:
            #     break

pickle.dump(tokensArray, open("target/tokens.p", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(classificationArray, open("target/sentiments.p", "wb"), pickle.HIGHEST_PROTOCOL)
