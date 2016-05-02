import convert
from keras.preprocessing import sequence

def test(model):
    while True:
        p = raw_input("Enter your sentence: ")  # raw_input() function
        words = p.split()
        if len(words) == 0:
            break
        X  = convert.convert(p)
        X_test = sequence.pad_sequences([X], maxlen=80)
        result = model.predict(X_test)
        # todo some conversion issue, cannot predict at the moment
        print("result:", result)


if __name__ == "__main__":
    test(None)
