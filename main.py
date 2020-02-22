import keras
from keras.datasets import imdb
import numpy as np

 

def decode_word(data, reverse_word_index):
    # 0,1,2 are prereserved indecis.
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in data])

def vectorize_sequeces(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1.
    return result

def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000)
    print(train_data[0])

    # Since we can decode 10000 words. We map the review text into an array of 10000 booleans.
    # Where each flag will represent the usage on one specific word.
    x_train = vectorize_sequeces(train_data)
    x_test = vectorize_sequeces(test_data)
    print(x_train[0])

    # In this data set a text consists of an array of word indices, where each id is representing a word.
    word_index = imdb.get_word_index()

    # Swap key with value. E.g. 'brights': 63815 -> 63815: 'brights'
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    print(decode_word(train_data[4], reverse_word_index))

main()