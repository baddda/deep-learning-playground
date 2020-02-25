import keras
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
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

    # Array of booleans. Value of 1 means positive review.
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    print(y_train)

    # In this data set a text consists of an array of word indices, where each id is representing a word.
    word_index = imdb.get_word_index()

    # Swap key with value. E.g. 'brights': 63815 -> 63815: 'brights'
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    print(decode_word(train_data[4], reverse_word_index))
    
    # Set up nodes.
    # Will look like this: 10000 nodes -> 16 nodes -> 16 nodes -> 1 nodes
    # The activiation function is the function on how the combined input to a node should be passed to the next node.
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Set up the optimizer, so basically the method of finding a global minimum in our neural network function.
    # Set up the loss function, so the backpropagation knows the direction.
    # Set up the metrics function. It's basically just to see how our model is performing. So it is not used for learning.
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the network.
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)


main()