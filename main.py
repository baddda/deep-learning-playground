import keras
from keras.datasets import imdb

def decodeWord(data):
    # 0,1,2 are prereserved indecis.
    ' '.join([reverse_word_index.get(i - 3, '?') for i in data])


(train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000)
print(train_data[0])

# In this data set a text consists of an array of word indices, where each id is representing a word.
word_index = imdb.get_word_index()
# Swap key with value. E.g. 'brights': 63815 -> 63815: 'brights'
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(decodeWord(train_data[0]))