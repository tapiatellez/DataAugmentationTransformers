#Libraries and packages
import os
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout

def importData(path):
    filepath_dict = {'sst2': path}
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names = ['sentences', 'labels'], sep='\t')
        df['source'] = source # Add another column filled with the source name
        df_list.append(df)
    df.info()
    df = pd.concat(df_list)
    print(df.iloc[0])
    return df
def getDataValues(df):
    df_sst2 = df[df['source'] == 'sst2']
    sentences = df_sst2['sentences'].values
    print("Sentences: ", sentences)
    labels = df_sst2['labels'].values
    return sentences, labels
#   Main
#   Get and prepare the data
test_data_file = importData('test_data_file.csv')
test_sentences, test_labels = getDataValues(test_data_file)
le = LabelEncoder()
test_labels = le.fit_transform(test_labels)
test_labels = test_labels.reshape(-1, 1)

#Process the test set data
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(test_sentences)
test_sequences = tok.texts_to_sequences(test_sentences)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen = max_len)

# load json and create model
json_file = open('modelRNN_plain.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelRNN_plain.h5")
print("Loaded model from disk RNN plain")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# load json and create model
json_file = open('modelCNN_plain.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelCNN_plain.h5")
print("Loaded model from disk CNN plain")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# load json and create model
json_file = open('modelRNN_mask_one.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelRNN_mask_one.h5")
print("Loaded model from disk RNN mask one: ")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# load json and create model
json_file = open('modelCNN_mask_one.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelCNN_mask_one.h5")
print("Loaded model from disk CNN mask one: ")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# load json and create model
json_file = open('modelRNN_mask_five.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelRNN_mask_five.h5")
print("Loaded model from disk RNN mask five: ")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# load json and create model
json_file = open('modelCNN_mask_five.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelCNN_mask_five.h5")
print("Loaded model from disk CNN mask five: ")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
