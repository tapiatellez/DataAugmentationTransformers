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

#Functions
def importData(path):
    filepath_dict = {'sst2': path}
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names = ['sentences', 'labels'], sep='\t')
        df['source'] = source # Add another column filled with the source name
        df_list.append(df)
    df.info()
    df = pd.concat(df_list)
    # print(df.iloc[0])
    return df
def getDataValues(df):
    df_sst2 = df[df['source'] == 'sst2']
    sentences = df_sst2['sentences'].values
    # print("Sentences: ", sentences)
    labels = df_sst2['labels'].values
    return sentences, labels
#   Define the RNN structure through a function.
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
#   Define the CNN structure through a function
def CNN():
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
#Main
#Import the data
path_list = ['train_data_file.csv', 'augmented_mask_train_data.csv', 'augmented_mask_train_data_5.csv', 'augmented_sentence_train_data.csv']
path = path_list[3]
train_data_file = importData(path)
test_data_file = importData('test_data_file.csv')
#Separate sentences from labels
train_sentences, train_labels = getDataValues(train_data_file)
test_sentences, test_labels = getDataValues(test_data_file)
print(train_data_file)
