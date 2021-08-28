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
        df = pd.read_csv(filepath, names = ['sentences', 'labels'], sep='\t', engine='python', error_bad_lines = False)
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
plain_list = ['base_data_0.csv',
              'base_data_1.csv',
              'base_data_2.csv',
              'base_data_3.csv',
              'base_data_4.csv',
              'base_data_5.csv',
              'base_data_6.csv',
              'base_data_7.csv',
              'base_data_8.csv',
              'base_data_9.csv',]
augmented_list = ['augmented_data_0.csv',
                  'augmented_data_1.csv',
                  'augmented_data_2.csv',
                  'augmented_data_3.csv',
                  'augmented_data_4.csv',
                  'augmented_data_5.csv',
                  'augmented_data_6.csv',
                  'augmented_data_7.csv',
                  'augmented_data_8.csv',
                  'augmented_data_9.csv',]
# results_dictionary = {}
# average_dictionary = {}
# for path in path_list:
#     average_dictionary[path] = np.array([0, 0])
#
output_file = open("results.txt", "w+")
output_string = ""
percentage = 1
for plain, augmented in zip(plain_list, augmented_list):
    plain_data_file = importData(plain)
    augmented_data_file = importData(augmented)
    test_data_file = importData('test_data_file.csv')
    #Separate sentences from labels
    plain_sentences, plain_labels = getDataValues(plain_data_file)
    augmented_sentences, augmented_labels = getDataValues(augmented_data_file)
    test_sentences, test_labels = getDataValues(test_data_file)
    # for sentence, label in zip(test_sentences, test_labels):
    #     print(sentence + "  " + str(label))
    plain_labels = [str(i) for i in plain_labels]
    augmented_labels = [str(i) for i in augmented_labels]
    test_labels = [str(i) for i in test_labels]
    le = LabelEncoder()
    plain_labels = le.fit_transform(plain_labels)
    plain_labels = plain_labels.reshape(-1,1)
    augmented_labels = le.fit_transform(augmented_labels)
    augmented_labels = augmented_labels.reshape(-1, 1)
    test_labels = le.fit_transform(test_labels)
    test_labels = test_labels.reshape(-1, 1)
    #   Tokenize the data, padd it and take the max words and length
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(plain_sentences)
    tok.fit_on_texts(augmented_sentences)
    plain_sequences = tok.texts_to_sequences(plain_sentences)
    plain_sequences_matrix = sequence.pad_sequences(plain_sequences, maxlen = max_len)
    augmented_sequences = tok.texts_to_sequences(augmented_sentences)
    augmented_sequences_matrix = sequence.pad_sequences(augmented_sequences, maxlen = max_len)
    # Let´s call the function to construct the Network and compile the model
    modelRNNA = RNN()
    modelRNNA.summary()
    modelRNNA.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    modelRNNB = RNN()
    modelRNNB.summary()
    modelRNNB.compile(loss = 'binary_crossentropy', optimizer = RMSprop(), metrics =['accuracy'])
    #   Fit the training data.
    modelRNNA.fit(plain_sequences_matrix,plain_labels,batch_size=128,epochs=10,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    modelRNNB.fit(augmented_sequences_matrix, augmented_labels, batch_size = 128, epochs = 10, callbacks =[EarlyStopping(monitor = 'val_loss', min_delta = 0.0001)])
    #   Process the test set data
    test_sequences = tok.texts_to_sequences(test_sentences)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    #Evaluate the model on the test set
    accrA = modelRNNA.evaluate(test_sequences_matrix,test_labels)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accrA[0],accrA[1]))
    #results.append({"RNNA": accr[1]})
    accrB = modelRNNB.evaluate(test_sequences_matrix, test_labels)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accrB[0],accrB[1]))
    #results.append({"RNNA": accr[1]})
    output_string = output_string + "Percentage:" + str(percentage*10) + " RNNA: " + str(accrA[1]) + " RNNB: " + str(accrB[1]) + "\n"
    percentage += 1
print(output_string)
output_file.write(output_string)
output_file.close()
