#Get the data
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow_datasets as tfdfs
import numpy as np

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

# Word embeddings
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

filepath_dict = {'sst2': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/sst2_train_500.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['label', 'sentences'], sep='\t')
    df['source'] = source # Add another column filled with the source name
    df_list.append(df)
df.info()
df = pd.concat(df_list)
print(df.iloc[0])

#   Split the data
from sklearn.model_selection import train_test_split

df_sst2 = df[df['source'] == 'sst2']
sentences = df_sst2['sentences'].values
#print("Sentences: ", sentences)
y = df_sst2['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size = 0.20, random_state = 1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print("Vocab size: ", vocab_size)
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#   Create the word embedding from pretrained data.
embedding_dim = 50
embedding_matrix = create_embedding_matrix('/Users/administrador/Downloads/IMDB Dataset.csv'
                                           , tokenizer.word_index
                                           ,embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights = [embedding_matrix],
                           input_length=maxlen,
                           trainable = True))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

#Create a file with the trained data in order to augment it and do some tests

#plot_graphs(history, accuracy)
# filepath_dict = {'sst2': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/augmented_data_500.txt'}
# df_list = []

# #   Let´s check with a completely new data
# filepath_dictN = {'amazon': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/sentiment labelled sentences/amazon_cells_labelled.txt'
#                   ,'imdb': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/sentiment labelled sentences/imdb_labelled.txt'
#                   ,'yelp': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/sentiment labelled sentences/yelp_labelled.txt'}
# df_listN = []
# for source, filepath in filepath_dictN.items():
#     df = pd.read_csv(filepath, names = ['sentences', 'labels'], sep='\t')
#     df['source'] = source # Add another column filled with the source name
#     df_listN.append(df)
# df.info()
# df = pd.concat(df_listN)
# print(df.iloc[0])
# #   Split the data
# from sklearn.model_selection import train_test_split
#
# df_sst2 = df[df['source'] == 'amazon']
# sentences = df_sst2['sentences'].values
# print("Sentences: ", sentences)
# y = df_sst2['labels'].values
#
# sentences_trainA, sentences_testA, y_trainA, y_testA = train_test_split(sentences, y, test_size = 0.20, random_state = 1000)
#
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(sentences_trainA)
# X_trainA = tokenizer.texts_to_sequences(sentences_trainA)
# X_testA = tokenizer.texts_to_sequences(sentences_testA)
# vocab_sizeA = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
# maxlen = 100
# X_trainA = pad_sequences(X_trainA, padding='post', maxlen=maxlen)
# X_testA = pad_sequences(X_testA, padding='post', maxlen=maxlen)
#
# history = model.fit(X_trainA, y_trainA,
#                     epochs=10,
#                     verbose=False,
#                     validation_data=(X_testA, y_testA),
#                     batch_size=10)
# loss, accuracy = model.evaluate(X_testA, y_testA, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
