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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

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
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

filepath_dict = {'yelp': '/Users/administrador/Downloads/sentiment labelled sentences/yelp_labelled.txt'
                 ,'imdb': '/Users/administrador/Downloads/sentiment labelled sentences/imdb_labelled.txt'
                 ,'amazon': '/Users/administrador/Downloads/sentiment labelled sentences/amazon_cells_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['sentences', 'label'], sep='\t')
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
# Create a dictionary with each parameters named as in the previous function.
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[5000],
                  embedding_dim=[50],
                  maxlen=[100])

# Main settings
epochs = 20
embedding_dim = 50
maxlen = 100
output_file = 'data/output.txt'
# Run grid search for each source (yelp, amazon, imdb)
for source, frame in df.groupby('source'):
    print('Running grid search for data set :', source)
    sentences = df['sentences'].values
    y = df['label'].values

    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # #Embedding matrix
    # embedding_matrix = create_embedding_matrix('/Users/administrador/Downloads/glove/glove.6B.50d.txt'
    #                                            ,tokenizer.word_index
    #                                            ,embedding_dim)

    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)

    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)

    # Save and evaluate results
    prompt = input(f'finished {source}; write to file and proceed? [y/n]')
    if prompt.lower() not in {'y', 'true', 'yes'}:
        break
    with open(output_file, 'a') as f:
        s = ('Running {} data set\nBest Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            source,
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
        print(output_string)
        f.write(output_string)

#
# model = Sequential()
# model.add(layers.Embedding(vocab_size, embedding_dim,
#                            weights = [embedding_matrix],
#                            input_length=maxlen,
#                            trainable = True))
# model.add(layers.Conv1D(128, 5, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
# #               metrics=['acc'])
# model.summary()
#
# history = model.fit(X_train, y_train,
#                     epochs=10,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
#
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# plot_history(history)

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
