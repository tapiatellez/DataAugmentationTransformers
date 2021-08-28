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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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
path_list = ['train_data_file.csv',
            'augmented_mask_train_data_1.csv',
            'augmented_mask_train_data_3.csv',
            'augmented_mask_train_data_5.csv',
            'augmented_mask_train_data_10.csv',
            'augmented_data_double_masking_1.csv',
            'augmented_data_double_masking_3.csv',
            'augmented_data_double_masking_5.csv',
            'augmented_data_double_masking_10.csv',
            'augmented_data_triple_masking_1.csv',
            'augmented_data_triple_masking_3.csv',
            'augmented_data_triple_masking_5.csv',
            'augmented_data_triple_masking_10.csv',
            'augmented_sentence_train_data.csv']

kf = KFold(n_splits = 5)
output_string = ""
for path in path_list:
    #Get the data
    resultsCNN = np.array([])
    resultsRNN = np.array([])
    data_file = importData(path)
    print("Data file: ")
    print(data_file)
    sentences, labels = getDataValues(data_file)
    for train_index, test_index in kf.split(sentences, labels):
        #Separate sentences from labels
        train_sentences, train_labels = sentences[train_index], labels[train_index]
        test_sentences, test_labels = sentences[test_index], labels[test_index]
        # for sentence, label in zip(test_sentences, test_labels):
        #     print(sentence + "  " + str(label))
        train_labels = [str(i) for i in train_labels]
        test_labels = [str(i) for i in test_labels]
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)
        train_labels = train_labels.reshape(-1,1)
        test_labels = le.fit_transform(test_labels)
        test_labels = test_labels.reshape(-1, 1)
        #   Tokenize the data, padd it and take the max words and length
        max_words = 1000
        max_len = 150
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(train_sentences)
        sequences = tok.texts_to_sequences(train_sentences)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen = max_len)
        # Let´s call the function to construct the Network and compile the model
        modelRNN = RNN()
        modelRNN.summary()
        modelRNN.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        #Save the model
        checkpoint_path = "/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/SST2/training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #   Fit the training data.
        modelRNN.fit(sequences_matrix,train_labels,batch_size=128,epochs=10,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
        #   Process the test set data
        test_sequences = tok.texts_to_sequences(test_sentences)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

        #Evaluate the model on the test set
        accr = modelRNN.evaluate(test_sequences_matrix,test_labels)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        #   Serialize model to JSON
        model_jsonRNN = modelRNN.to_json()
        with open("modelRNN_mask_five.json", "w") as json_file:
            json_file.write(model_jsonRNN)
        #   Serialize weights to HDF5
        modelRNN.save_weights("modelRNN_mask_five.h5")
        print("Saved model to disk")
        # load json and create model
        json_file = open('modelRNN_mask_five.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("modelRNN_mask_five.h5")
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

        #   Let's call the function to construct the CNN Network and compile the model
        embedding_dim = 150
        vocab_size = len(tok.word_index)+1
        modelCNN = CNN()
        modelCNN.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['acc'])
        modelCNN.summary()
        modelCNN.fit(sequences_matrix, train_labels,
                     epochs = 10,
                     verbose = False)
        #Evaluate the model on the test set
        accrCNN = modelCNN.evaluate(test_sequences_matrix, test_labels)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accrCNN[0],accrCNN[1]))

        #   Serialize model to JSON
        model_jsonCNN = modelCNN.to_json()
        with open("modelCNN_mask_five.json", "w") as json_file:
            json_file.write(model_jsonCNN)
        #   Serialize weights to HDF5
        modelCNN.save_weights("modelCNN_mask_five.h5")
        print("Saved model to disk")
        # load json and create model
        json_file = open('modelCNN_mask_five.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("modelCNN_mask_five.h5")
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(test_sequences_matrix, test_labels, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        resultsCNN = np.append(resultsCNN, score[1])
        resultsRNN = np.append(resultsRNN, accr[1])
    output_string = output_string + "Results for file: " + path + "\n"
    print("Results for file: ", path)
    print("The results for the five runs:")
    output_string = output_string + "The results for the five runs: CNN: " + str(resultsCNN) + " Average: " + str(np.average(resultsCNN)) + "\n"
    print("For CNN:", resultsCNN)
    print("The average for CNN:", np.average(resultsCNN))
    output_string = output_string + "The results for the five runs: RNN: " + str(resultsRNN) + " Average: " + str(np.average(resultsRNN)) + "\n"
    print("For RNN:", resultsRNN)
    print("The average for RNN:", np.average(resultsRNN))
print(output_string)
