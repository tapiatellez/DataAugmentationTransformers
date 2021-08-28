#Libraries and packages
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
#Load the data into Pandas dataframe
df = pd.read_csv('/Users/administrador/Downloads/spam.csv',delimiter=',',encoding='latin-1')
print(df.head())
print(df.iloc[0])
# Get rid of the columns that are not required
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
print(df.info())
#Understand the distribution
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
#plt.show()
#Create the input and output vectors
X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
#   Split into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
#   Tokenize the data, padd it and take the max words and length
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
# Let´s call the function to construct the Network and compile the model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#   Fit the training data.
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#   Process the test set data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#Evaluate the model on the test set
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#   Let's call the function to construct the CNN Network and compile the model
embedding_dim = 100
vocab_size = len(tok.word_index)+1
modelCNN = CNN()
modelCNN.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
modelCNN.summary()
#   Fit the CNN
modelCNN.fit(sequences_matrix, Y_train,
             epochs = 10,
             verbose = False,
             validation_split = 0.2, callbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0.0001)])
#Evaluate the model on the test set
accrCNN = modelCNN.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accrCNN[0],accrCNN[1]))
