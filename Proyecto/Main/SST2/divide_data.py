import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer

def importData():
    filepath_dict = {'sst2': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/sst2_train_500.txt'}
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names = ['labels', 'sentences'], sep='\t')
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

#Main
#Get the data
data_file = importData()
sentences = data_file['sentences'].values
labels = data_file['labels'].values
random_list = random.sample(range(500), 100)
training_data = []
test_data = []
for i in range(500):
    if i in random_list:
        test_data.append(sentences[i] + "\t" + str(labels[i]))
    else:
        training_data.append(sentences[i] + "\t" + str(labels[i]))
print("Test data: ", test_data)
print("Training data: ", training_data)

#Create files
train_data_file = open("train_data_file.csv", "w+")
test_data_file = open("test_data_file.csv", "w+")
train_data_string = ""
test_data_string = ""
for text in training_data:
    train_data_string = train_data_string + text + "\n"
for text in test_data:
    test_data_string = test_data_string + text + "\n"
train_data_file.write(train_data_string)
test_data_file.write(test_data_string)
train_data_file.close()
test_data_file.close()
