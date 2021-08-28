import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def importData():
    filepath_dict = {'spam': 'spam.csv'}
    data_file_list = []
    for source, filepath in filepath_dict.items():
        data_file = pd.read_csv(filepath, names = ['labels', 'sentences'], sep='\t')
        data_file['source'] = source # Add another column filled with the source name
        data_file_list.append(data_file)
    data_file.info()
    data_file = pd.concat(data_file_list)
    print(data_file.iloc[0])
    return data_file

def getDataValues(data_file):
    data_file_sst2 = data_file[data_file['source'] == 'sst2']
    sentences = data_file_sst2['sentences'].values
    print("Sentences: ", sentences)
    labels = data_file_sst2['labels'].values
    return sentences, labels

#   Main
#   Get the data
#   Load the data into Pandas dataframe
data_file = pd.read_csv('imdb_labelled.txt', names = ["sentence", "label"], sep = '\t')
print(data_file.head())
print(data_file.info())
#   Understand the distribution
# sns.countplot(data_file.sentence)
# plt.xlabel('Label')
# plt.title('Number of ham and spam messages')
#plt.show()
#   Get the first and the second column
sentences = data_file.sentence
labels = data_file.label
#   Obtain 300 negative texts
negative_labels_list = []
negative_sentences_list = []
counter = 0
for i in range(len(labels)):
    if labels[i] == 0:
        negative_labels_list.append(labels[i])
        negative_sentences_list.append(sentences[i])
        print(sentences[i] + "\t" + str(labels[i]))
        counter += 1
    if counter > 299:
        break
#   Obtain 300 positive texts
positive_labels_list = []
positive_sentences_list = []
counter = 0
for i in range(len(labels)):
    if labels[i] == 1:
        positive_labels_list.append(labels[i])
        positive_sentences_list.append(sentences[i])
        print(sentences[i] + "\t" + str(labels[i]))
        counter += 1
    if counter > 299:
        break
#  Create a single data list with our 600 sentences
complete_sentences_list = positive_sentences_list + negative_sentences_list
complete_labels_list = positive_labels_list + negative_labels_list
print(complete_sentences_list)
print(complete_labels_list)
#   Select a random number of sentences
random_list = random.sample(range(len(complete_sentences_list)), 100)
training_data = []
test_data = []
for i in range(len(complete_labels_list)):
    if i in random_list:
        test_data.append(complete_sentences_list[i] + "\t" + str(complete_labels_list[i]))
    else:
        training_data.append(complete_sentences_list[i] + "\t" + str(complete_labels_list[i]))
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
print(data_file.info())
