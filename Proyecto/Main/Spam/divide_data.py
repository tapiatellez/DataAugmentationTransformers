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
data_file = pd.read_csv('/Users/administrador/Downloads/spam.csv',delimiter=',',encoding='latin-1')
print(data_file.head())
print(data_file.iloc[0])
#   Get rid of the columns that are not required
data_file.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
print(data_file.info())
#   Understand the distribution
sns.countplot(data_file.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
#plt.show()
#   Get the first and the second column
sentences = data_file.v2
labels = data_file.v1
#   Obtain 300 spam labels
print(labels)
spam_labels_list = []
spam_sentences_list = []
counter = 0
for i in range(len(labels)):
    if labels[i] == 'spam':
        spam_labels_list.append(labels[i])
        spam_sentences_list.append(sentences[i])
        print(sentences[i] + "\t" + labels[i])
        counter += 1
    if counter > 299:
        break

#   Obtain 300 ham sentences
ham_labels_list = []
ham_sentences_list = []
counter = 0
for i in range(len(labels)):
    if labels[i] == 'ham':
        ham_labels_list.append(labels[i])
        ham_sentences_list.append(sentences[i])
        print(sentences[i] + "\t" + labels[i])
        counter += 1
    if counter > 299:
        break

#  Create a single data list with our 600 sentences
complete_sentences_list = spam_sentences_list + ham_sentences_list
complete_labels_list = spam_labels_list + ham_labels_list
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
