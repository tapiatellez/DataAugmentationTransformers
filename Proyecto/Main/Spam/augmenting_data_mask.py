import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer

def importData():
    filepath_dict = {'spam': 'train_data_file.csv'}
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
    df_sst2 = df[df['source'] == 'spam']
    sentences = df_sst2['sentences'].values
    print("Sentences: ", sentences)
    labels = df_sst2['labels'].values
    return sentences, labels
def createSentenceList(predicted_sentences, label):
    sen_list = []
    for dic in predicted_sentences:
        sen = dic['sequence']
        sen = sen.replace('<s> ', '')
        sen = sen.replace('</s>', '')
        sen = sen.replace('<s>', '')
        sen = str(label) + "\t" + sen
        sen_list.append(sen)
    return sen_list
def createSentenceListOne(predicted_sentences, label):
    sen_list = []
    sen = predicted_sentences[0]['sequence']
    sen = sen.replace('<s> ', '')
    sen = sen.replace('</s>', '')
    sen = sen.replace('<s>', '')
    sen = sen + "\t" + str(label)
    sen_list.append(sen)
    return sen_list

def createDataAugmentationDictionary(sentences, labels):
    sentences, labels = getDataValues(data_file)
    print(sentences)
    #   Create five new sentences for each sentence in the data
    nlp = pipeline("fill-mask")
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        print("Label: ", type(label))
        print("Sentence: ", type(sentence))
        if len(tokenized_sentence) > 1:
            mask = nlp.tokenizer.mask_token
            print("Mask: ", mask)
            print("Length of sentence: ", len(tokenized_sentence))
            rn = randint(0, len(tokenized_sentence)-1)
            print("Random number: ", rn)
            print("Selected word: ", tokenized_sentence[rn])
            tokenized_sentence[rn] = mask
            print("Tokenized sentence with mask: ", tokenized_sentence)
            untokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)
            print("Untokenized sentence with mask: ", untokenized_sentence)
            predicted_sentences = nlp(untokenized_sentence)
            print("Predicted Sentences: ", predicted_sentences)
            sentences_list = createSentenceListOne(predicted_sentences, label)
            print("Predicted Sentences List: ", sentences_list)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    #print(predicted_sentences_dict)
    return(predicted_sentences_dict)
#   Split the data
from sklearn.model_selection import train_test_split
# def createDataAugmentationDictionaryMulti(sentences, labels):

#Main
#Get the data
data_file = importData()
#Separate sentences from labels
sentences, labels = getDataValues(data_file)
print(sentences)
#   Create five new sentences for each sentence in the data
data_augmentation_dictionary = createDataAugmentationDictionary(sentences, labels)
#   Create file with augmented data.
augmented_file = open("augmented_mask_train_data.csv", "w+")
augmented_string = ""
for sen, sen_list in data_augmentation_dictionary.items():
    #print(sen)
    augmented_string = augmented_string + sen + "\n"
    if sen_list:
        for s in sen_list:
            augmented_string = augmented_string + s + "\n"

augmented_file.write(augmented_string)
augmented_file.close()
print(augmented_string)
