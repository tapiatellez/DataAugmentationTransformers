from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

def importData():
    filepath_dict = {'st': 'train_data_file.csv'}
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
    df_sst2 = df[df['source'] == 'st']
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
    for sen in predicted_sentences:
        sen = sen['sequence']
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
def createDataAugmentationDictionary(sentences, labels):
    #   Create 10 new sentences for each sentence in the data
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        print("Label: ", type(label))
        print("Sentence: ", type(sentence))
        if len(tokenized_sentence) > 1:
            mask = tokenizer.mask_token
            print("Mask: ", mask)
            print("Length of sentence: ", len(tokenized_sentence))
            rn = randint(0, len(tokenized_sentence)-1)
            print("Random number: ", rn)
            print("Selected word: ", tokenized_sentence[rn])
            tokenized_sentence[rn] = mask
            print("Tokenized sentence with mask: ", tokenized_sentence)
            untokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)
            print("Untokenized sentence with mask: ", untokenized_sentence)
            input = tokenizer.encode(untokenized_sentence, return_tensors = "pt")
            print("Input: ", input)
            mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

            token_logits = model(input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_10_tokens = torch.topk(mask_token_logits, 10, dim = 1).indices[0].tolist()
            sentences_list = create_sentences_list_for_ten(top_10_tokens, untokenized_sentence, label)
            print("Predicted Sentences List: ", sentences_list)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    return predicted_sentences_dict
def create_sentences_list_for_ten(top_10_tokens, sentence, label):
    sen_list = []
    for token in top_10_tokens:
        sen = sentence.replace(tokenizer.mask_token, tokenizer.decode([token]))
        sen = sen + "\t" + str(label)
        sen_list.append(sen)
    return sen_list
#   The following function receives a list of sentences, a list of labels and
#   the number of run they are in. It returns the percentage of sentences and
#   labels corresponding to the run number.
def get_percentage(sentences, labels, i):
    length = (i+1)*40
    sp = []
    lp = []
    for j in range(length):
        sp.append(sentences[j])
        lp.append(labels[j])
    return sp, lp
#   The following function receives a list with the percentage of sentences
#   ,a list with the same percentage of labels and the run number. It creates a
#   a unique file to that percentage of the data.
def create_file_for_base_data(sentences, labels, i):
    name = "base_data_" + str(i) +".csv"
    base_file = open(name, "w+")
    output_string = ""
    for sen, label in zip(sentences, labels):
        output_string = output_string + sen + "\t" + str(label) + "\n"
    base_file.write(output_string)
    base_file.close()

#   The following function receives the data_augmentation_dictionary and returns
#   a unique file for the data provided by this dictionary.
def create_file_for_augmented_data(data_augmentation_dictionary, i):
    name = "augmented_data_" + str(i) +".csv"
    augmented_file = open(name, "w+")
    augmented_string = ""
    for sen, sen_list in data_augmentation_dictionary.items():
        augmented_string = augmented_string + sen + "\n"
        if sen_list:
            for s in sen_list:
                augmented_string = augmented_string + s + "\n"
    augmented_file.write(augmented_string)
    augmented_file.close()

#   Split the data
from sklearn.model_selection import train_test_split
# def createDataAugmentationDictionaryMulti(sentences, labels):

#Main
#Get the data
data_file = importData()
#Separate sentences from labels
sentences, labels = getDataValues(data_file)
print(sentences)
print(len(sentences))

for i in range(10):
    #   Obtain the respective percentage
    sentences_percent, labels_percent = get_percentage(sentences, labels, i)
    create_file_for_base_data(sentences_percent, labels_percent, i)
    data_augmentation_dictionary = createDataAugmentationDictionary(sentences_percent, labels_percent)
    create_file_for_augmented_data(data_augmentation_dictionary, i)


#   Create five new sentences for each sentence in the data
# data_augmentation_dictionary = createDataAugmentationDictionary(sentences, labels)
# #   Create file with augmented data.
# augmented_file = open("augmented_mask_train_data_3.csv", "w+")
# augmented_string = ""
# for sen, sen_list in data_augmentation_dictionary.items():
#     #print(sen)
#     augmented_string = augmented_string + sen + "\n"
#     if sen_list:
#         for s in sen_list:
#             augmented_string = augmented_string + s + "\n"
#
# augmented_file.write(augmented_string)
# augmented_file.close()
#print(augmented_string)
