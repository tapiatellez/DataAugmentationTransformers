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

#   The following function doesn't take any parameters. It reads a file by using
#   the pandas library and returns the sentences and labels obtained from the
#   file in a list.
def importData():
    filepath_dict = {'st': 'train_data_file.csv'}
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names = ['sentences', 'labels'], sep='\t')
        df['source'] = source # Add another column filled with the source name
        df_list.append(df)
    #df.info()
    df = pd.concat(df_list)
    #print(df.iloc[0])
    return df

#   The following function receives a list with the values read from the file.
#   It obtains the sentences and the labels and returns them.
def getDataValues(df):
    df_sst2 = df[df['source'] == 'st']
    sentences = df_sst2['sentences'].values
    #print("Sentences: ", sentences)
    labels = df_sst2['labels'].values
    return sentences, labels

#   The following function receives the predicted sentences provided by the
#   Transformer, it cleans them, adds a label and returns them.
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
#   The following function receives the predicted sentences and the label
#   corresponding to them. It cleans the sentences, adds the label and returns
#   a list with the sentences addded.
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
#   The following function utilizes a pipeline Transformer (An easier tu use
#   version of Transformers).
#   The function receives a sentences list with their respective labels list. It
#   uses nltk to do some preprocessing of the sentence and mask it, it predicts
#   five sentences based on the masking sentence adds them into a list and
#   produces a dictionary with the structure: sentence_augmented/list_of_sentences.
#   It returns this dictionary.
def createDataAugmentationDictionary(sentences, labels):
    sentences, labels = getDataValues(data_file)
    #print(sentences)
    #   Create five new sentences for each sentence in the data
    nlp = pipeline("fill-mask")
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        #print("Label: ", type(label))
        #print("Sentence: ", type(sentence))
        if len(tokenized_sentence) > 1:
            mask = nlp.tokenizer.mask_token
            #print("Mask: ", mask)
            #print("Length of sentence: ", len(tokenized_sentence))
            rn = randint(0, len(tokenized_sentence)-1)
            #print("Random number: ", rn)
            #print("Selected word: ", tokenized_sentence[rn])
            tokenized_sentence[rn] = mask
            #print("Tokenized sentence with mask: ", tokenized_sentence)
            untokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)
            #print("Untokenized sentence with mask: ", untokenized_sentence)
            predicted_sentences = nlp(untokenized_sentence)
            #print("Predicted Sentences: ", predicted_sentences)
            sentences_list = createSentenceListOne(predicted_sentences, label)
            #print("Predicted Sentences List: ", sentences_list)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    #print(predicted_sentences_dict)
    return(predicted_sentences_dict)
#   This function creates the augmented sentences by using BERT.
#   The function receives a list of sentences with their respective labels and
#   and the amount of sentences you wish to create. It returns a dictionary with
#   the structure: {sentence-to-be-augmented:list-of-augmented-sentences}, finally
#   it returns this dictionary.
def createDataAugmentationDictionaryGood(sentences, labels, amount):
    #   Create 10 new sentences for each sentence in the data
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        #print("Label: ", type(label))
        #print("Sentence: ", type(sentence))
        if len(tokenized_sentence) > 1:
            mask = tokenizer.mask_token
            #print("Mask: ", mask)
            #print("Length of sentence: ", len(tokenized_sentence))
            rn = randint(0, len(tokenized_sentence)-1)
            #print("Random number: ", rn)
            #print("Selected word: ", tokenized_sentence[rn])
            tokenized_sentence[rn] = mask
            #print("Tokenized sentence with mask: ", tokenized_sentence)
            untokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)
            #print("Untokenized sentence with mask: ", untokenized_sentence)
            input = tokenizer.encode(untokenized_sentence, return_tensors = "pt")
            #print("Input: ", input)
            mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

            token_logits = model(input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_10_tokens = torch.topk(mask_token_logits, amount, dim = 1).indices[0].tolist()
            sentences_list = create_sentences_list_for_ten(top_10_tokens, untokenized_sentence, label)
            #print("Predicted Sentences List: ", sentences_list)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    return predicted_sentences_dict
#   The following function receives a list of tokens, a sentence and a label. It
#   returns a list of sentences where for each sentence its mask position is
#   replaced by a the token in turn.
def create_sentences_list_for_ten(top_10_tokens, sentence, label):
    sen_list = []
    for token in top_10_tokens:
        sen = sentence.replace(tokenizer.mask_token, tokenizer.decode([token]))
        sen = sen + "\t" + str(label)
        sen_list.append(sen)
    return sen_list
#   The follwing function receives a list of numbers, sentences and their respective
#   labels. It creates as many files as there're numbers in the lis of numbers
#   Each with number in the end of the name of the file. Each file contains
#   the base data plus the augmented data, where for each sentence we add as many
#   sentences we add n(number) sentences. The sentences are added via BERT.
def augment_data(number_list, sentences, labels):
    for number in number_list:
        data_augmentation_dictionary = createDataAugmentationDictionaryGood(sentences, labels, number)
        #   Create file with augmented data.
        augmented_file = open("augmented_mask_train_data_" + str(number) + ".csv", "w+")
        augmented_string = ""
        for sen, sen_list in data_augmentation_dictionary.items():
            #print(sen)
            augmented_string = augmented_string + sen + "\n"
            if sen_list:
                for s in sen_list:
                    augmented_string = augmented_string + s + "\n"
        print("File ", "augmented_mask_train_data_" + str(number) + ".csv", " created.")
        augmented_file.write(augmented_string)
        augmented_file.close()

#Main
#Get the data
data_file = importData()
print("Getting the data.")
#Separate sentences from labels
sentences, labels = getDataValues(data_file)
print("Separating sentences from labels.")
#   Create five new sentences for each sentence in the data
number_list = [1, 3, 5, 10]
print("Augmenting data.")
augment_data(number_list, sentences, labels)
print("Success.")
