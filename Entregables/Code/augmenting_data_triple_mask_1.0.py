from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
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

#   The following function receives a list of sentences, labels and the amount
#   of sentences to create. It uses BERT to return, by multi-masking a dictionary
#   with the following structure: {base-sentence : list-of-sentences-multimasked}.
def create_sentence_by_multi_masking_three(sentences, labels, amount):
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        #print("Label: ", label)
        #print("Sentence: ", sentence)
        if len(tokenized_sentence) > 2:
            maskA = tokenizer.mask_token
            maskB = tokenizer.mask_token
            maskC = tokenizer.mask_token
            #print("Length of sentence: ", len(tokenized_sentence))
            rns = random.sample(range(0, len(tokenized_sentence)), 3)
            #print("Random numbers: ", rns)

            sentenceA = word_tokenize(sentence)
            sentenceA[rns[0]] = maskA
            #print("Tokenized sentenceA with mask: ", sentenceA)
            sentenceB = word_tokenize(sentence)
            sentenceB[rns[1]] = maskB
            #print("Tokenized sentenceB with mask: ", sentenceB)
            sentenceC = word_tokenize(sentence)
            sentenceC[rns[2]] = maskC
            #print("Tokenized sentenceC with mask: ", sentenceC)

            untokenized_sentenceA = TreebankWordDetokenizer().detokenize(sentenceA)
            #print("Untokenized sentenceA with mask: ", untokenized_sentenceA)
            untokenized_sentenceB = TreebankWordDetokenizer().detokenize(sentenceB)
            #print("Untokenized sentenceB with mask: ", untokenized_sentenceB)
            untokenized_sentenceC = TreebankWordDetokenizer().detokenize(sentenceC)

            inputA = tokenizer.encode(untokenized_sentenceA, return_tensors = "pt")
            inputB = tokenizer.encode(untokenized_sentenceB, return_tensors = "pt")
            inputC = tokenizer.encode(untokenized_sentenceC, return_tensors = "pt")
            mask_token_indexA = torch.where(inputA == tokenizer.mask_token_id)[1]
            mask_token_indexB = torch.where(inputB == tokenizer.mask_token_id)[1]
            mask_token_indexC = torch.where(inputC == tokenizer.mask_token_id)[1]

            token_logitsA = model(inputA)[0]
            token_logitsB = model(inputB)[0]
            token_logitsC = model(inputC)[0]
            mask_token_logitsA = token_logitsA[0, mask_token_indexA, :]
            mask_token_logitsB = token_logitsB[0, mask_token_indexB, :]
            mask_token_logitsC = token_logitsC[0, mask_token_indexC, :]

            tokensA = torch.topk(mask_token_logitsA, amount, dim = 1).indices[0].tolist()
            #print("TokensA: ", tokenizer.decode([tokensA[0]]))
            tokensB = torch.topk(mask_token_logitsB, amount, dim = 1).indices[0].tolist()
            #print("TokensB: ", tokenizer.decode([tokensB[0]]))
            tokensC = torch.topk(mask_token_logitsC, amount, dim = 1).indices[0].tolist()
            #print("TokensC: ", tokenizer.decode([tokensC[0]]))
            sentences_list = create_sentences_list_for_n(tokensA, tokensB, tokensC, word_tokenize(sentence), label, rns)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    return predicted_sentences_dict

#   The following function receives lists of tokens, the original sentence, the
#   label of the sentence and a list of random numbers. It returns a list of the
#   of sentences, where each of the sentences has been filled with the token in
#   its respective random position.
def create_sentences_list_for_n(tokensA, tokensB, tokensC, original, label, randoms):
    sen_list = []
    for tokenA, tokenB, tokenC in zip(tokensA, tokensB, tokensC):
        tokenized_sentence = original
        tokenized_sentence[randoms[0]] = tokenizer.decode([tokenA])
        tokenized_sentence[randoms[1]] = tokenizer.decode([tokenB])
        tokenized_sentence[randoms[2]] = tokenizer.decode([tokenC])
        sen = TreebankWordDetokenizer().detokenize(tokenized_sentence)
        sen = sen + "\t" + str(label)
        sen_list.append(sen)
    return sen_list
#   The following function receives a list of numbers, sentences, and their
#   respective labels. It creates a file for each of the numbers in the numbers
#   list and the number of augmented sentences per sentence in the base data is
#   based on this number.
def augment_data(numbers_list, sentences, labels):
    for number in numbers_list:
        #print(sentences)
        #   Create five new sentences for each sentence in the data
        data_augmentation_dictionary = create_sentence_by_multi_masking_three(sentences, labels, number)
        #   Create file with augmented data.
        augmented_file = open("augmented_data_triple_masking_" + str(number)+ ".csv", "w+")
        augmented_string = ""
        for sen, sen_list in data_augmentation_dictionary.items():
            ##print(sen)
            augmented_string = augmented_string + sen + "\n"
            if sen_list:
                for s in sen_list:
                    augmented_string = augmented_string + s + "\n"
        print("File " +"augmented_data_triple_masking_" + str(number)+ ".csv" + " created.")
        augmented_file.write(augmented_string)
        augmented_file.close()
#Main
#   Get the data
print("Importing data...")
data_file = importData()
#   Separate sentences from labels
print("Separating sentences from labels...")
sentences, labels = getDataValues(data_file)
#   Create list of numbers
numbers_list = [1, 3, 5, 10]
#   Augment Data
print("Augmenting data...")
augment_data(numbers_list, sentences, labels)
print("Finish without errors. ")
