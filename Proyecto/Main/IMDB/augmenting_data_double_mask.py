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
#   Split the data
from sklearn.model_selection import train_test_split
#   The following function creates a new sentence by multimasking the original
#   sentence.
def create_sentence_by_multi_masking_one(sentences, labels):
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        print("Label: ", label)
        print("Sentence: ", sentence)
        if len(tokenized_sentence) > 1:
            maskA = tokenizer.mask_token
            maskB = tokenizer.mask_token
            print("Length of sentence: ", len(tokenized_sentence))
            rns = random.sample(range(0, len(tokenized_sentence)), 2)
            print("Random numbers: ", rns)
            sentenceA = word_tokenize(sentence)
            sentenceA[rns[0]] = maskA
            print("Tokenized sentenceA with mask: ", sentenceA)
            sentenceB = word_tokenize(sentence)
            sentenceB[rns[1]] = maskB
            print("Tokenized sentenceB with mask: ", sentenceB)
            print("Original tokenized sentence: ", word_tokenize(sentence))
            untokenized_sentenceA = TreebankWordDetokenizer().detokenize(sentenceA)
            print("Untokenized sentenceA with mask: ", untokenized_sentenceA)
            untokenized_sentenceB = TreebankWordDetokenizer().detokenize(sentenceB)
            print("Untokenized sentenceB with mask: ", untokenized_sentenceB)

            inputA = tokenizer.encode(untokenized_sentenceA, return_tensors = "pt")
            inputB = tokenizer.encode(untokenized_sentenceB, return_tensors = "pt")
            mask_token_indexA = torch.where(inputA == tokenizer.mask_token_id)[1]
            mask_token_indexB = torch.where(inputB == tokenizer.mask_token_id)[1]

            token_logitsA = model(inputA)[0]
            token_logitsB = model(inputB)[0]
            mask_token_logitsA = token_logitsA[0, mask_token_indexA, :]
            mask_token_logitsB = token_logitsB[0, mask_token_indexB, :]

            tokensA = torch.topk(mask_token_logitsA, 1, dim = 1).indices[0].tolist()
            print("TokensA: ", tokenizer.decode([tokensA[0]]))
            tokensB = torch.topk(mask_token_logitsB, 1, dim = 1).indices[0].tolist()
            print("TokensB: ", tokenizer.decode([tokensB[0]]))
            predicted_sentence = tokenized_sentence
            predicted_sentence[rns[0]] = tokenizer.decode([tokensA[0]])
            predicted_sentence[rns[1]] = tokenizer.decode([tokensB[0]])
            print("Original sentence: ", sentence)
            print("Predicted sentence: ", TreebankWordDetokenizer().detokenize(predicted_sentence))
            sentences_list = []
            sen = TreebankWordDetokenizer().detokenize(predicted_sentence)
            sen = sen + "\t" + str(label)
            sentences_list.append(sen)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    return predicted_sentences_dict
#   The following function creates a new sentence by multimasking the original
#   sentence.
def create_sentence_by_multi_masking_three(sentences, labels):
    predicted_sentences_dict = {}
    for sentence, label in zip(sentences, labels):
        tokenized_sentence = word_tokenize(sentence)
        print("Label: ", label)
        print("Sentence: ", sentence)
        if len(tokenized_sentence) > 1:
            maskA = tokenizer.mask_token
            maskB = tokenizer.mask_token
            print("Length of sentence: ", len(tokenized_sentence))
            rns = random.sample(range(0, len(tokenized_sentence)), 2)
            print("Random numbers: ", rns)
            sentenceA = word_tokenize(sentence)
            sentenceA[rns[0]] = maskA
            print("Tokenized sentenceA with mask: ", sentenceA)
            sentenceB = word_tokenize(sentence)
            sentenceB[rns[1]] = maskB
            print("Tokenized sentenceB with mask: ", sentenceB)
            print("Original tokenized sentence: ", word_tokenize(sentence))
            untokenized_sentenceA = TreebankWordDetokenizer().detokenize(sentenceA)
            print("Untokenized sentenceA with mask: ", untokenized_sentenceA)
            untokenized_sentenceB = TreebankWordDetokenizer().detokenize(sentenceB)
            print("Untokenized sentenceB with mask: ", untokenized_sentenceB)

            inputA = tokenizer.encode(untokenized_sentenceA, return_tensors = "pt")
            inputB = tokenizer.encode(untokenized_sentenceB, return_tensors = "pt")
            mask_token_indexA = torch.where(inputA == tokenizer.mask_token_id)[1]
            mask_token_indexB = torch.where(inputB == tokenizer.mask_token_id)[1]

            token_logitsA = model(inputA)[0]
            token_logitsB = model(inputB)[0]
            mask_token_logitsA = token_logitsA[0, mask_token_indexA, :]
            mask_token_logitsB = token_logitsB[0, mask_token_indexB, :]

            tokensA = torch.topk(mask_token_logitsA, 10, dim = 1).indices[0].tolist()
            print("TokensA: ", tokenizer.decode([tokensA[0]]))
            tokensB = torch.topk(mask_token_logitsB, 10, dim = 1).indices[0].tolist()
            print("TokensB: ", tokenizer.decode([tokensB[0]]))
            sentences_list = create_sentences_list_for_n(tokensA, tokensB, word_tokenize(sentence), label, rns)
            predicted_sentences_dict[sentence + "\t" + str(label)] = sentences_list
        else:
            predicted_sentences_dict[sentence + "\t" + str(label)] = None
    return predicted_sentences_dict
def create_sentences_list_for_n(tokensA, tokensB, original, label, randoms):
    sen_list = []
    for tokenA, tokenB in zip(tokensA, tokensB):
        tokenized_sentence = original
        tokenized_sentence[randoms[0]] = tokenizer.decode([tokenA])
        tokenized_sentence[randoms[1]] = tokenizer.decode([tokenB])
        sen = TreebankWordDetokenizer().detokenize(tokenized_sentence)
        sen = sen + "\t" + str(label)
        sen_list.append(sen)
    return sen_list

#Main
#Get the data
data_file = importData()
#Separate sentences from labels
sentences, labels = getDataValues(data_file)
print(sentences)
#   Create five new sentences for each sentence in the data
data_augmentation_dictionary = create_sentence_by_multi_masking_three(sentences, labels)
#   Create file with augmented data.
augmented_file = open("augmented_data_double_masking_ten.csv", "w+")
augmented_string = ""
for sen, sen_list in data_augmentation_dictionary.items():
    #print(sen)
    augmented_string = augmented_string + sen + "\n"
    if sen_list:
        for s in sen_list:
            augmented_string = augmented_string + s + "\n"

augmented_file.write(augmented_string)
augmented_file.close()

#create_sentence_by_multi_masking(sentences, labels)
#print(augmented_string)
