from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer

nlp = pipeline("fill-mask")
mask = nlp.tokenizer.mask_token
#print(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
sentenceBert = "[CLS] My name is albert I love the world. [SEP]"
sentence = "My name is albert I love the world."
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenized_sentence = tokenizer.tokenize(sentence)
#print(tokenized_sentence)
print(word_tokenize(sentence))
tokenized_sentence = word_tokenize(sentence)
mask = nlp.tokenizer.mask_token
print("Mask: ", mask)
print("Length of sentence: ", len(tokenized_sentence))
rn = randint(0, len(tokenized_sentence)-2)
print("Random number: ", rn)
print("Selected word: ", tokenized_sentence[rn])
tokenized_sentence[rn] = mask
print("Tokenized sentence with mask: ", tokenized_sentence)
untokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)
print("Untokenized sentence with mask: ", untokenized_sentence)
predicted_sentences = nlp(untokenized_sentence)
print(predicted_sentences[0]['sequence'])
sentences_dict = {}
counter = 1
for d in predicted_sentences:
    print(d)
    print(d['sequence'])
    sentence = d['sequence']
    sentence = sentence.replace('<s> ', '')
    sentence = sentence.replace('</s>', '')
    sentences_dict['Sentence' + str(counter)] = sentence
    counter += 1
print(sentences_dict)
