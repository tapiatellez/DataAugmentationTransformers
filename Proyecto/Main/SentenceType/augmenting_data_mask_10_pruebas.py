from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
from nltk.tokenize.treebank import TreebankWordDetokenizer

sequence = f"Distilled models are {tokenizer.mask_token} than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
sequenceB = f"I'm an honest person, or at least I thought I was."
tokenized_sentence = word_tokenize(sequenceB)
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

for token in top_10_tokens:
    print("Token: ", tokenizer.decode([token]))

    print(untokenized_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))


# top_10_tokens = torch.topk(mask)
# rns = random.sample(range(0, len(sequenceB)), 2)
# maskB = tokenizer.mask_token
# inputB = tokenizer.encode(sequenceB)
# print("InputB: ", inputB)


# input = tokenizer.encode(sequence, return_tensors="pt")
# mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
#
# token_logits = model(input)[0]
# mask_token_logits = token_logits[0, mask_token_index, :]
#
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
# top_10_tokens = torch.topk(mask_token_logits, 10, dim = 1).indices[0].tolist()
# #for token in top_5_tokens:
#     #print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
# print("---------------------------------------")
# print("Original sequence: ")
# print(sequence)
# for token in top_10_tokens:
#     print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
