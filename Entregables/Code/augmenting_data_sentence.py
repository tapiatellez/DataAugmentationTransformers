import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

import pandas as pd
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
def createAugmentedList(sentences, labels):
    augmented_list = []
    for sentence, label in zip(sentences, labels):
        augmented_list.append(sentence + "  " + label)
        input_ids = tokenizer.encode(sentences[0], return_tensors = 'tf')
        greedy_output = model.generate(input_ids, max_length = len(sentences[0]))
        # set seed to reproduce results. Feel free to change the seed though to get different results
        tf.random.set_seed(0)

        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        sample_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        print("Output:\n" + 100 * '-')
        for i, sample_output in enumerate(sample_outputs):
          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        augmented_list.append(tokenizer.decode(sample_outputs[0], skip_special_tokens = True) + "   " + label)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')
# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

#Main
#Get the data
data_file = importData()
#Separate sentences from labels
sentences, labels = getDataValues(data_file)
print("Original text: ", sentences[0])

#   Create a new sentence
augmented_file = open("augmented_sentence_train_data.csv", "w+")
augmented_string = ""
counter = 1
for sen, label in zip(sentences, labels):
    print("Sen: ", sen)
    input_ids = tokenizer.encode(sen, return_tensors = 'tf')
    greedy_output = model.generate(input_ids, max_length = len(sen))
    # set seed to reproduce results. Feel free to change the seed though to get different results
    tf.random.set_seed(0)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    print("Original sentence: ", sen + "   " + str(label))
    created = tokenizer.decode(sample_outputs[0], skip_special_tokens = True)
    created = created.replace("\n", " ")
    print("Created sentence: ", created + "   " + str(label))
    original = sen + "\t" + str(label)
    created = created + "\t" + str(label)
    augmented_string = augmented_string + original + "\n"
    augmented_string = augmented_string + created + "\n"
    print("Finished creating sentence number: ", counter)
    counter += 1
augmented_file.write(augmented_string)
augmented_file.close()
print(augmented_string)
