import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from random import randint
from nltk.tokenize.treebank import TreebankWordDetokenizer

#   The following function doesn't take any parameters. It reads a file by using
#   the pandas library and returns the sentences and labels obtained from the
#   file in a list.
def importData():
    filepath_dict = {'spam': 'train_data_file.csv'}
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
    df_sst2 = df[df['source'] == 'spam']
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
#   The following function receives a list of sentences and their respective
#   labels. It creates a file with the base data plus one augmented sentences
#   for each sentence y the base data.
def augment_data(sentences, labels):
    augmented_file = open("augmented_sentence_train_data.csv", "w+")
    augmented_string = ""
    counter = 1
    for sen, label in zip(sentences, labels):
        #print("Sen: ", sen)
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
        #print("Output:\n" + 100 * '-')
        #for i, sample_output in enumerate(sample_outputs):
          #print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

        #print("Original sentence: ", sen + "   " + str(label))
        created = tokenizer.decode(sample_outputs[0], skip_special_tokens = True)
        created = created.replace("\n", " ")
        #print("Created sentence: ", created + "   " + str(label))
        original = sen + "\t" + str(label)
        created = created + "\t" + str(label)
        augmented_string = augmented_string + original + "\n"
        augmented_string = augmented_string + created + "\n"
        #print("Finished creating sentence number: ", counter)
        counter += 1
    augmented_file.write(augmented_string)
    print("File augmented_sentence_train_data.csv created")
    augmented_file.close()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#Main
#Get the data
print("Importing data...")
data_file = importData()
#Separate sentences from labels
print("Separating sentences from labels...")
sentences, labels = getDataValues(data_file)
#   Create the augmented file
print("Augmenting data. ")
augment_data(sentences, labels)
print("Finished without errors.")
