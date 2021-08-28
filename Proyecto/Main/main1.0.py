import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing import sequence

filepath_dict = {'sst2': '/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/augmented_data_500.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['label', 'sentences'], sep='\t')
    df['source'] = source # Add another column filled with the source name
    df_list.append(df)
df.info()
df = pd.concat(df_list)
print(df.iloc[0])
#   Split the data
from sklearn.model_selection import train_test_split

df_sst2 = df[df['source'] == 'sst2']
sentences = df_sst2['sentences'].values
#print("Sentences: ", sentences)
y = df_sst2['label'].values

batch_size = 32
raw_train_ds = tf.keras.preprocessing.text.text_dataset_from_directory(
    "/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/augmented_data_500.txt",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "/Users/administrador/Documents/MaestriaINAOE/AprendizajeMaquinaII/Proyecto/Main/augmented_data_500.txt",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
print(
    "Number of batches in raw_train_ds: %d"
    % tf.data.experimental.cardinality(raw_train_ds)
)
print(
    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
)
