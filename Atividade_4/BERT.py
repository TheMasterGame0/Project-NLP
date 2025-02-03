import numpy as np
import pandas as p
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Possibilita a transformação de classes Alpha-numéricas em classes numéricas.
import matplotlib.pyplot as plt

# Create and save in 2 files
def create_Train_Test(data):
  """
    Create and save in 2 files the Training and Test group from a dataFrame
  """

  # Convert Labels to a number 
  conversor = LabelEncoder() 
  y = conversor.fit_transform(data.iloc[:, -1])
  data.insert(len(data.columns), 'Y', y)
  
  # Divide data in to groups
  train, test = train_test_split(data, test_size=0.3)

  # Save train on .txt
  f = open("Atividade_4/train.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(train, index=False))
  f.close()

  # Save test on .txt
  f = open("Atividade_4/test.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(test, index=False))
  f.close()

def tokenizar(dado,tokenizer):
  # tokenize and encode sequences
  tokens = tokenizer.batch_encode_plus(
      dado.tolist(),
      max_length = 30,
      pad_to_max_length=True,
      truncation=True
  )


data = p.read_csv("Atividade_3/Dmoz-Computers.csv")
create_Train_Test(data)

# import BERT-base pretrained model
# bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Take the training file
f = open("Atividade_4/train.csv", "r")
temp = p.read_csv(f)
trainX, trainY = temp.iloc[:, -3], temp.iloc[:, -1]
f.close()

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in trainX]

print(seq_len)
plt.hist(p.Series(seq_len))
plt.show()