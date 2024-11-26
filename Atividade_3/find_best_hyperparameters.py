import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Possibilita a transformação de classes Alpha-numéricas em classes numéricas.
import re
import math

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
  train, test = train_test_split(data, test_size=0.2)

  # Save train on .txt
  f = open("Atividade_3/train.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(train, index=False))
  f.close()

  # Save test on .txt
  f = open("Atividade_3/test.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(test, index=False))
  f.close()

def prepareParameters (data):
  # Create and save in 2 files
  create_Train_Test(data)

  # Take the training file
  f = open("Atividade_3/train.csv", "r")
  temp = p.read_csv(f)
  trainX, trainY = temp.iloc[:, :-2], temp.iloc[:, -1]
  f.close()

  # Naive Bayes Classifier Parameters
  nFiles = len(trainX)  
  classes = sorted(trainY.unique().tolist())

  dictClasses = {classes[i]: 0 for i in range(len(classes))}
  for i in trainY:
    dictClasses[i] += 1 
  
  # Probability of each class among the files
  probClasses = {k: v / nFiles for k, v in dictClasses.items()}
                  
  words_Classes_Dict = [{} for i in range(len(classes))]
  total_Words_Class = [0 for i in range(len(classes))]
  total_Word_tipes = set()

  # Get all the words of each class
  for c, text in zip(trainY, trainX.loc[:, "text"]):
    # Using set here to remove duplicates in the same text (trying to apply the same logic as the Binary Naive Bayes)
    words = set(re.split("\s|[^0-9][?!.,]+", re.sub("[()\":\[\]]", "", text.lower())))
    for word in words:
      if word !='':
        total_Words_Class[c] += 1   # Total words of each class
        words_Classes_Dict[c][word] = words_Classes_Dict[c].get(word, 0) + 1
        total_Word_tipes.add(word)  # Total words types

  # Words ordered by frequency
  for c in classes:
    words_Classes_Dict[c] = dict(sorted(words_Classes_Dict[c].items(), key=lambda item: item[1], reverse=True))
  
  # Probability of each word in each class
  prob_Words_Classes = [{} for i in range(len(classes))]
  for c in classes:
    # Add 1 at numerator and the total words types at denominator
    # This remove the chance of zero probability 
    for word in total_Word_tipes:
      prob_Words_Classes[c][word] = (words_Classes_Dict[c].get(word, 0) + 1) / (total_Words_Class[c] + len(total_Word_tipes))

  nbc = p.DataFrame(columns=["Word"]+classes)
  for word in total_Word_tipes:
    line = [word]
    for i in range(len(classes)):
      line.append(math.log(prob_Words_Classes[i][word]))
    # Add new line to dataframe with all the probabilities to this word
    nbc.loc[len(nbc.index)] = line
  
  # Add probability of classes on NBC
  line = ["probClasses"]
  for i in range(len(classes)):
    line.append(math.log(probClasses[i]))
  nbc.loc[len(nbc.index)] = line
  
  # Add total words of each classes on NBC
  line = ["totalWords"]
  for i in range(len(classes)):
    line.append(total_Words_Class[i])
  nbc.loc[len(nbc.index)] = line

  # Save NBC on .txt
  f = open("Atividade_3/NBC.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(nbc, index=False))
  f.close()

def Naive_Bayes_Classifier(frase):
  # Take the training file
  f = open("Atividade_3/NBC.csv", "r")
  nbc = p.read_csv(f)
  f.close()

  # Prepare the words
  words = set(re.split("\s|[^0-9][?!.,]+", re.sub("[()\":\[\]]", "", frase.lower())))

  # Get the probability of each class
  classes = nbc.columns.to_list()[1:]
  probClasses = nbc.iloc[-2].to_list()[1:]
  total_Words_Class = nbc.iloc[-1].to_list()[1:]

  # Get total Word types
  total_Word_tipes = nbc["Word"].to_list()[:-2]

  results = probClasses
  for word in words:
    if word == "" or word not in total_Word_tipes:
      for i in range(len(classes)):
        results[i] += 1/(total_Words_Class[i] + len(total_Word_tipes))
    else:
      for i in range(len(classes)):
        results[i] += nbc.query(f'Word == "{word}"').iat[0, i+1]
  return classes[results.index(max(results))]

# data = p.read_csv("Atividade_3/Dmoz-Computers.csv")
# prepareParameters(data)

# test_frase = "TCTMD A source of clinical and technical information and physician education in interventional cardiology and interventional vascular medicine." # Medicine

# print(Naive_Bayes_Classifier(test_frase))
