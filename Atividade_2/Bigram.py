import tokenizer as t
import os, json
import re

# Functions to create the Bigram tuples ( Very similar to the Vocabulary for tokenizer).
def bigramPairs(ids):
  dic = {}
  sentence = ids
  for pair in zip (sentence, sentence[1:]):
    # Increase the number of repeats of some pair in text
    # The .get take back the value on dict ou return the default value (0)  
    dic[pair] = dic.get(pair, 0) + 1
    
  if dic != {}:
    temp = sorted(dic.items(), key = lambda x: -x[1])[0]
    return temp[0]
  else:
    return (256, 257)

def buildBigram(merges):
  bigram = {}
  path = 'Atividade_1/corpus/'
  files = [file for file in os.listdir(path) if file.endswith('.json')]

  # Loop to take all the text in training 
  for i in range(len(files) - 2000): 
    # Open the JSON file
    f = open(path + files[i], "r")
    data = json.load(f)["text"]
    f.close()

    # Dividing the sentences 
    sentences = re.split("[?!.]\s", data)

    for sentence in sentences:
      # Making Bigram pairs
      pars = bigramPairs(t.encoder([256] + list(map(int, sentence.encode("utf-8"))) + [257], merges, True))

      bigram[pars] = bigram.get(pars, 0) + 1
  
    print("Done: ", i ," files")
  return bigram

# Take the merges from .txt
f = open("Atividade_2/merges.txt", "r")
stringMerges = f.read().split(";")
f.close()

merges = {}
for m in stringMerges:
  lista = m.split(":")
  if (lista != ['']):
    chave, valor = lista[0], lista[1]
    merges[chave] = int(valor)

bigram = buildBigram(merges)

# Save bigram on .txt
f = open("Atividade_2/bigram.txt", "w") # Will always erase and write everything again.
# Converte bigram to string.
string = ""
for key, valor in sorted(bigram.items(), key = lambda x: -x[1]):
  string += str(key) + ": " + str(valor) + ";"
f.write(string)
f.close()
    


