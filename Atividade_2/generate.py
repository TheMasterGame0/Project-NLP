import random
import tokenizer as t

# Take the bigram from .txt
f = open("bigram.txt", "r")
stringBigram = f.read().split(";")
f.close()

# Take the merges from .txt
f = open("merges.txt", "r")
stringMerges = f.read().split(";")
f.close()

bigram = {}
total = 0
for m in stringBigram:
  lista = m.split(":")
  if (lista != ['']):
    chave, valor = lista[0], lista[1]
    bigram[chave] = int(valor)
    total += int(valor)

# Crate dict with probability of each pair
# Limit to only pairs with more than 0.001% # Removed
newBigram = {}
for key in bigram.keys():
  value = bigram[key]/total
  newBigram[key] = value

# Get and merges
merges = {}
for m in stringMerges:
  lista = m.split(":")
  if (lista != ['']):
    chave, valor = lista[0], lista[1]
    merges[chave] = int(valor)

# Building Vocab to decode
vocab = {idx: bytes([idx]) for idx in range(256)}
# This part only works because we have sure that merges indexes are in increasing order
vocab[256] = bytes("<S>".encode("utf-8")) # "<S>"
vocab[257] = bytes("<E>".encode("utf-8")) # "<E>"
for pair, idx in merges.items():
  pair = pair.replace("(", "").replace(")", "").split(",")
  vocab[idx] = vocab[int(pair[0])] + vocab[int(pair[1])]


# Take some token in Bigram
def choseToken(token):
  choosenTokens = []
  hightChanceTokens = []

  for chave, valor in newBigram.items():
    if "("+token+"," in str(chave):
      
      choosenTokens.append(chave)
      if valor >= 0.10:
        hightChanceTokens.append(chave)
  
  if len(choosenTokens) > 0:
    if len(hightChanceTokens) > 0:
      return str(random.choice(hightChanceTokens))
    return str(random.choice(choosenTokens))
  
  for chave, valor in merges.items():
    if "("+token+"," in str(chave):
      for chaveB, valorB in newBigram.items():
        if "("+str(valor)+"," in str(chaveB):
          choosenTokens.append(chaveB)
          if valorB >= 0.10:
            hightChanceTokens.append(chaveB)

  if len(choosenTokens) > 0:
    if len(hightChanceTokens) > 0:
      return str(random.choice(hightChanceTokens))
    return str(random.choice(choosenTokens))


# initialPairs = {}
# for key in newBigram.keys():
#   for token in choosenTokens:
#     if str(token) in key:
#       initialPairs[key] = newBigram[key]
  