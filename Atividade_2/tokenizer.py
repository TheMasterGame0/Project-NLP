#import re

# Functions to create Vocabulary for tokenizer
def countPairs(ids):
  dic = {}
  for pair in zip (ids, ids[1:]):
    # Increase the number of repeats of some pair in text
    # The .get take back the value on dict ou return the default value (0)  
    dic[pair] = dic.get(pair, 0) + 1
  return dic

def update(ids, pair, newID):
  """
    Take some list of ids, look for all occurrences of the pair and replace by newID
  """
  newIds = []
  i = 0
  size = len(ids)
  while i < size:
    if (i < size - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]):
      newIds.append(newID)
      i += 2
    else:
      newIds.append(ids[i])
      i += 1
  return newIds

def makeVocab(vocabSize, ids, newIdStart = 0, merges = {}, internalPrints = False):
  """
    Tokenize the given array to match vocabSize.
  """ 
  numberUpdates = vocabSize - 258 # 256 UTF-8 more 2 special characters

  # merges: Register tokens creation. newId : (oldId, oldId).    

  try:
    for i in range(numberUpdates):
      stats = countPairs(ids)
      # Returns the top pair (the one with more repeats)
      topPair = max(stats, key=stats.get)
      
      #Verify the existence of repeats in merges
      if topPair not in merges:
        newId = 258 + i + newIdStart
        merges[str(topPair)] = newId
      else: 
        newId = merges[str(topPair)]
        
      # New list of tokens
      ids = update(ids, topPair, newId)
      # Register
      if (internalPrints == True):
        print("Updated pair ("+str(topPair[0])+", "+str(topPair[1])+") to id " + str(newId))
    
    return merges, ids
  except: 
    return  merges, ids
# Decoding and encoding text
def decoder(ids, vocab):
  """
    Take a list of integers and give back a Python string
  """
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encoder(text, merges, isID = False):
  if isID:
    tokens = text
  else:
    # Take the text and convert to raw bytes tokens
    tokens = list(map(int, text.encode("utf-8")))

  while len(tokens) >=2:
    stats = countPairs(tokens)
    # Takes the pair with smallest id from merge dictionary
    pair = min(stats, key= lambda p: merges.get(p, float("inf")))
    if str(pair) not in merges:
      break
    newId = merges[str(pair)]
    tokens = update(tokens, pair, newId)

  return tokens


# import os, json

# path = 'Atividade_1/corpus/'
# files = [file for file in os.listdir(path) if file.endswith('.json')]

# merges = {}
# # 256 =  <S>
# # 257 =  <E>


# # Loop to take all the text in training 
# for i in range(len(files)): 
#   # Open the JSON file
#   f = open(path + files[i], "r")
#   data = json.load(f)["text"]
#   f.close()

#   # Dividing the sentences 
#   sentences = re.split("[?!.]\s", data)

#   for sentence in sentences:
#     # Making Vocab
#     tokens = [256] + list(map(int, sentence.encode("utf-8"))) + [257]

#     if merges:
#       merge, ids = makeVocab(260, tokens, merges[list(merges)[-1]] - 257, merges)
#     else: 
#       merge, ids = makeVocab(260, tokens)

#     merges = {**merge, **merges}
    

#   print("Done: ", i ," files")


# # Save merges on .txt
# f = open("Atividade_2/merges.txt", "w") # Will always erase and write everything again.
# # Converte merges to string.
# string = ""
# for key in merges.keys():
#   string += str(key) + ": " + str(merges[key]) + ";"
# f.write(string)
# f.close()
