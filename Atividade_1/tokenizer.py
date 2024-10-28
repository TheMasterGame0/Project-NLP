# Take te string to be tokenized
text = "Some random string to be used as an example to be tokenized. To have the best use of the tokenizer and have a good result, it is important to have some long enough text to find some pattern to be followed. This example is not good but it is some start to understand the proposal and the ideas behind the code being done.\nExpanding the text!\nOn the first try, this text was long, but the number of tokens was exactly equal to the number of Unicode bytes, for this reason was increased a little bit the size of the text (probably will not change the equality of the sizes but will provide a better result in the end)."

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

def makeVocab(vocabSize, ids, internalPrints = False):
  """
    Tokenize the given array to match vocabSize.
  """ 
  numberUpdates = vocabSize - 256
  # Register tokens creation. newId : (oldId, oldId).
  merges = {}
  for i in range(numberUpdates):
    stats = countPairs(ids)
    # Returns the top pair (the one with more repeats)
    topPair = max(stats, key=stats.get)
    newId = 256 + i
    # New list of tokens
    ids = update(ids, topPair, newId)
    # Register
    if (internalPrints == True):
      print("Updated pair ("+str(topPair[0])+", "+str(topPair[1])+") to id " + str(newId))

    merges[topPair] = newId
  
  return merges, ids

# Decoding and encoding text
def decoder(ids, vocab):
  """
    Take a list of integers and give back a Python string
  """
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encoder(text, merges):
  # Take the text and convert to raw bytes tokens
  tokens = list(map(int, text.encode("utf-8")))
  while len(tokens) >=2:
    stats = countPairs(ids)
    # Takes the pair with smallest id from merge dictionary
    pair = min(stats, key= lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break
    
    newId = merges[pair]
    tokens = update(tokens, pair, newId)

  return tokens

# Making Vocab
tokens = list(map(int, text.encode("utf-8")))
merges, ids = makeVocab(276, tokens)
# Creating dictionary with all vocab
vocab = {idx: bytes([idx]) for idx in range(256)}
# This part only works because we have sure that merges indexes are in increasing order
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]

# print(decoder(encoder(text, merges), vocab))

# Make a list going from the pair with most repeats to the least.
# print(sorted(((value, pair) for pair,value in stats.items()), reverse=True))

# Results
print("Text length: ", len(text))
print("Initial tokens length: ", len(tokens))
print("Final tokens length: ", len(ids))
print(f"Compression rate: {len(tokens)/len(ids):.2f}X")
