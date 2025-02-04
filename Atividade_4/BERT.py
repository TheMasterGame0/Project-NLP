import numpy as np
import pandas as p
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Possibilita a transformação de classes Alpha-numéricas em classes numéricas.

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW

# Create and save in 2 files
def create_Train_Eval_Test(data):
  """
    Create and save in 2 files the Training and Test group from a dataFrame
  """

  # Convert Labels to a number 
  conversor = LabelEncoder() 
  y = conversor.fit_transform(data.iloc[:, -1])
  data.insert(len(data.columns), 'Y', y)

  # split train dataset into train, validation and test sets
  train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['Y'], 
                                                                      random_state=2018, 
                                                                      test_size=0.3, 
                                                                      stratify=data['Y'])


  val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                  random_state=2018, 
                                                                  test_size=0.5, 
                                                                  stratify=temp_labels)

  # Save train on .txt
  f = open("Atividade_4/train.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(train_text, index=False))
  f.close()

  # Save test on .txt
  f = open("Atividade_4/test.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(test_text, index=False))
  f.close()

def tokenizar(dado,tokenizer):
  # tokenize and encode sequences
  tokens = tokenizer.batch_encode_plus(
      dado.tolist(),
      max_length = 30,
      pad_to_max_length=True,
      truncation=True
  )

def makeTensor(arquivo):
  f = open(arquivo, "r")
  temp = p.read_csv(f)
  X, Y = temp.iloc[:, -3], temp.iloc[:, -1]
  f.close()

  tokens = tokenizar(X,tokenizer)
  sequence = torch.tensor(tokens['input_ids'])
  mask = torch.tensor(tokens['attention_mask'])
  y = torch.tensor(Y.tolist())

  # wrap tensors
  train = TensorDataset(sequence, mask, y)
  # dataloader
  dataloader = DataLoader(train, sampler=RandomSampler(train), batch_size=16)

  # class weights
  class_weights = compute_class_weight('balanced', np.unique(Y), Y)
  weights= torch.tensor(class_weights, dtype=torch.float)
  weights = weights.to(device)

  cross_entropy = nn.NLLLoss(weight=weights) 

class BERTmodel(nn.Module):
  def __init__(self, bert):
    super(BERTmodel, self).__init__()
    
    self.bert = bert 
    # dropout layer
    self.dropout = nn.Dropout(0.1)
    # relu activation function
    self.relu =  nn.ReLU()
    # dense layer 1
    self.fc1 = nn.Linear(768,512)
    # dense layer 2 (Output layer)
    self.fc2 = nn.Linear(512,2)
    #softmax activation function
    self.softmax = nn.LogSoftmax(dim=1)

  #define the forward pass
  def forward(self, sent_id, mask):
    #pass the inputs to the model  
    _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
  
    x = self.fc1(cls_hs)
    x = self.relu(x)
    x = self.dropout(x)
    # output layer
    x = self.fc2(x)
    # apply softmax activation
    x = self.softmax(x)

    return x

# function to train the model
def train(dataloader, cross_entropy):
    print("Training...")
    model.train()
    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step,batch in enumerate(dataloader):
        
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(dataloader)
  
      # predictions are in the form of (no. of batches, size of batch, no. of classes).
      # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate(dataloader):
    
    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    
    # empty list to save the model predictions
    total_preds = []

    for step,batch in enumerate(dataloader):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

# data = p.read_csv("Atividade_3/Dmoz-Computers.csv")
# create_Train_Test(data)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

makeTensor("Atividade_4/train.csv")

# makeTensor("Atividade_4/test.csv")

# import BERT-base pretrained model
# specify GPU
device = torch.device("cuda")
bert = AutoModel.from_pretrained('bert-base-uncased')
for param in bert.parameters():
    param.requires_grad = False
# Bert pre treinado
model = BERTmodel(bert)
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(),lr = 1e-5)


# set initial loss to infinite
best_valid_loss = float('inf')

epochs = 5

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
