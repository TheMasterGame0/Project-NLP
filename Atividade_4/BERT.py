import numpy as np
import pandas as p
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder #Possibilita a transformação de classes Alpha-numéricas em classes numéricas.

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm.notebook import tqdm

# Create and save in 2 files
def create_Train_Test(data):
  """
    Create and save in 2 files the Training and Test group from a dataFrame
  """

  # Convert Labels to a number 
  conversor = LabelEncoder() 
  y = conversor.fit_transform(data.iloc[:, -1])
  data.insert(len(data.columns), 'Y', y)

  # split train dataset into train, validation and test sets
  X_train, X_val, y_train, y_val = train_test_split(data.text.values, 
                                                  data.Y.values, 
                                                  test_size=0.3, 
                                                  stratify=data.label.values)
  # Save train on .txt
  df = {
    'text': X_train,
    'label': y_train,
  }
  f = open("train.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(p.DataFrame(df), index=False))
  f.close()
  
  # Save test on .txt
  df = {
    'text': X_val,
    'label': y_val,
  }
  f = open("test.csv", "w") # Will always erase and write everything again.
  f.write(p.DataFrame.to_csv(p.DataFrame(df), index=False))
  f.close()

  return y

def tokenizar(dado,tokenizer):
  # tokenize and encode sequences
  tokens = tokenizer.batch_encode_plus(
      dado.tolist(),
      add_special_tokens=True, 
      return_attention_mask=True, 
      pad_to_max_length=True, 
      max_length=30, 
      return_tensors='pt'
  )

  return tokens

def makeTensor(arquivo, tokenizer):
  f = open(arquivo, "r")
  temp = p.read_csv(f)
  X, Y = temp.iloc[:, 0], temp.iloc[:, 1]
  f.close()

  tokens = tokenizar(X,tokenizer)
  sequence = tokens['input_ids']
  mask = tokens['attention_mask']
  y = torch.tensor(Y.tolist())

  # wrap tensors
  train = TensorDataset(sequence, mask, y)
  # dataloader
  dataloader = DataLoader(train, sampler=RandomSampler(train), batch_size=16) 

  return dataloader

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluate(model, device, dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


device = torch.device('cpu')
data = p.read_csv("Dmoz-Computers.csv")
labels = create_Train_Test(data)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

dataloader_train = makeTensor("train.csv",tokenizer)
dataloader_test = makeTensor("test.csv",tokenizer)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(labels),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 1

scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# for epoch in tqdm(range(1, epochs+1)):
    
#     model.train()
    
#     loss_train_total = 0

#     progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
#     for batch in progress_bar:

#         model.zero_grad()
        
#         batch = tuple(b.to(device) for b in batch)
        
#         inputs = {'input_ids':      batch[0],
#                   'attention_mask': batch[1],
#                   'labels':         batch[2],
#                  }       

#         outputs = model(**inputs)
        
#         loss = outputs[0]
#         loss_train_total += loss.item()
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()
#         scheduler.step()
        
#         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
