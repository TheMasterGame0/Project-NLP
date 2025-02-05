import BERT
import torch
from transformers import BertForSequenceClassification
device = torch.device('cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(BERT.labels),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_0.model', map_location=torch.device('cpu')))

_, predictions, true_vals = BERT.evaluate(BERT.dataloader_test, device, BERT.dataloader_test)
BERT.accuracy_per_class(predictions, true_vals)