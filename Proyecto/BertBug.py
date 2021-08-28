import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

model = BertForMaskedLM.from_pretrained('bert-large-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
ARTICLE_TO_SUMMARIZE = "My friends are <mask> but they eat too many carbs."
inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], return_tensors='pt')
input_ids = inputs['input_ids']
#generated_ids = model(, attention_mask=inputs['attention_mask'])[0]
logits = model(input_ids)[0]
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(10)
print(tokenizer.decode(predictions).split())
