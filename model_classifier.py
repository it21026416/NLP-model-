# model_classifier.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def classify_text(text, model_path='path/to/classifier/model'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()  # Assuming binary classification for simplicity 
 