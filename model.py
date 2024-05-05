##Description of this file for audit purpose - This file defines the model architecture. Since im using a pre-trained BERT model from the transformers library, this will be quite straightforward.

from transformers import TFBertForSequenceClassification

def create_model(num_labels):
    # Ensure that you're using the TensorFlow-specific class
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

