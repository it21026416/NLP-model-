import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import to_categorical
from transformers import BertTokenizer
import os



file_path = os.path.join('C:', os.sep, 'Users', 'ASUS', 'Desktop', 'NLP', 'data.json')


def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    records = []
    for category in data['PolicyCategories']:
        for config in category['Configurations']:
            record = {
                'Description': config.get('Description', 'No description available'),
                'Category': category['Category'],
            }
            records.append(record)
    df = pd.DataFrame(records)

    # Encode the categories numerically
    label_encoder = LabelEncoder()
    df['CategoryEncoded'] = label_encoder.fit_transform(df['Category'])
    num_labels = len(df['CategoryEncoded'].unique())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['CategoryEncoded'], test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_labels)
    y_test = to_categorical(y_test, num_classes=num_labels)

    return X_train, X_test, y_train, y_test, num_labels

# Function to tokenize and create TensorFlow datasets
def tokenize_and_create_datasets(X_train, X_test, y_train, y_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the training and testing set
    train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=128, return_tensors="tf")
    test_encodings = tokenizer(X_test.to_list(), truncation=True, padding=True, max_length=128, return_tensors="tf")
    
    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    }, y_train)).batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask']
    }, y_test)).batch(32)

    return train_dataset, test_dataset


 
# Load and preprocess data
X_train, X_test, y_train, y_test, num_labels = load_and_preprocess_data(file_path)

# Tokenize and create datasets
train_dataset, test_dataset = tokenize_and_create_datasets(X_train, X_test, y_train, y_test)
