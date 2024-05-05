import torch

from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

def load_models():
    try:
        # Load the classifier model (assuming BERT for classification)
        tokenizer_classify = BertTokenizer.from_pretrained('bert-base-uncased')
        # Check if tokenizer has a pad token, if not, set it to eos_token
        if tokenizer_classify.pad_token is None:
            tokenizer_classify.add_special_tokens({'pad_token': '[PAD]'})
        
        model_classify = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        # Load the generation model (assuming GPT-2 for policy generation)
        tokenizer_generate = GPT2Tokenizer.from_pretrained('gpt2')
        # Similar check for the generation tokenizer
        if tokenizer_generate.pad_token is None:
            tokenizer_generate.add_special_tokens({'pad_token': '[PAD]'})
        
        model_generate = GPT2LMHeadModel.from_pretrained('gpt2')
        
        return tokenizer_classify, model_classify, tokenizer_generate, model_generate
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None, None, None, None



def classify_text(text, tokenizer, model):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, axis=1)
    categories = {0: "Data Leakage Prevention", 1: "Browser Customization", 2: "Threat Prevention"}  # Example categories
    return categories[prediction.item()]

def generate_text(prompt, tokenizer, model):
    # Encode the prompt to tensor of tokens
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Ensure the input does not exceed model limits
    input_ids = inputs['input_ids'][:, :model.config.max_position_embeddings]  # Limit input size to model's maximum length

    # Generate attention mask based on actual tokens (excluding padding)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    try:
        # Generate text using the model, specifying max_new_tokens
        max_base_length = input_ids.size(1)
        max_total_length = min(1024, max_base_length + 50)
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=max_total_length,  # Increase the max_length if the input is already close to 512
            max_new_tokens=50,  # Alternatively, specify how many new tokens to generate
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in text generation: {e}")
        return "Error generating text."

# Now the function includes a check to increase the max_length and also uses max_new_tokens to define a clear generation boundary.

def main():
    tokenizer_classify, model_classify, tokenizer_generate, model_generate = load_models()
    if None in [tokenizer_classify, model_classify, tokenizer_generate, model_generate]:
        print("Model loading failed. Exiting.")
        return

    while True:
        user_input = input("Enter your security policy requirement (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        category = classify_text(user_input, tokenizer_classify, model_classify)
        prompt = f"Generate a detailed security policy for {category}:"
        policy = generate_text(prompt, tokenizer_generate, model_generate)
        print(f"Suggested Policy for {category}:\n{policy}\n")

if __name__ == "__main__":
    main()
