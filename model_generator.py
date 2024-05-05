# model_generator.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name='gpt-2', max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = input_ids.ne(1).int()  # Generate attention mask (ignore padding)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
