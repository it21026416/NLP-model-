from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Use the correct token and model identifier
token = "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"
model_identifier = "gpt2"  # or the specific variant like "gpt2"

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_identifier, use_auth_token=token)
    model = GPT2LMHeadModel.from_pretrained(model_identifier, use_auth_token=token)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load model or tokenizer: {e}")
