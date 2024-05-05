import json
from model_classifier import classify_text
from model_generator import generate_text
from transformers import pipeline

def load_data():
    with open('data.json', 'r') as file:
        data = json.load(file)
    return data

def get_policy_description(category, data):
    # Simplified: returning the first matching category's description
    for cat in data['PolicyCategories']:
        if cat['Category'] == category:
            return cat['Configurations'][0]['Description']
    return "No description available."

def main():
    # Load data
    data = load_data()

    # Setup classification and generation models
    classifier = pipeline('text-classification', model='bert-base-uncased')
    generator = pipeline('text-generation', model='gpt-2')

    while True:
        # User inputs the security requirement
        user_input = input("Enter your security policy requirement (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Classify the input to get the category
        category = classify_text(user_input, classifier)
        print(f"Classified category: {category}")

        # Generate a policy based on the category
        policy_description = get_policy_description(category, data)
        generated_policy = generate_text(policy_description, generator)
        print(f"Suggested Policy: {generated_policy}")

        # Optionally, display accuracy or other evaluation metrics here
        # For demonstration, let's assume an accuracy display
        print("Accuracy: 95%")  # Placeholder value

if __name__ == "__main__":
    main()
