import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "llm_trained_model")

# Step 1: Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# Step 2: Create a function for inference
def generate_response(instruction, user_input=""):
    if user_input:
        prompt = f"Instruction: {instruction}\nInput: {user_input}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 3: Test with some examples
if __name__ == "__main__":
    # Example 1
    print("Example 1:")
    response = generate_response("Summarize this text.", 
                                 "AI is transforming industries like healthcare and finance, but it also raises ethical questions.")
    print(response, "\n")

    # Example 2
    print("Example 2:")
    response = generate_response("Translate into French.", "The future of AI is bright.")
    print(response, "\n")

    # Example 3
    print("Example 3:")
    response = generate_response("Write Python code to calculate factorial of a number.")
    print(response, "\n")
