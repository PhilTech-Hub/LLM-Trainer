import gradio as gr
from transformers import pipeline

# Load fine-tuned model (make sure llm_trained_model/ is included in your repo or pushed to HF model hub)
generator = pipeline("text-generation", model="./llm_trained_model")

def chat(instruction, user_input=""):
    if user_input:
        prompt = f"Instruction: {instruction}\nInput: {user_input}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 LLM Trainer with JSON\nFine-tuned GPT-like model for custom instruction–response tasks")

    with gr.Row():
        instruction = gr.Textbox(label="Instruction", placeholder="e.g., Translate into French")
        user_input = gr.Textbox(label="Input (optional)", placeholder="e.g., Hello, how are you?")

    output = gr.Textbox(label="Model Response")

    btn = gr.Button("Generate Response")
    btn.click(fn=chat, inputs=[instruction, user_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
