📚 LLM Trainer with JSON 

This project shows how to train a small language model (LLM) on your own dataset written in JSON format.

We use the Hugging Face transformers library and the distilgpt2 model (a lightweight GPT-2 variant) to fine-tune the model.

The project is designed to help learners and junior engineers understand:

How to prepare custom datasets for LLMs.

How to train a model using Hugging Face.

How to run inference (make the model answer questions).

How to deploy the trained model for others to use.

🧠 What is this project?

Imagine you want an AI that can answer your company’s FAQs, or translate text, or act as a chatbot.
Instead of using GPT-4 directly, you can fine-tune a smaller model on your dataset.

🚀 **Live Demo on Hugging Face Spaces**:  
👉 [Try it here](https://huggingface.co/spaces/philemon-victor/llm-trainer-demo)  


Here, we created a dataset of instruction–response pairs (questions and answers) and trained the distilgpt2 model to follow them.

📂 Project Structure
LLM-Trainer/
│── datasets/
│   └── llm_training_dataset_100.json   # training data
│── scripts/
│   ├── train_llm_with_json.py          # training pipeline
│   ├── run_inference.py                # script for testing
│── requirements.txt                    # dependencies
│── README.md                           # documentation
│── .gitignore


datasets/ → contains your JSON dataset.

scripts/ → Python scripts for training & testing.

llm_trained_model/ → created after training; holds your fine-tuned model.

results/ and logs/ → training outputs.

📊 Dataset Format

The dataset is a list of JSON objects like this:

[
  {
    "id": "q1",
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "response": "Bonjour, comment ça va ?"
  },
  {
    "id": "q2",
    "instruction": "Summarize this paragraph",
    "input": "Artificial Intelligence is transforming industries...",
    "response": "AI is changing industries worldwide."
  }
]


instruction → tells the model what to do.

input → optional (can be empty).

response → expected answer.

⚙️ Installation

Follow these steps carefully if you’re new:

1️⃣ Clone the repository
git clone https://github.com/<your-username>/llm-trainer.git
cd llm-trainer

2️⃣ Create a virtual environment

This keeps project libraries separate from your system Python.

python -m venv .venv


Activate it:

On Windows:

.\.venv\Scripts\activate


On Mac/Linux:

source .venv/bin/activate


You should now see (.venv) before your terminal prompt.

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

🚀 Training the Model

Run the training script:

cd scripts
python train_llm_with_json.py


What happens:

Loads the dataset from datasets/llm_training_dataset_100.json.

Prepares it for the model.

Fine-tunes distilgpt2.

Saves the trained model inside llm_trained_model/.

Training may take a few minutes depending on your computer.

🧪 Running Inference (Testing the Model)

After training, test the model with your own input.
Create a file called run_inference.py like this:

from transformers import pipeline

# Load your fine-tuned model
generator = pipeline("text-generation", model="./llm_trained_model")

prompt = "Instruction: Translate to French\nInput: I love learning AI\nResponse:"
result = generator(prompt, max_length=100, num_return_sequences=1)

print("Model Output:", result[0]["generated_text"])


Run it:

python run_inference.py

🌐 Deployment Options
Option A: Hugging Face Spaces (easiest, free)

Create a free Hugging Face account
.

Create a new Space → choose Gradio app.

Upload:

app.py (Gradio UI file).

requirements.txt.

Your trained model.

This gives you a public web app like:

https://huggingface.co/spaces/<username>/llm-trainer

Option B: Run as an API (FastAPI)

Add a simple run_api.py:

from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="./llm_trained_model")

@app.post("/predict")
def predict(prompt: str):
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return {"output": result[0]["generated_text"]}


Run locally:

uvicorn run_api:app --reload


Now you have an API at http://127.0.0.1:8000/predict.

📦 Requirements

Your requirements.txt should include:

transformers>=4.44.0
datasets>=2.19.0
accelerate>=0.33.0
pandas>=2.0.0
torch>=2.2.0

📝 Notes for Beginners

Training on a laptop works but may be slow.

Use a small model (distilgpt2) for learning.

Bigger models need GPUs (Google Colab, Kaggle, or paid cloud).

Always check dataset formatting errors if training crashes.

🤝 Contributing

If you’re learning, try:

Adding new instruction–response pairs to the dataset.

Training longer epochs.

Swapping distilgpt2 for a larger model like gpt2-medium.

📜 License

MIT License — feel free to use and modify.
