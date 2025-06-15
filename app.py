import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model, solve_prompt
from huggingface_hub import snapshot_download
import uvicorn

# FastAPI app setup
app = FastAPI()

# Define the directory to store the model
MODEL_DIR = "models/Phi-3.5-mini-instruct"  # Folder to store the model

# Request model for payload
class PromptRequest(BaseModel):
    prompt: str  # Only the prompt is required

# Function to check if model exists and download if not
def check_and_download_model(model_name: str, model_dir: str):
    """Checks if model is present locally, and downloads it if not."""
    model_files = os.path.join(model_dir, "pytorch_model.bin")  # Example of a required file
    if not os.path.exists(model_files):  # Check for a key file to confirm the model is downloaded
        print(f"Model not found locally. Downloading model {model_name}...")
        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=model_dir)
        print(f"Model downloaded to: {model_dir}")
    else:
        print(f"Model already exists at: {model_dir}. Skipping download.")

# Load model when the app starts
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
check_and_download_model(MODEL_NAME, MODEL_DIR)

# Load model from the local directory
print("Loading the model from the local directory...")
model = load_model(MODEL_DIR)
print("Model loaded successfully.")

# FastAPI endpoint for prompt processing
@app.post("/generate-response/")
async def generate_response(request: PromptRequest):
    try:
        payload = {"prompts": request.prompt, "seed": 42, "max_tokens": 10000, "temperature": 0.4, "top_p": 0.9}
        print("Processing prompt...")
        responses = solve_prompt(payload)
        print("Prompt processed successfully.")
        return responses
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Add __main__ block to run the app
if __name__ == "__main__":
    print("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Set the port to 8000
