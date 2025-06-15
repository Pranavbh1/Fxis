import os
import random
import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import logging

load_dotenv()

hf_token=os.getenv("HF_TOKEN")

# Setup logging for debug purposes
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Fixed parameters for generation
SEED = 42
MAX_TOKENS = 10000
TEMPERATURE = 0.4
TOP_P = 0.9


# Step 1: Ensure deterministic behavior
def set_deterministic(seed: int, *, any_card: bool = False):
    """Set deterministic behavior across libraries (Python, NumPy, PyTorch, and CUDA)."""
    log.info(f"Setting deterministic mode with seed {seed}, any_card={any_card}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Enforce deterministic cuBLAS

    if any_card:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronize GPU operations

# Step 2: Define prompt processing functions
def make_prompt(prompt: str) -> str:
    """Formats the prompt for the LLM."""
    system_msg = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ You are a helpful AI assistant }}}}<|eot_id|>"
    user_msg = f"<|start_header_id|>user<|end_header_id|>\n{{{{ {prompt} }}}}<|eot_id|>"
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    return f"{system_msg}{user_msg}{assistant_start}"

def generate_responses(model: LLM, prompts: list, sampling_params: SamplingParams) -> dict:
    """Generate responses for a list of prompts using the LLM."""
    requests = [make_prompt(prompt) for prompt in prompts]
    responses = model.generate(requests, sampling_params, use_tqdm=True)
    return {
        prompt: response.outputs[0].text for prompt, response in zip(prompts, responses)
    }

# Step 3: Setup and download the model
def download_model(model_name: str):
    """Downloads the model from Hugging Face Hub."""
    log.info(f"Downloading model: {model_name}")
    return snapshot_download(repo_id=model_name,token=hf_token)

# Step 4: Load the LLM model
def load_model(model_name: str):
    local_dir = download_model(model_name)
    model = LLM(
        model=local_dir,
        tensor_parallel_size=torch.cuda.device_count(),  # Adjust for multiple GPUs
        max_model_len=6144,
        enforce_eager=True,  # Enable eager execution for compatibility
    )
    log.info(f"Model downloaded to: {local_dir}")
    return model

# Step 5: Solve the prompt
def solve_prompt(payload):
    """Generate responses based on the payload containing the prompt."""
    set_deterministic(SEED)

    # Create SamplingParams
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )

    log.info(f"SamplingParams: {sampling_params}")
    prompts = payload["prompt"].splitlines()
    log.info(f"Generating responses for {len(prompts)} prompts.")
    responses = generate_responses(model, prompts, sampling_params)
    return responses
