from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
# revision = "2024-03-13"
local_model_path = (
    # "/home/karl/Documents/Github/label-recognizer/moondream/pretrained_weights"
    # "/home/karl/Documents/Github/label-recognizer/checkpoints/pretrained_weights_03_13"
    "/home/karl/Documents/Github/label-recognizer/checkpoints/pretrained_weights_05_20"
    # "/home/wch/Documents/Github/label-recognizer/checkpoints/moondream2"
)

DTYPE = torch.float32
DEVICE = "cuda"

# Download and cache the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revision,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Save the model and tokenizer locally
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
