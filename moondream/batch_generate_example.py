from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# device, dtype = detect_device()
DEVICE = "cuda"
DTYPE = (
    # torch.float32 if DEVICE == "cpu" else torch.float16
    torch.float32
    if DEVICE == "cpu"
    else torch.float32
)  # CPU doesn't support float16
MD_REVISION = "2024-05-20"

# Define the local path where the model and tokenizer are saved
local_model_path = (
    # "/home/karl/Documents/Github/label-recognizer/moondream/pretrained_weights"
    "/home/karl/Documents/Github/label-recognizer/moondream/pretrained_weights_03_13"
)

# model_id = "vikhyatk/moondream2"
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
# moondream = Moondream.from_pretrained(
#     model_id,
#     revision=LATEST_REVISION,
#     torch_dtype=dtype,
# ).to(device=device)

# Load the model and tokenizer from the local directory
moondream = Moondream.from_pretrained(
    local_model_path,
    use_safetensors=True,
    # attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
).to(device=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
moondream.eval()


image1 = Image.open("assets/demo-1.jpg")
image2 = Image.open("assets/demo-2.jpg")
prompts = [
    "What is the girl doing?",
    "What color is the girl's hair?",
    "What is this?",
    "What is behind the stand?",
]

answers = moondream.batch_answer(
    # images=[image1, image1, image2, image2],
    images=[image1, image1, image2, image2],
    prompts=prompts,
    tokenizer=tokenizer,
)

for question, answer in zip(prompts, answers):
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()
