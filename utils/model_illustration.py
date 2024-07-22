from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import os


DEVICE = "cuda"
DTYPE = (
    torch.float32 if DEVICE == "cpu" else torch.float16
)  # CPU doesn't support float16

local_model_path = "/path/to/local/directory"

model_id = "vikhyatk/moondream2"
MD_REVISION = "2024-03-13"

# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
# moondream = Moondream.from_pretrained(
#     model_id,
#     revision=LATEST_REVISION,
#     torch_dtype=dtype,
# ).to(device=device)

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision=MD_REVISION,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
moondream.eval()


# image1 = Image.open("assets/demo-1.jpg")
# image2 = Image.open("assets/demo-2.jpg")
# prompts = [
#     "What is the girl doing?",
#     "What color is the girl's hair?",
#     "What is this?",
#     "What is behind the stand?",
# ]

# answers = moondream.batch_answer(
#     images=[image1, image1, image2, image2],
#     prompts=prompts,
#     tokenizer=tokenizer,
# )

# for question, answer in zip(prompts, answers):
#     print(f"Q: {question}")
#     print(f"A: {answer}")
#     print()


# Function to calculate parameter count and storage size
def get_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_size


# Separate visual and textual parts if applicable
# Assuming the model has `vision_model` and `text_model` attributes
visual_params, visual_size = get_model_stats(moondream.vision_encoder)
textual_params, textual_size = get_model_stats(moondream.text_model)

# Print stats
print(
    f"Visual Model: Parameters: {visual_params:,}, Size: {visual_size / (1024 ** 2):.2f} MB"
)
print(
    f"Textual Model: Parameters: {textual_params:,}, Size: {textual_size / (1024 ** 2):.2f} MB"
)

# Calculate tokenizer size
tokenizer_files = [
    "added_tokens.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
]
tokenizer_size = sum(
    os.path.getsize(os.path.join(local_model_path, f))
    for f in tokenizer_files
    if os.path.exists(os.path.join(local_model_path, f))
)

print(f"Tokenizer Size: {tokenizer_size / (1024 ** 2):.2f} MB")
