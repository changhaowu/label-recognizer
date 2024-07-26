# from label_recognizer.moondream.moondream import (
# from ..moondream.moondream import Moondream
# 打印 PYTHONPATH 环境变量
import sys
import os
import argparse

print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

# 打印 sys.path
print(f"sys.path: {sys.path}")

# 确认当前工作目录
print(f"Current working directory: {os.getcwd()}")


from moondream.moondream import Moondream
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import os
from torch.utils.data import DataLoader, Dataset


DEVICE = "cuda"
# DTYPE = (
#     torch.float32 if DEVICE == "cpu" else torch.float16
# )  # CPU doesn't support float16
DTYPE = torch.float16

local_model_path = "checkpoints/pretrained_weights_05_20"

model_id = "vikhyatk/moondream2"
MD_REVISION = "2024-05-20"

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


def stock_label_loader(
    model,
    tokenizer,
    args,
):
    """
    Returns dataloaders for training, validation, and testing.

    Args:
        model (object): The model object.
        tokenizer (object): The tokenizer object.
        args (object): The arguments object.

    Returns:
        dict: A dictionary containing dataloaders for training, validation, and testing.
    """

    from PIL import Image
    import json

    ANSWER_EOS = "<|endoftext|>"
    IMG_TOKENS = 729
    DEVICE = "cuda"

    PROMPT = """
        Analyze the text in the provided image and extract the product name, price, and unit. Ensure the product name is accurately read from the image and not assumed. Follow these instructions precisely:

        1. Identify the product name, typically a recognizable item name found in the image.
        2. Determine the unit of measurement, which could be "kg", "L", or "st". If "kg" or "L" occur with "st", prefer to read "kg" or "L" rather than "st".
        3. Detect the price, associated with the identified unit (kg, L, or st). If "kg" or "L" are present in the image, use the price closest to these units. If neither "kg" nor "L" are present, then search for "st" and its respective price.

        Respond exclusively in the JSON format below, with no additional text or explanations. Include your response within `{ }`, then conclude your response with "<|endoftext|>". 
        The output should only be one combination of the most likely product name, price, and unit. Example format:

        ```json
        {
            "name": "Ryggfilé Alaska Pollock",
            "avg_price": "133",
            "unit": "kg"
        }
        <|endoftext|>
        """

    class PriceTagDataset(Dataset):

        def __init__(self, base_dir=None, split="train"):
            super().__init__()
            data = []

            if base_dir is None:
                base_dir = "./"

            labels_dir = os.path.join(base_dir, f"data_{split}", "labels")
            images_dir = os.path.join(base_dir, f"data_{split}", "images")

            for filename in os.listdir(labels_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(labels_dir, filename)
                    filename_without_extension = os.path.splitext(filename)[0]
                    image_filename = filename_without_extension + ".png"
                    image_path = os.path.join(images_dir, image_filename)

                    with open(file_path, "r") as json_file:
                        json_data = json.load(json_file)  # Read JSON data
                        with Image.open(image_path) as image:
                            data.append(
                                {
                                    "image": image.convert("RGB"),
                                    # "image": image.convert("L"),
                                    "qa": [
                                        {
                                            "question": PROMPT,
                                            "answer": json_data,
                                        }
                                    ],
                                }
                            )
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    datasets = {
        "train": PriceTagDataset(split="train"),
        "val": PriceTagDataset(split="validation"),
        "test": PriceTagDataset(split="test"),
    }

    def collate_fn(batch):
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            image = sample["image"]
            image = model.vision_encoder.preprocess(image)
            img_embs = model.vision_encoder(image.unsqueeze(0)).squeeze(0).to(DEVICE)

            # toks = [tokenizer.bos_token_id]
            toks = [
                model.text_model.get_input_embeddings()(
                    torch.tensor([tokenizer.bos_token_id], device=DEVICE)
                )
            ]
            # labs = [-100] * (IMG_TOKENS + 1)
            toks.append(img_embs)
            labs = [-100] * (
                img_embs.size(0) + 1
            )  # Adjusted to match image embedding size

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                q_t = torch.tensor(q_t).to(DEVICE)
                q_t_embeds = model.text_model.get_input_embeddings()(q_t)
                toks.append(q_t_embeds)
                # labs.extend([-100] * len(q_t))
                labs.extend([-100] * q_t_embeds.size(0))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
                ).input_ids
                a_t = torch.tensor(a_t).to(DEVICE)
                a_t_embeds = model.text_model.get_input_embeddings()(a_t)
                toks.append(a_t_embeds)
                labs.extend(a_t.tolist())

            # # Debug: Print the dimensions of each tensor in toks
            # for idx, t in enumerate(toks):
            #     print(f"toks[{idx}] shape: {t.shape}")

            toks = torch.cat(toks, dim=0)
            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i] = torch.cat(
                [
                    tokens_acc[i],
                    torch.zeros((pad_i, tokens_acc[i].size(1)), device=DEVICE),
                ],
                dim=0,
            )
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            torch.stack(tokens_acc),
            # torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        )

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.nsamples,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.nsamples,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.nsamples,
            collate_fn=collate_fn,
        ),
    }

    return dataloaders


# # 定义钩子函数来打印每一层的输出
# def print_layer_output(module, input, output):
#     print(f"Layer: {module.__class__.__name__}")
#     print(f"Input shape: {input[0].shape}")
#     print(
#         f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}"
#     )
#     # print(f"Output: {output[0] if isinstance(output, tuple) else output}")
#     print("-" * 50)


import logging

# 设置 logging
logging.basicConfig(level=logging.INFO, filename="model_output.log", filemode="w")
logger = logging.getLogger(__name__)


def print_layer_output(module, input, output):
    logger.info(f"Layer: {module.__class__.__name__}")
    logger.info(f"Input shape: {input[0].shape}")
    logger.info(
        f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}"
    )
    # 打印 CUDA 显存使用情况
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        free_memory = total_memory - reserved_memory

        logger.info(f"CUDA memory allocated: {allocated_memory / (1024 ** 2):.2f} MiB")
        logger.info(f"CUDA memory reserved: {reserved_memory / (1024 ** 2):.2f} MiB")
        logger.info(f"CUDA memory free: {free_memory / (1024 ** 2):.2f} MiB")
        logger.info(f"CUDA memory usage: {allocated_memory / total_memory:.2%}")

        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# 为模型的每一层注册钩子
for name, module in moondream.text_model.named_modules():
    module.register_forward_hook(print_layer_output)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLaMA model")
parser.add_argument(
    # "--nsamples", type=int, default=128, help="Number of calibration samples."
    "--nsamples",
    type=int,
    default=16,
    help="Number of calibration samples.",
)


parser.add_argument("--eval_zero_shot", action="store_true")
args = parser.parse_args()

train_dataloader = stock_label_loader(moondream, tokenizer, args)["train"]

batch = next(iter(train_dataloader))
tokens, labels = batch
token = tokens[0].unsqueeze(0)
label = labels[0].unsqueeze(0)

# 执行推理
with torch.no_grad():
    outputs = moondream.text_model(
        inputs_embeds=token,
        labels=label,
    )


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


# # Function to calculate parameter count and storage size
# def get_model_stats(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     total_size = sum(p.numel() * p.element_size() for p in model.parameters())
#     return total_params, total_size


# # Separate visual and textual parts if applicable
# # Assuming the model has `vision_model` and `text_model` attributes
# visual_params, visual_size = get_model_stats(moondream.vision_encoder)
# textual_params, textual_size = get_model_stats(moondream.text_model)

# # Print stats
# print(
#     f"Visual Model: Parameters: {visual_params:,}, Size: {visual_size / (1024 ** 2):.2f} MB"
# )
# print(
#     f"Textual Model: Parameters: {textual_params:,}, Size: {textual_size / (1024 ** 2):.2f} MB"
# )

# # Calculate tokenizer size
# tokenizer_files = [
#     "added_tokens.json",
#     "tokenizer_config.json",
#     "tokenizer.json",
#     "vocab.json",
#     "merges.txt",
# ]
# tokenizer_size = sum(
#     os.path.getsize(os.path.join(local_model_path, f))
#     for f in tokenizer_files
#     if os.path.exists(os.path.join(local_model_path, f))
# )

# print(f"Tokenizer Size: {tokenizer_size / (1024 ** 2):.2f} MB")
