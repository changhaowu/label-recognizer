import math
import os
import sys
from PIL import Image
import json
import io

import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from moondream.moondream import Moondream, detect_device, LATEST_REVISION

DEVICE = "cuda"
CONTINUE = 1
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
# DTYPE = torch.float32
MD_REVISION = "2024-04-02"
EPOCHS = 1
# Number of samples to process in each batch. Set this to the highest value that doesn't cause an
# out-of-memory error. Decrease it if you're running out of memory. Batch size 8 currently uses around
# 15 GB of GPU memory during fine-tuning.
BATCH_SIZE = 1
# Number of batches to process before updating the model. You can use this to simulate a higher batch
# size than your GPU can handle. Set this to 1 to disable gradient accumulation.
GRAD_ACCUM_STEPS = 1
# Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
# of thumb, increase it by 1.4 times each time you double the effective batch size.
#
# Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
#
# Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
# training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
# cosine schedule.
LR = 3e-6
# LR = 3e-5
# Whether to use Weights and Biases for logging training metrics.
# USE_WANDB = True
USE_WANDB = False
ANSWER_EOS = "<|endoftext|>"
# Number of tokens used to represent each image.
IMG_TOKENS = 729

MAX_NEW_TOKENS = 128
# MODE = "TRAIN"
MODE = "reg"

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

# weights_path = "./checkpoints/pretrained_weights_05_20"
# tokenizer_path = "./checkpoints/pretrained_weights_05_20"

# weights_path = "checkpoints/moondream-ft_lr_5e-06_epoch_50"
# tokenizer_path = "checkpoints/moondream-ft_lr_5e-06_epoch_50"

weights_path = "checkpoints/moondream-ft_lr_3e-06_epoch_10"
tokenizer_path = "checkpoints/moondream-ft_lr_3e-06_epoch_10"

# weights_path = "checkpoints/lfs_raw"
# tokenizer_path = "checkpoints/lfs_raw"


tokenizer = AutoTokenizer.from_pretrained(
    # "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True
    pretrained_model_name_or_path=tokenizer_path,
    trust_remote_code=True,
)

moondream = Moondream.from_pretrained(
    # "./checkpoints/moondream-ft" if CONTINUE else "vikhyatk/moondream2",
    pretrained_model_name_or_path=weights_path if CONTINUE else "vikhyatk/moondream2",
    revision=MD_REVISION,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)

for param in moondream.parameters():
    if not param.requires_grad:
        print(f"Parameter {param} does not require grad")


def collate_fn(batch):
    images = [sample["image"] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]

    labels_acc = []
    tokens_acc = []
    ground_truth_answers = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample["qa"]:
            q_t = tokenizer(
                f"\n\nQuestion: {qa['question']}\n\nAnswer:", add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(
                f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

            ground_truth_answers.append(
                qa["answer"]
            )  # Collect unencoded ground truth answers

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
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    print("padding_i: ", pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        ground_truth_answers,  # Return unencoded ground truth answers
    )


def convert_to_numeric(decoded_answers, ground_truth_answers):
    numeric_data = []

    for i, decoded_answer in enumerate(decoded_answers):
        try:
            print("type?", type(decoded_answer), type(ground_truth_answers[i]))
            print("outcome?", decoded_answer, "ground_truth?", ground_truth_answers[i])

            response = json.loads(decoded_answer)
            ground_truth = ground_truth_answers[i]

            response_price = float(response["avg_price"].replace(".", ""))
            ground_truth_price = float(ground_truth["avg_price"].replace(".", ""))

            response_unit = len(response["unit"])
            ground_truth_unit = len(ground_truth["unit"])

            response_name = sum([ord(char) for char in response["name"]])
            ground_truth_name = sum([ord(char) for char in ground_truth["name"]])

            numeric_data.append(
                [
                    response_price,
                    ground_truth_price,
                    response_unit,
                    ground_truth_unit,
                    response_name,
                    ground_truth_name,
                ]
            )
        except json.JSONDecodeError:
            numeric_data.append([0, 0, 0, 0, 0, 0])

    return numeric_data


def custom_loss(numeric_tensor):
    reg_loss = 0

    for entry in numeric_tensor:
        (
            response_price,
            ground_truth_price,
            response_unit,
            ground_truth_unit,
            response_name,
            ground_truth_name,
        ) = entry

        # Calculate penalties
        if response_unit != ground_truth_unit:
            reg_loss += 10

        response_price_str = str(int(response_price))
        ground_truth_price_str = str(int(ground_truth_price))

        # Compare lengths
        if len(response_price_str) != len(ground_truth_price_str):
            reg_loss += 5
            if len(response_price_str) < len(ground_truth_price_str):
                response_price_str = response_price_str.zfill(
                    len(ground_truth_price_str)
                )
            else:
                ground_truth_price_str = ground_truth_price_str.zfill(
                    len(response_price_str)
                )

        # Compare digit by digit
        for gt_digit, res_digit in zip(ground_truth_price_str, response_price_str):
            if gt_digit != res_digit:
                reg_loss += 3

        # Compare the decimal points
        response_decimal_length = (
            len(str(response_price).split(".")[1]) if "." in str(response_price) else 0
        )
        ground_truth_decimal_length = (
            len(str(ground_truth_price).split(".")[1])
            if "." in str(ground_truth_price)
            else 0
        )
        if response_decimal_length != ground_truth_decimal_length:
            reg_loss += 3

        # # Optionally check the name (not requested but can be added)
        # if response_name != ground_truth_name:
        #     reg_loss += 1

        print(
            "checkpoint",
            response_price,
            ground_truth_price,
            response_unit,
            ground_truth_unit,
            response_name,
            ground_truth_name,
        )

    return torch.tensor(reg_loss / 10, dtype=torch.float32, requires_grad=True).to(
        DEVICE
    )


def decode_answer(
    inputs_embeds,
    tokenizer,
    attn_mask,
    result_queue=None,
    **kwargs,
):

    generate_config = {
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.bos_token_id,
        "max_new_tokens": MAX_NEW_TOKENS,
        **kwargs,
    }

    # print("inputs_embeds shape", inputs_embeds.unsqueeze(0).shape)
    # print("inputs_ids", inputs_embeds)
    # print("attn_mask shape", attn_mask.shape)

    moondream.text_model.transformer.gradient_checkpointing_enable()

    output_ids = moondream.text_model.generate(
        inputs_embeds=inputs_embeds.unsqueeze(0),
        attention_mask=attn_mask,
        **generate_config,
    )

    # print("output_ids", output_ids)
    # print("output_ids shape", output_ids.shape)

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    cleaned_answer = answer.strip()

    # print("cleaned_answer", cleaned_answer)

    # Use the result_queue to pass the result if it is provided
    if result_queue:
        result_queue.put(cleaned_answer)
    else:
        return cleaned_answer


def compute_loss(batch):

    images, tokens, labels, attn_mask, ground_truth_answers = batch

    moondream.text_model.transformer.gradient_checkpointing_enable()

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder(images)

    # print("img_embs shape", img_embs.shape)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    print("inputs_embeds shape", inputs_embeds.shape)
    print("attn_mask shape", attn_mask.shape)

    # outputs = moondream.text_model(
    #     inputs_embeds=inputs_embeds,
    #     labels=labels,
    #     attention_mask=attn_mask,
    # )

    if MODE == "reg":

        decoded_answers = []
        for _, input_embeds in enumerate(inputs_embeds):
            # print("input_embeds shape", input_embeds.shape)
            decoded_answers.append(
                decode_answer(
                    input_embeds,
                    tokenizer,
                    attn_mask,
                )
            )

        numeric_data = convert_to_numeric(decoded_answers, ground_truth_answers)
        numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32).to(DEVICE)
        reg_loss = custom_loss(numeric_tensor)
    else:
        reg_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(DEVICE)

    return outputs.loss, reg_loss


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


dataloaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    ),
    "val": DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    ),
    "test": DataLoader(
        datasets["test"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    ),
}


def train():
    moondream.text_model.train()
    moondream.text_model.transformer.gradient_checkpointing_enable()

    total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
    optimizer = torch.optim.Adam(
        [
            {"params": moondream.text_model.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    if USE_WANDB:
        import wandb

        wandb.init(
            project="moondream-ft",
            config={
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
                "LR": LR,
            },
        )

    i = 0
    for epoch in range(EPOCHS):
        epoch_ce_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            # print("the batch:", i)
            ce_loss, reg_loss = compute_loss(batch)
            loss = ce_loss + reg_loss * torch.norm(ce_loss)
            loss.backward()

            # 累积损失
            epoch_ce_loss += ce_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_steps += 1

            # print(
            #     f"Epoch {epoch + 1}/{EPOCHS}, CE Loss: {epoch_ce_loss}, Reg Loss: {epoch_reg_loss}"
            # )

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if i % 10 == 0 and USE_WANDB:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(dataloaders["val"])

            if USE_WANDB:
                wandb.log(
                    {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                    | ({"loss/val": val_loss} if i % 10 == 0 else {})
                )

        # 计算每个 epoch 的平均损失
        avg_ce_loss = epoch_ce_loss / epoch_steps
        avg_reg_loss = epoch_reg_loss / epoch_steps

        # 打印每个 epoch 的损失
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Avg CE Loss: {avg_ce_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}"
        )

    if USE_WANDB:
        wandb.finish()

    # Save the model and tokenizer locally
    moondream.save_pretrained(f"checkpoints/moondream-ft_lr_{LR}_epoch_{EPOCHS}")
    tokenizer.save_pretrained(f"checkpoints/moondream-ft_lr_{LR}_epoch_{EPOCHS}")


def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # 使用适当的图像格式
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


import random


def test():
    test_images = set()
    for i, test_data in enumerate(datasets["val"]):
        img_bytes = image_to_bytes(test_data["image"])
        if img_bytes in test_images:
            print(f"Duplicate found at index {i}")
            continue
        test_images.add(img_bytes)

        enc_image = moondream.encode_image(test_data["image"])

        response = moondream.answer_question(
            enc_image,
            PROMPT,
            tokenizer,
        )

        print(f"Response for image {i}: {response}")
        for item in test_data["qa"]:
            print("Ground Truth: \n", json.dumps(item["answer"], indent=4))

        break


if __name__ == "__main__":
    # call train or test according to the command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
