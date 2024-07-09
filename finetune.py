import math
import os
import sys
from PIL import Image

import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
CONTINUE = 1
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MD_REVISION = "2024-04-02"
EPOCHS = 2
# Number of samples to process in each batch. Set this to the highest value that doesn't cause an
# out-of-memory error. Decrease it if you're running out of memory. Batch size 8 currently uses around
# 15 GB of GPU memory during fine-tuning.
BATCH_SIZE = 8
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
LR = 3e-5
# Whether to use Weights and Biases for logging training metrics.
USE_WANDB = False
ANSWER_EOS = "<|endoftext|>"
# Number of tokens used to represent each image.
IMG_TOKENS = 729


class PriceTagDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        data = []
        for filename in os.listdir(f"./data_{split}/labels"):
            if filename.endswith(".json"):
                file_path = os.path.join(f"./data_{split}/labels", filename)
                filename_without_extension = os.path.splitext(filename)[0]
                image_filename = filename_without_extension + ".png"
                image_path = os.path.join(f"./data_{split}/images", image_filename)
                with open(file_path, "r") as json_file:
                    with Image.open(image_path) as image:
                        json_data = json_file.read()
                        data.append(
                            {
                                "image": image.convert("RGB"),
                                "qa": [
                                    {
                                        "question": 'Find average price for goods in each labels in these images, respond with following json format: ```{"name": <good name, without brand name>, "price": <average price, per L(liter), kg(kilogram) or st(piece)>, "unit": <L, kg or st>}```',
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
    "train": PriceTagDataset("train"),
    "val": PriceTagDataset("validation"),
    "test": PriceTagDataset("test"),
}

tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True
)

weights_path = "./moondream/pretrained_weights_03_13"

moondream = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/moondream-ft" if CONTINUE else "vikhyatk/moondream2",
    revision=MD_REVISION,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)


def collate_fn(batch):
    images = [sample["image"] for sample in batch]
    images = torch.stack(moondream.vision_encoder.preprocess(images))
    images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

    labels_acc = []
    tokens_acc = []

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

    return (
        images.to(dtype=DTYPE),
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss(batch):
    images, tokens, labels, attn_mask = batch

    images = images.to(DEVICE)
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder.encoder(images)
        img_embs = moondream.vision_encoder.projection(img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


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
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            loss = compute_loss(batch)
            loss.backward()

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

    if USE_WANDB:
        wandb.finish()

    moondream.save_pretrained("checkpoints/moondream-ft")


def test():
    for test_data in datasets["test"]:
        enc_image = moondream.encode_image(test_data["image"])
        print(
            moondream.answer_question(
                enc_image,
                'Find average price for goods in each labels in these images, respond with following json format: ```{"name": <good name, without brand name>, "price": <average price, per L(liter), kg(kilogram) or st(piece)>, "unit": <L, kg or st>}```',
                tokenizer,
            )
        )


if __name__ == "__main__":
    # call train or test according to the command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
